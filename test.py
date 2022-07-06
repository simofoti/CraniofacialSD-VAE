import os
import json
import pickle
import tqdm
import trimesh
import torch.nn
import pytorch3d.loss

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.io import write_video
from torchvision.utils import make_grid, save_image
from pytorch3d.renderer import BlendParams
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_points
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets
from utils import create_alpha_cmap, plot_confusion_matrix


class Tester:
    def __init__(self, model_manager, norm_dict,
                 train_load, test_load, out_dir, config):
        self._manager = model_manager
        self._manager.eval()
        self._device = model_manager.device
        self._norm_dict = norm_dict
        self._normalized_data = config['data']['normalize_data']
        self._out_dir = out_dir
        self._config = config
        self._train_loader = train_load
        self._test_loader = test_load
        self._is_vae = self._manager.is_vae
        self._is_rae = self._manager.is_rae
        self.latent_stats = self.compute_latent_stats(train_load)

        self.coma_landmarks = [
            1337, 1344, 1163, 878, 3632, 2496, 2428, 2291, 2747,
            3564, 1611, 2715, 3541, 1576, 3503, 3400, 3568, 1519,
            203, 183, 870, 900, 867, 3536]
        self.uhm_landmarks = [
            10754, 10826, 9123, 10667, 19674, 28739, 4831, 19585,
            8003, 22260, 12492, 27386, 1969, 31925, 31158, 20963,
            1255, 9881, 32055, 45778, 5355, 27515, 18482, 33691]

    def __call__(self):
        self.set_renderings_size(256)
        self.set_rendering_background_color([1, 1, 1])

        # Qualitative evaluations
        sns.set_theme(style="ticks")
        self.per_variable_range_experiments(use_z_stats=False)
        self.plot_embeddings()
        if self._config['data']['swap_features']:
            self.latent_swapping(next(iter(self._test_loader)).x)
        self.random_generation_and_rendering(n_samples=16)
        self.random_generation_and_save(n_samples=16)
        self.interpolate()
        if self._config['data']['dataset_type'] == 'faces':
            self.direct_manipulation()

        # Quantitative evaluation
        self.evaluate_gen(self._test_loader, n_sampled_points=2048)
        recon_errors = self.reconstruction_errors(self._test_loader)
        train_set_diversity = self.compute_diversity_train_set()
        diversity = self.compute_diversity()
        specificity = self.compute_specificity()
        metrics = {'recon_errors': recon_errors,
                   'train_set_diversity': train_set_diversity,
                   'diversity': diversity,
                   'specificity': specificity}

        outfile_path = os.path.join(self._out_dir, 'eval_metrics.json')
        with open(outfile_path, 'w') as outfile:
            json.dump(metrics, outfile)

    def _unnormalize_verts(self, verts, dev=None):
        d = self._device if dev is None else dev
        return verts * self._norm_dict['std'].to(d) + \
            self._norm_dict['mean'].to(d)

    def set_renderings_size(self, size):
        self._manager.renderer.rasterizer.raster_settings.image_size = size

    def set_rendering_background_color(self, color=None):
        color = [1, 1, 1] if color is None else color
        blend_params = BlendParams(background_color=color)
        self._manager.default_shader.blend_params = blend_params
        self._manager.simple_shader.blend_params = blend_params

    def compute_latent_stats(self, data_loader):
        storage_path = os.path.join(self._out_dir, 'z_stats.pkl')
        try:
            with open(storage_path, 'rb') as file:
                z_stats = pickle.load(file)
        except FileNotFoundError:
            latents_list = []
            for data in tqdm.tqdm(data_loader):
                if self._config['data']['swap_features']:
                    data.x = data.x[self._manager.batch_diagonal_idx, ::]
                latents_list.append(self._manager.encode(
                    data.x.to(self._device)).detach().cpu())
            latents = torch.cat(latents_list, dim=0)
            z_means = torch.mean(latents, dim=0)
            z_stds = torch.std(latents, dim=0)
            z_mins, _ = torch.min(latents, dim=0)
            z_maxs, _ = torch.max(latents, dim=0)
            z_stats = {'means': z_means, 'stds': z_stds,
                       'mins': z_mins, 'maxs': z_maxs}

            with open(storage_path, 'wb') as file:
                pickle.dump(z_stats, file)
        return z_stats

    @staticmethod
    def string_to_color(rgba_string, swap_bw=True):
        rgba_string = rgba_string[1:-1]  # remove [ and ]
        rgb_values = rgba_string.split()[:-1]
        colors = [int(c) / 255 for c in rgb_values]
        if colors == [1., 1., 1.] and swap_bw:
            colors = [0., 0., 0.]
        return tuple(colors)

    def per_variable_range_experiments(self, z_range_multiplier=1,
                                       use_z_stats=True, save_suffix=None):
        if self._is_vae and not use_z_stats:
            latent_size = self._manager.model_latent_size
            z_means = torch.zeros(latent_size)
            z_mins = -3 * z_range_multiplier * torch.ones(latent_size)
            z_maxs = 3 * z_range_multiplier * torch.ones(latent_size)
        else:
            z_means = self.latent_stats['means']
            z_mins = self.latent_stats['mins'] * z_range_multiplier
            z_maxs = self.latent_stats['maxs'] * z_range_multiplier

        # Create video perturbing each latent variable from min to max.
        # Show generated mesh and error map next to each other
        # Frames are all concatenated along the same direction. A black frame is
        # added before start perturbing the next latent variable
        n_steps = 10
        all_frames, all_rendered_differences, max_distances = [], [], []
        all_renderings = []
        for i in tqdm.tqdm(range(z_means.shape[0])):
            z = z_means.repeat(n_steps, 1)
            z[:, i] = torch.linspace(
                z_mins[i], z_maxs[i], n_steps).to(self._device)

            gen_verts = self._manager.generate(z.to(self._device))

            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            differences_from_first = self._manager.compute_vertex_errors(
                gen_verts, gen_verts[0].expand(gen_verts.shape[0], -1, -1))
            max_distances.append(differences_from_first[-1, ::])
            renderings = self._manager.render(gen_verts).detach().cpu()
            all_renderings.append(renderings)
            differences_renderings = self._manager.render(
                gen_verts, differences_from_first,
                error_max_scale=5).cpu().detach()
            all_rendered_differences.append(differences_renderings)
            frames = torch.cat([renderings, differences_renderings], dim=-1)
            all_frames.append(
                torch.cat([frames, torch.zeros_like(frames)[:2, ::]]))

        s = save_suffix if save_suffix is not None else ''
        write_video(
            os.path.join(self._out_dir, f'latent_exploration{s}.mp4'),
            torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

        # Same video as before, but effects of perturbing each latent variables
        # are shown in the same frame. Only error maps are shown.
        grid_frames = []
        grid_nrows = 8
        if self._config['data']['swap_features']:
            z_size = self._config['model']['latent_size']
            grid_nrows = z_size // len(self._manager.latent_regions)

        stacked_frames = torch.stack(all_rendered_differences)
        for i in range(stacked_frames.shape[1]):
            grid_frames.append(
                make_grid(stacked_frames[:, i, ::], padding=10,
                          pad_value=1, nrow=grid_nrows))
        save_image(grid_frames[-1],
                   os.path.join(self._out_dir,
                                f'latent_exploration_tiled{s}.png'))
        write_video(
            os.path.join(self._out_dir, f'latent_exploration_tiled{s}.mp4'),
            torch.stack(grid_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=1)

        # Same as before, but only output meshes are used
        stacked_frames_meshes = torch.stack(all_renderings)
        grid_frames_m = []
        for i in range(stacked_frames_meshes.shape[1]):
            grid_frames_m.append(
                make_grid(stacked_frames_meshes[:, i, ::], padding=10,
                          pad_value=1, nrow=grid_nrows))
        write_video(
            os.path.join(self._out_dir, f'latent_exploration_out_tiled{s}.mp4'),
            torch.stack(grid_frames_m, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

        # Create a plot showing the effects of perturbing latent variables in
        # each region of the face
        df = pd.DataFrame(columns=['mean_dist', 'z_var', 'region'])
        df_row = 0
        for zi, vert_distances in enumerate(max_distances):
            for region, indices in self._manager.template.feat_and_cont.items():
                regional_distances = vert_distances[indices['feature']]
                mean_regional_distance = torch.mean(regional_distances)
                df.loc[df_row] = [mean_regional_distance.item(), zi, region]
                df_row += 1

        sns.set_theme(style="ticks")
        palette = {k: self.string_to_color(k) for k in
                   self._manager.template.feat_and_cont.keys()}
        grid = sns.FacetGrid(df, col="region", hue="region", palette=palette,
                             col_wrap=4, height=3)

        grid.map(plt.plot, "z_var", "mean_dist", marker="o")
        plt.savefig(os.path.join(self._out_dir,
                                 f'latent_exploration_split{s}.svg'))

        sns.relplot(data=df, kind="line", x="z_var", y="mean_dist",
                    hue="region", palette=palette)
        plt.savefig(os.path.join(self._out_dir, f'latent_exploration{s}.svg'))

    def random_latent(self, n_samples, z_range_multiplier=1):
        if self._is_vae:  # sample from normal distribution if vae
            z = torch.randn([n_samples, self._manager.model_latent_size])
        elif self._is_rae:
            z = self._manager.sample_gaussian_mixture(n_samples)
        else:
            z_means = self.latent_stats['means']
            z_mins = self.latent_stats['mins'] * z_range_multiplier
            z_maxs = self.latent_stats['maxs'] * z_range_multiplier

            uniform = torch.rand([n_samples, z_means.shape[0]],
                                 device=z_means.device)
            z = uniform * (z_maxs - z_mins) + z_mins
        return z

    def random_generation(self, n_samples=16, z_range_multiplier=1,
                          denormalize=True):
        z = self.random_latent(n_samples, z_range_multiplier)
        gen_verts = self._manager.generate(z.to(self._device))
        if self._normalized_data and denormalize:
            gen_verts = self._unnormalize_verts(gen_verts)
        return gen_verts

    def random_generation_and_rendering(self, n_samples=16,
                                        z_range_multiplier=1):
        gen_verts = self.random_generation(n_samples, z_range_multiplier)
        renderings = self._manager.render(gen_verts).cpu()
        grid = make_grid(renderings, padding=10, pad_value=1)
        save_image(grid, os.path.join(self._out_dir, 'random_generation.png'))

    def random_generation_and_save(self, n_samples=16, z_range_multiplier=1):
        out_mesh_dir = os.path.join(self._out_dir, 'random_meshes')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)

        gen_verts = self.random_generation(n_samples, z_range_multiplier)

        self.save_batch(gen_verts, out_mesh_dir)

    def save_batch(self, batch_verts, out_mesh_dir):
        for i in range(batch_verts.shape[0]):
            mesh = trimesh.Trimesh(
                batch_verts[i, ::].cpu().detach().numpy(),
                self._manager.template.face.t().cpu().numpy())
            mesh.export(os.path.join(out_mesh_dir, str(i) + '.ply'))

    def reconstruction_errors(self, data_loader):
        print('Compute reconstruction errors')
        data_errors = []
        for data in tqdm.tqdm(data_loader):
            if self._config['data']['swap_features']:
                data.x = data.x[self._manager.batch_diagonal_idx, ::]
            data = data.to(self._device)
            gt = data.x

            recon = self._manager.forward(data)[0]

            if self._normalized_data:
                gt = self._unnormalize_verts(gt)
                recon = self._unnormalize_verts(recon)

            errors = self._manager.compute_vertex_errors(recon, gt)
            data_errors.append(torch.mean(errors.detach(), dim=1))
        data_errors = torch.cat(data_errors, dim=0)
        return {'mean': torch.mean(data_errors).item(),
                'median': torch.median(data_errors).item(),
                'max': torch.max(data_errors).item()}

    def compute_diversity_train_set(self):
        print('Computing train set diversity')
        previous_verts_batch = None
        mean_distances = []
        for data in tqdm.tqdm(self._train_loader):
            if self._config['data']['swap_features']:
                x = data.x[self._manager.batch_diagonal_idx, ::]
            else:
                x = data.x

            current_verts_batch = x
            if self._normalized_data:
                current_verts_batch = self._unnormalize_verts(
                    current_verts_batch, x.device)

            if previous_verts_batch is not None:
                verts_batch_distances = self._manager.compute_vertex_errors(
                    previous_verts_batch, current_verts_batch)
                mean_distances.append(torch.mean(verts_batch_distances, dim=1))
            previous_verts_batch = current_verts_batch
        return torch.mean(torch.cat(mean_distances, dim=0)).item()

    def compute_diversity(self, n_samples=10000):
        print('Computing generative model diversity')
        samples_per_batch = 20
        mean_distances = []
        for _ in tqdm.tqdm(range(n_samples // samples_per_batch)):
            verts_batch_distances = self._manager.compute_vertex_errors(
                self.random_generation(samples_per_batch),
                self.random_generation(samples_per_batch))
            mean_distances.append(torch.mean(verts_batch_distances, dim=1))
        return torch.mean(torch.cat(mean_distances, dim=0)).item()

    def compute_specificity(self, n_samples=100):
        print('Computing generative model specificity')
        min_distances = []
        for _ in tqdm.tqdm(range(n_samples)):
            sample = self.random_generation(1)

            mean_distances = []
            for data in self._train_loader:
                if self._config['data']['swap_features']:
                    x = data.x[self._manager.batch_diagonal_idx, ::]
                else:
                    x = data.x

                if self._normalized_data:
                    x = self._unnormalize_verts(x.to(self._device))
                else:
                    x = x.to(self._device)

                v_dist = self._manager.compute_vertex_errors(
                    x, sample.expand(x.shape[0], -1, -1))
                mean_distances.append(torch.mean(v_dist, dim=1))
            min_distances.append(torch.min(torch.cat(mean_distances, dim=0)))
        return torch.mean(torch.stack(min_distances)).item()

    def evaluate_gen(self, data_loader, n_sampled_points=None):
        all_sample = []
        all_ref = []
        for data in tqdm.tqdm(data_loader):
            if self._config['data']['swap_features']:
                data.x = data.x[self._manager.batch_diagonal_idx, ::]
            data = data.to(self._device)
            if self._normalized_data:
                data.x = self._unnormalize_verts(data.x)

            ref = data.x
            sample = self.random_generation(data.x.shape[0])

            if n_sampled_points is not None:
                subset_idxs = np.random.choice(ref.shape[1], n_sampled_points)
                ref = ref[:, subset_idxs]
                sample = sample[:, subset_idxs]

            all_ref.append(ref)
            all_sample.append(sample)

        sample_pcs = torch.cat(all_sample, dim=0)
        ref_pcs = torch.cat(all_ref, dim=0)
        print("Generation sample size:%s reference size: %s"
              % (sample_pcs.size(), ref_pcs.size()))

        # Compute metrics
        metrics = compute_all_metrics(
            sample_pcs, ref_pcs, self._config['optimization']['batch_size'])
        metrics = {k: (v.cpu().detach().item()
                       if not isinstance(v, float) else v) for k, v in
                   metrics.items()}
        print(metrics)

        sample_pcl_npy = sample_pcs.cpu().detach().numpy()
        ref_pcl_npy = ref_pcs.cpu().detach().numpy()
        jsd = jsd_between_point_cloud_sets(sample_pcl_npy, ref_pcl_npy)
        print("JSD:%s" % jsd)
        metrics["jsd"] = jsd

        outfile_path = os.path.join(self._out_dir, 'eval_metrics_gen.json')
        with open(outfile_path, 'w') as outfile:
            json.dump(metrics, outfile)

    def latent_swapping(self, v_batch=None):
        if v_batch is None:
            v_batch = self.random_generation(2, denormalize=False)
        else:
            assert v_batch.shape[0] >= 2
            v_batch = v_batch.to(self._device)
            if self._config['data']['swap_features']:
                v_batch = v_batch[self._manager.batch_diagonal_idx, ::]
            v_batch = v_batch[:2, ::]

        z = self._manager.encode(v_batch)
        z_0, z_1 = z[0, ::], z[1, ::]

        swapped_verts = []
        for key, z_region in self._manager.latent_regions.items():
            z_swap = z_0.clone()
            z_swap[z_region[0]:z_region[1]] = z_1[z_region[0]:z_region[1]]
            swapped_verts.append(self._manager.generate(z_swap))

        all_verts = torch.cat([v_batch, torch.cat(swapped_verts, dim=0)], dim=0)

        if self._normalized_data:
            all_verts = self._unnormalize_verts(all_verts)

        out_mesh_dir = os.path.join(self._out_dir, 'latent_swapping')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)
        self.save_batch(all_verts, out_mesh_dir)

        source_dist = self._manager.compute_vertex_errors(
            all_verts, all_verts[0, ::].expand(all_verts.shape[0], -1, -1))
        target_dist = self._manager.compute_vertex_errors(
            all_verts, all_verts[1, ::].expand(all_verts.shape[0], -1, -1))

        renderings = self._manager.render(all_verts)
        renderings_source = self._manager.render(all_verts, source_dist, 5)
        renderings_target = self._manager.render(all_verts, target_dist, 5)
        grid = make_grid(torch.cat(
            [renderings, renderings_source, renderings_target], dim=-2),
            padding=10, pad_value=1, nrow=renderings.shape[0])
        save_image(grid, os.path.join(out_mesh_dir, 'latent_swapping.png'))

    def fit_vertices(self, target_verts, lr=5e-3, iterations=250,
                     target_noise=0, target_landmarks=None):
        # Scale and position target_verts
        target_verts = target_verts.unsqueeze(0).to(self._device)
        if target_landmarks is None:
            target_landmarks = target_verts[:, self.coma_landmarks, :]
        target_landmarks = target_landmarks.to(self._device)

        if target_noise > 0:
            target_verts = target_verts + (torch.randn_like(target_verts) *
                                           target_noise /
                                           self._manager.to_mm_const)
            target_landmarks = target_landmarks + (
                torch.randn_like(target_landmarks) *
                target_noise / self._manager.to_mm_const)

        z = self.latent_stats['means'].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr)
        gen_verts = None
        for i in range(iterations):
            optimizer.zero_grad()
            gen_verts = self._manager.generate_for_opt(z.to(self._device))
            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            if i < iterations // 3:
                er = self._manager.compute_mse_loss(
                    gen_verts[:, self.uhm_landmarks, :], target_landmarks)
            else:
                er, _ = pytorch3d.loss.chamfer_distance(gen_verts, target_verts)

            er.backward()
            optimizer.step()
        return gen_verts, target_verts.squeeze()

    def fit_coma_data(self, base_dir='meshes2fit',
                      noise=0, export_meshes=False):
        print(f"Fitting CoMA meshes with noise = {noise} mm")
        out_mesh_dir = os.path.join(self._out_dir, 'fitting')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)

        names_and_scale = {}
        for dirpath, _, fnames in os.walk(base_dir):
            for f in fnames:
                if f.endswith('.ply'):
                    if f[:5] in ['03274', '03275', '00128', '03277']:
                        names_and_scale[f] = 9
                    else:
                        names_and_scale[f] = 8

        dataframes = []
        for m_id, scale in tqdm.tqdm(names_and_scale.items()):
            df_id = m_id.split('.')[0]
            subd = False
            mesh_path = os.path.join(base_dir, m_id)
            target_mesh = trimesh.load_mesh(mesh_path, 'ply', process=False)
            target_verts = torch.tensor(
                target_mesh.vertices, dtype=torch.float,
                requires_grad=False, device=self._device)

            # scale and translate to match template. Values manually computed
            target_verts *= scale
            target_verts[:, 1] += 0.15

            # If target mesh was subdivided use original target to retrieve its
            # landmarks
            target_landmarks = None
            if 'subd' in m_id:
                subd = True
                df_id = m_id.split('_')[0]
                base_path = os.path.join(base_dir, m_id.split('_')[0] + '.ply')
                base_mesh = trimesh.load_mesh(base_path, 'ply', process=False)
                base_verts = torch.tensor(
                    base_mesh.vertices, dtype=torch.float,
                    requires_grad=False, device=self._device)
                target_landmarks = base_verts[self.coma_landmarks, :]
                target_landmarks = target_landmarks.unsqueeze(0)
                target_landmarks *= scale
                target_landmarks[:, 1] += 0.15

            out_verts, t_verts = self.fit_vertices(
                target_verts, target_noise=noise,
                target_landmarks=target_landmarks)

            closest_p_errors = self._manager.to_mm_const * \
                self._dist_closest_point(out_verts, target_verts.unsqueeze(0))

            dataframes.append(pd.DataFrame(
                {'id': df_id, 'noise': noise, 'subdivided': subd,
                 'errors': closest_p_errors.squeeze().detach().cpu().numpy()}))

            if export_meshes:
                mesh_name = m_id.split('.')[0]
                out_mesh = trimesh.Trimesh(
                    out_verts[0, ::].cpu().detach().numpy(),
                    self._manager.template.face.t().cpu().numpy())
                out_mesh.export(os.path.join(
                    out_mesh_dir, mesh_name + f"_fit_{str(noise)}" + '.ply'))
                target_mesh.vertices = t_verts.detach().cpu().numpy()
                target_mesh.export(os.path.join(
                    out_mesh_dir, mesh_name + f"_t_{str(noise)}" + '.ply'))
        return pd.concat(dataframes)

    def fit_coma_data_different_noises(self, base_dir='meshes2fit'):
        noises = [0, 2, 4, 6, 8]
        dataframes = []
        for n in noises:
            dataframes.append(self.fit_coma_data(base_dir, n, True))
        df = pd.concat(dataframes)
        df.to_pickle(os.path.join(self._out_dir, 'coma_fitting.pkl'))

        sns.set_theme(style="ticks")
        plt.figure()
        sns.lineplot(data=df, x='noise', y='errors',
                     markers=True, dashes=False, ci='sd')
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting.svg'))

        plt.figure()
        sns.boxplot(data=df, x='noise', y='errors', showfliers=False)
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting_box.svg'))

        plt.figure()
        sns.violinplot(data=df[df.errors < 3], x='noise', y='errors',
                       split=False)
        plt.savefig(os.path.join(self._out_dir, 'coma_fitting_violin.svg'))

    @staticmethod
    def _point_mesh_distance(points, verts, faces):
        points = points.squeeze()
        verts_packed = verts.to(points.device)
        faces_packed = torch.tensor(faces, device=points.device).t()
        first_idx = torch.tensor([0], device=points.device)

        tris = verts_packed[faces_packed]

        point_to_face = point_face_distance(points, first_idx, tris,
                                            first_idx, points.shape[0])
        return point_to_face / points.shape[0]

    @staticmethod
    def _dist_closest_point(x, y):
        # for each point on x return distance to closest point in y
        x, x_lengths, x_normals = _handle_pointcloud_input(x, None, None)
        y, y_lengths, y_normals = _handle_pointcloud_input(y, None, None)
        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
        cham_x = x_nn.dists[..., 0]
        return cham_x

    def direct_manipulation(self, z=None, indices=None, new_coords=None,
                            lr=0.1, iterations=50, affect_only_zf=True):
        if z is None:
            z = self.latent_stats['means'].unsqueeze(0)
            # z = self.random_latent(1)
            z = z.clone().detach().requires_grad_(True)
        if indices is None and new_coords is None:
            indices = [8816, 8069, 8808]
            new_coords = torch.tensor([[-0.0108174, 0.0814601, 0.664498],
                                       [-0.1821480, 0.0190682, 0.419531],
                                       [-0.0096422, 0.3058790, 0.465528]])
        new_coords = new_coords.unsqueeze(0).to(self._device)

        colors = self._manager.template.colors.to(torch.long)
        features = [str(colors[i].cpu().detach().numpy()) for i in indices]
        assert all(x == features[0] for x in features)

        zf_idxs = self._manager.latent_regions[features[0]]

        optimizer = torch.optim.Adam([z], lr)
        initial_verts = self._manager.generate_for_opt(z.to(self._device))
        if self._normalized_data:
            initial_verts = self._unnormalize_verts(initial_verts)
        gen_verts = None
        for i in range(iterations):
            optimizer.zero_grad()
            gen_verts = self._manager.generate_for_opt(z.to(self._device))
            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            loss = self._manager.compute_mse_loss(
                gen_verts[:, indices, :], new_coords)
            loss.backward()

            if affect_only_zf:
                z.grad[:, :zf_idxs[0]] = 0
                z.grad[:, zf_idxs[1]:] = 0
            optimizer.step()

        # Save output meshes
        out_mesh_dir = os.path.join(self._out_dir, 'direct_manipulation')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)

        initial_mesh = trimesh.Trimesh(
            initial_verts[0, ::].cpu().detach().numpy(),
            self._manager.template.face.t().cpu().numpy())
        initial_mesh.export(os.path.join(out_mesh_dir, 'initial.ply'))

        new_mesh = trimesh.Trimesh(
            gen_verts[0, ::].cpu().detach().numpy(),
            self._manager.template.face.t().cpu().numpy())
        new_mesh.export(os.path.join(out_mesh_dir, 'new.ply'))

        for i, coords in zip(indices, new_coords[0, ::].detach().cpu().numpy()):
            sphere = trimesh.creation.icosphere(radius=0.01)
            sphere.vertices = sphere.vertices + coords
            sphere.export(os.path.join(out_mesh_dir, f'target_{i}.ply'))

            sphere = trimesh.creation.icosphere(radius=0.01)
            sphere.vertices += initial_verts[0, i, :].cpu().detach().numpy()
            sphere.export(os.path.join(out_mesh_dir, f'selected_{i}.ply'))

    def interpolate(self):
        with open(os.path.join('precomputed', 'data_split.json'), 'r') as fp:
            data = json.load(fp)
        test_list = data['test']
        meshes_root = self._test_loader.dataset.root

        # Pick first test mesh and find most different mesh in test set
        v_1 = None
        distances = [0]
        for i, fname in enumerate(test_list):
            mesh_path = os.path.join(meshes_root, fname + '.ply')
            mesh = trimesh.load_mesh(mesh_path, 'ply', process=False)
            mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                                      requires_grad=False, device='cpu')
            if i == 0:
                v_1 = mesh_verts
            else:
                distances.append(
                    self._manager.compute_mse_loss(v_1, mesh_verts).item())

        m_2_path = os.path.join(
            meshes_root, test_list[np.asarray(distances).argmax()] + '.ply')
        m_2 = trimesh.load_mesh(m_2_path, 'ply', process=False)
        v_2 = torch.tensor(m_2.vertices, dtype=torch.float, requires_grad=False)

        v_1 = (v_1 - self._norm_dict['mean']) / self._norm_dict['std']
        v_2 = (v_2 - self._norm_dict['mean']) / self._norm_dict['std']

        z_1 = self._manager.encode(v_1.unsqueeze(0).to(self._device))
        z_2 = self._manager.encode(v_2.unsqueeze(0).to(self._device))

        features = list(self._manager.template.feat_and_cont.keys())

        # Interpolate per feature
        if self._config['data']['swap_features']:
            z = z_1.repeat(len(features) // 2, 1)
            all_frames, rows = [], []
            for feature in features:
                zf_idxs = self._manager.latent_regions[feature]
                z_1f = z_1[:, zf_idxs[0]:zf_idxs[1]]
                z_2f = z_2[:, zf_idxs[0]:zf_idxs[1]]
                z[:, zf_idxs[0]:zf_idxs[1]] = self.vector_linspace(
                    z_1f, z_2f, len(features) // 2).to(self._device)

                gen_verts = self._manager.generate(z.to(self._device))
                if self._normalized_data:
                    gen_verts = self._unnormalize_verts(gen_verts)

                renderings = self._manager.render(gen_verts).cpu()
                all_frames.append(renderings)
                rows.append(make_grid(renderings, padding=10,
                            pad_value=1, nrow=len(features)))
                z = z[-1, :].repeat(len(features) // 2, 1)

            save_image(
                torch.cat(rows, dim=-2),
                os.path.join(self._out_dir, 'interpolate_per_feature.png'))
            write_video(
                os.path.join(self._out_dir, 'interpolate_per_feature.mp4'),
                torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

        # Interpolate per variable
        z = z_1.repeat(3, 1)
        all_frames = []
        for z_i in range(self._manager.model_latent_size):
            z_1f = z_1[:, z_i]
            z_2f = z_2[:, z_i]
            z[:, z_i] = torch.linspace(z_1f.item(),
                                       z_2f.item(), 3).to(self._device)

            gen_verts = self._manager.generate(z.to(self._device))
            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            renderings = self._manager.render(gen_verts).cpu()
            all_frames.append(renderings)
            z = z[-1, :].repeat(3, 1)

        write_video(
            os.path.join(self._out_dir, 'interpolate_per_variable.mp4'),
            torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)

        # Interpolate all features
        zs = self.vector_linspace(z_1, z_2, len(features))

        gen_verts = self._manager.generate(zs.to(self._device))
        if self._normalized_data:
            gen_verts = self._unnormalize_verts(gen_verts)

        renderings = self._manager.render(gen_verts).cpu()
        im = make_grid(renderings, padding=10, pad_value=1, nrow=len(features))
        save_image(im, os.path.join(self._out_dir, 'interpolate_all.png'))

    @staticmethod
    def vector_linspace(start, finish, steps):
        ls = []
        for s, f in zip(start[0], finish[0]):
            ls.append(torch.linspace(s, f, steps))
        res = torch.stack(ls)
        return res.t()

    def plot_embeddings(self, embedding_mode='lda'):
        tr_z, tr_l = self._manager.train_latents_and_labels
        if tr_z is None:
            tr_z, tr_l = self._manager.encode_all(self._train_loader, True)
        ts_z, ts_l = self._manager.encode_all(self._test_loader, False)
        tr_y = np.array(self._manager.class2idx(np.concatenate(tr_l['y'])))
        ts_y = np.array(self._manager.class2idx(np.concatenate(ts_l['y'])))

        tr_z_np = torch.cat(tr_z, dim=0).numpy()
        ts_z_np = torch.cat(ts_z, dim=0).numpy()

        if embedding_mode == 'lda':
            tr_z_np_2d = self._manager.lda_project_latents_in_2d(tr_z_np)
            ts_z_np_2d = self._manager.lda_project_latents_in_2d(ts_z_np)
            x1 = np.concatenate([tr_z_np_2d, ts_z_np_2d])[:, 0]
            x2 = np.concatenate([tr_z_np_2d, ts_z_np_2d])[:, 1]
        elif embedding_mode == 'tsne':
            z_np = np.concatenate([tr_z_np, ts_z_np])
            z_embedded = TSNE(n_components=2, init='random').fit_transform(z_np)
            x1, x2 = z_embedded[:, 0], z_embedded[:, 1]
        else:
            raise NotImplementedError

        df = pd.DataFrame({
            'x1': x1, 'x2': x2,
            'class': self._manager.idx2class(np.concatenate([tr_y, ts_y])),
            'type': ['train'] * tr_y.shape[0] + ['test'] * ts_y.shape[0],
            'aug': np.concatenate([np.concatenate(tr_l['augmented']),
                                  np.concatenate(ts_l['augmented'])])
        })

        # TRAIN vs TEST
        plt.clf()
        sns.scatterplot(data=df, x='x1', y='x2', hue='class', style='type')
        plt.savefig(os.path.join(self._out_dir,
                                 embedding_mode + '_emb_train_vs_test.svg'))

        # TRAIN REAL vs TRAIN AUG
        plt.clf()
        sns.scatterplot(data=df.loc[df['type'] == 'train'],
                        x='x1', y='x2', hue='class', style='aug')
        plt.savefig(os.path.join(self._out_dir,
                                 embedding_mode + '_emb_real_vs_aug.svg'))

        # TRAIN REAL vs TRAIN AUG, distributions on real
        plt.clf()
        sns.kdeplot(data=df.loc[(df['type'] == 'train') & (~df['aug'])],
                    x='x1', y='x2', hue='class', fill=True)
        sns.scatterplot(data=df.loc[df['aug']],
                        x='x1', y='x2', hue='class')
        plt.savefig(
            os.path.join(self._out_dir,
                         embedding_mode + '_emb_real_dist_vs_sc_aug.svg'))

        # TRAIN REAL, trying to shade and blend distributions
        plt.clf()
        cmaps = [create_alpha_cmap(c) for c in ['coral', 'teal', 'royalblue',
                                                'mediumseagreen', 'orchid']]

        for c, cmap in zip(set(tr_y), cmaps):
            sns.kdeplot(
                data=df.loc[(df['type'] == 'train') & (~df['aug']) &
                            (df['class'] == self._manager.idx2class(c))],
                x='x1', y='x2', fill=True, thresh=0, levels=256, cmap=cmap
            )
            sns.kdeplot(
                data=df.loc[(df['type'] == 'train') & (~df['aug']) &
                            (df['class'] == self._manager.idx2class(c))],
                x='x1', y='x2', levels=5, color="w", linewidths=1
            )
        plt.savefig(os.path.join(self._out_dir,
                                 embedding_mode + '_emb_distributions.svg'))

        self.plot_embeddings_per_region(tr_z_np, tr_y, tr_l)

    def plot_embeddings_per_region(self, tr_z_np, tr_y, tr_l):
        plt.clf()
        per_region_dfs_list = []
        for key, z_region in self._manager.latent_regions.items():
            if z_region[1] - z_region[0] > 2:
                tr_z_np_region = tr_z_np[:, z_region[0]:z_region[1]]
                z_r_embeddings = LinearDiscriminantAnalysis(
                    n_components=2, store_covariance=True).fit_transform(
                    tr_z_np_region, tr_y)
                x1, x2 = z_r_embeddings[:, 0], z_r_embeddings[:, 1]
            else:
                x1 = tr_z_np[:, z_region[0]]
                x2 = tr_z_np[:, z_region[1] - 1]

            per_region_dfs_list.append(pd.DataFrame({
                'x1': x1, 'x2': x2,
                'class': self._manager.idx2class(tr_y),
                'aug': np.concatenate(tr_l['augmented']),
                'region': np.array([key] * tr_y.shape[0])
            }))
        df = pd.concat(per_region_dfs_list)

        # also augmented data are scattered
        g = sns.FacetGrid(df, col='region', col_wrap=5, height=2)
        g.map(sns.scatterplot, 'x1', 'x2', 'class', s=10)
        g.add_legend()
        plt.savefig(os.path.join(self._out_dir, 'emb_all_train.svg'))

    def test_classifiers(self):
        ts_z, ts_l = self._manager.encode_all(self._test_loader, False)
        ts_z_np = torch.cat(ts_z, dim=0).numpy()
        ts_ly = np.concatenate(ts_l['y'])
        ts_y = np.array(self._manager.class2idx(ts_ly))

        accuracy_mlp = self._manager.mlp_classifier_epoch(ts_z, ts_l, False)[1]
        accuracy_svm = self._manager.classifier_svm.score(ts_z_np, ts_y)
        accuracy_lda = self._manager.lda.score(ts_z_np, ts_y)
        accuracy_qda = self._manager.qda.score(ts_z_np, ts_y)
        metrics = {'accuracy_mlp': accuracy_mlp,
                   'accuracy_svm': accuracy_svm,
                   'accuracy_lda': accuracy_lda,
                   'accuracy_qda': accuracy_qda}

        print(metrics)
        outfile_path = os.path.join(self._out_dir, 'accuracies.json')
        with open(outfile_path, 'w') as outfile:
            json.dump(metrics, outfile)

        pred_mlp = torch.cat(
            [self._manager.classifier_mlp(z.to(self._device))[1] for z in ts_z],
            dim=0).cpu().detach().numpy()
        pred_svm = self._manager.classifier_svm.predict(ts_z_np)
        pred_lda = self._manager.lda.predict(ts_z_np)
        pred_qda = self._manager.qda.predict(ts_z_np)

        confmat_mlp = confusion_matrix(ts_ly, self._manager.idx2class(pred_mlp),
                                       normalize='true')
        confmat_svm = confusion_matrix(ts_ly, self._manager.idx2class(pred_svm),
                                       normalize='true')
        confmat_lda = confusion_matrix(ts_ly, self._manager.idx2class(pred_lda),
                                       normalize='true')
        confmat_qda = confusion_matrix(ts_ly, self._manager.idx2class(pred_qda),
                                       normalize='true')

        labels = unique_labels(ts_ly)
        plot_confusion_matrix(confmat_mlp, labels,
                              os.path.join(self._out_dir, 'confmat_mlp.svg'))
        plot_confusion_matrix(confmat_svm, labels,
                              os.path.join(self._out_dir, 'confmat_svm.svg'))
        plot_confusion_matrix(confmat_lda, labels,
                              os.path.join(self._out_dir, 'confmat_lda.svg'))
        plot_confusion_matrix(confmat_qda, labels,
                              os.path.join(self._out_dir, 'confmat_qda.svg'))

        self.confusion_matrices_per_region(ts_z_np, ts_ly)

    def confusion_matrices_per_region(self, ts_z_np, ts_ly):
        tr_z, tr_l = self._manager.train_latents_and_labels
        if tr_z is None:
            tr_z, tr_l = self._manager.encode_all(self._train_loader, True)
        tr_y = np.array(self._manager.class2idx(np.concatenate(tr_l['y'])))
        tr_z_np = torch.cat(tr_z, dim=0).numpy()

        plt.clf()
        confusion_matrices = {}
        for key, z_region in self._manager.latent_regions.items():
            tr_z_np_region = tr_z_np[:, z_region[0]:z_region[1]]
            r_lda = LinearDiscriminantAnalysis(
                n_components=2, store_covariance=True).fit(tr_z_np_region, tr_y)
            pred_r_lda = r_lda.predict(ts_z_np[:, z_region[0]:z_region[1]])
            confmat_r_lda = confusion_matrix(
                ts_ly, self._manager.idx2class(pred_r_lda), normalize='true')
            confusion_matrices[key] = confmat_r_lda

        sns.set(color_codes=True)
        labels = unique_labels(ts_ly)
        n_cols = 5
        n_regions = len(confusion_matrices.keys())
        n_rows = n_regions // n_cols + (n_regions % n_cols > 0)
        plt.figure(figsize=(7.5 * n_cols, 6 * n_rows))
        for n, (region, cf) in enumerate(confusion_matrices.items()):
            ax = plt.subplot(n_rows, n_cols, n + 1)
            g = sns.heatmap(cf, annot=True, cmap="YlGnBu", ax=ax)
            g.set_title(region)
            g.set_xticklabels(labels)
            g.set_yticklabels(labels)
            g.set(ylabel="True Label", xlabel="Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(self._out_dir, 'region_confmats_lda.svg'),
                    bbox_inches='tight')


if __name__ == '__main__':
    import argparse
    import utils
    from data_generation_and_loading import get_data_loaders
    from model_manager import ModelManager

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='none',
                        help="ID of experiment")
    parser.add_argument('--output_path', type=str, default='.',
                        help="outputs path")
    opts = parser.parse_args()
    model_name = opts.id

    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_dir = os.path.join(output_directory, 'checkpoints')

    configurations = utils.get_config(
        os.path.join(output_directory, "config.yaml"))

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print("GPU not available, running on CPU")
    else:
        device = torch.device('cuda')

    manager = ModelManager(
        configurations=configurations, device=device,
        precomputed_storage_path=configurations['data']['precomputed_path'])
    manager.resume(checkpoint_dir)

    train_loader, val_loader, test_loader, normalization_dict, d_classes = \
        get_data_loaders(configurations, manager.template)

    manager.set_class_conversions_and_weights(d_classes)

    tester = Tester(manager, normalization_dict, train_loader, test_loader,
                    output_directory, configurations)

    # tester()
    tester.test_classifiers()
    # tester.plot_embeddings(embedding_mode='lda')
    # tester.direct_manipulation()
    # tester.fit_coma_data_different_noises()
    # tester.set_renderings_size(256)
    # tester.set_rendering_background_color()
    # tester.interpolate()
    # tester.latent_swapping(next(iter(test_loader)).x)
    # tester.per_variable_range_experiments()
    # tester.random_generation_and_rendering(n_samples=16)
    # tester.random_generation_and_save(n_samples=16)
    # print(tester.reconstruction_errors(test_loader))
    # print(tester.compute_specificity(train_loader, 100))
    # print(tester.compute_diversity_train_set())
    # print(tester.compute_diversity())
    # tester.evaluate_gen(test_loader, n_sampled_points=2048)
