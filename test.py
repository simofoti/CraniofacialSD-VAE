import os
import json
import pickle
import tqdm
import trimesh
import torch.nn
import geomloss
import pytorch3d.loss

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision.io import write_video
from torchvision.utils import make_grid, save_image
from pytorch3d.renderer import BlendParams
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_points
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from scipy.stats import multivariate_normal
from scipy.linalg import eigh, orthogonal_procrustes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

from utils import create_alpha_cmap, plot_confusion_matrix, \
    procedures2attributes_dict, colour2attribute_dict, plot_2d_arrow


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
        self.latent_stats = self.compute_latent_stats(train_load)
        self._region_ldas = self._manager.region_ldas
        self._region_qdas = self._manager.region_qdas

        self.template_landmarks_idx = [14336, 14250, 13087, 13145, 4134,
                                       871, 4166, 303, 15614, 7166,
                                       3904, 16465, 9246, 4643, 10122,
                                       4548, 2893, 2985, 830, 2004]

    def __call__(self):
        self.set_renderings_size(256)
        self.set_rendering_background_color([1, 1, 1])

        # Qualitative evaluations
        sns.set_theme(style="ticks")
        self.latent_traversals(use_z_stats=False)
        self.plot_embeddings()
        self.random_generation_and_rendering(n_samples=16)
        self.random_generation_and_save(n_samples=16)

        # Quantitative evaluation
        self.test_classifiers()
        recon_errors = self.reconstruction_errors(self._test_loader)
        train_set_diversity = self.compute_diversity_train_set()
        diversity = self.compute_diversity()
        metrics = {'recon_errors': recon_errors,
                   'train_set_diversity': train_set_diversity,
                   'diversity': diversity}

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

    def latent_traversals(self, z_range_multiplier=1,
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

    def save_batch(self, batch_verts, out_mesh_dir, v_colours=None):
        for i in range(batch_verts.shape[0]):
            vc = None
            if v_colours is not None:
                vc = v_colours[i, ::].cpu().detach().numpy()

            mesh = trimesh.Trimesh(
                batch_verts[i, ::].cpu().detach().numpy(),
                self._manager.template.face.t().cpu().numpy(),
                vertex_colors=vc)
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
                'max': torch.max(data_errors).item(),
                'std': torch.std(data_errors).item()}

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

    def fit_mesh(self, new_m_path, new_m_landmarks_path,
                 lr=5e-3, iterations=250):
        new_m_mesh = trimesh.load_mesh(new_m_path, process=False)
        new_m_verts = torch.tensor(
            new_m_mesh.vertices, dtype=torch.float,
            requires_grad=False, device=self._device)
        new_m_verts = new_m_verts.to(self._device)

        with open(new_m_landmarks_path) as f:
            points_dicts = json.load(f)
        points_lists = [[p['x'], p['y'], p['z']] for p in points_dicts]
        new_m_landmarks = torch.tensor(points_lists, dtype=torch.float,
                                       requires_grad=False, device=self._device)

        # the new_m mesh needs to be centred and aligned with the meshes
        # generated by the model. Alignement performed with Procrustes
        template_landmarks = self._manager.template.pos[
            self.template_landmarks_idx, :]
        translation_tpl = np.mean(template_landmarks.cpu().numpy(), 0)
        centered_tpl_lnd = template_landmarks - translation_tpl
        norm_tpl = np.linalg.norm(centered_tpl_lnd)
        centered_tpl_lnd /= norm_tpl

        translation_new_m = np.mean(new_m_landmarks.cpu().numpy(), 0)
        centered_new_m_lnd = new_m_landmarks.cpu() - translation_new_m
        norm_new_m = np.linalg.norm(centered_new_m_lnd)
        centered_new_m_lnd /= norm_new_m

        rotation, scale = orthogonal_procrustes(centered_tpl_lnd.numpy(),
                                                centered_new_m_lnd.numpy())

        aligned_new_m_verts = new_m_verts.cpu().numpy() - translation_new_m
        aligned_new_m_verts /= norm_new_m
        aligned_new_m_verts = np.dot(aligned_new_m_verts, rotation.T) * scale
        aligned_new_m_verts = (aligned_new_m_verts * norm_tpl) + translation_tpl

        aligned_new_m_lnd = np.dot(centered_new_m_lnd, rotation.T) * scale
        aligned_new_m_lnd = (aligned_new_m_lnd * norm_tpl) + translation_tpl

        # Generate new meshes until they fit the aligned mesh provided as input
        aligned_new_m_verts = torch.tensor(aligned_new_m_verts,
                                           device=self._device).unsqueeze(0)
        aligned_new_m_lnd = torch.tensor(aligned_new_m_lnd,
                                         device=self._device).unsqueeze(0)

        z_l = [torch.randn_like(self.latent_stats['means']) for _ in range(15)]
        z_l.append(self.latent_stats['means'])
        z = torch.stack(z_l, dim=0).clone().detach().requires_grad_(True)
        # z = self.latent_stats['means'].clone().detach().requires_grad_(True)

        # offset_verts = torch.full(self._norm_dict['mean'].shape, 0.0,
        #                           device=self._device, requires_grad=True)
        # optimizer = torch.optim.Adam([offset_verts], lr)

        optimizer = torch.optim.Adam([z], lr)
        gen_verts = None
        ch_losses, geom_losses, lnd_losses = [], [], []

        gloss_func = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, reach=0.3)
        for i in tqdm.tqdm(range(iterations)):
            optimizer.zero_grad()
            gen_verts = self._manager.generate_for_opt(z.to(self._device))
            if self._normalized_data:
                gen_verts = self._unnormalize_verts(gen_verts)

            lnd_loss = self._manager.compute_mse_loss(
                gen_verts[:, self.template_landmarks_idx, :],
                aligned_new_m_lnd.expand(16, -1, -1))
            ch_loss = pytorch3d.loss.chamfer_distance(
                gen_verts, aligned_new_m_verts.expand(16, -1, -1))[0]
            # geom_loss = gloss_func(gen_verts, aligned_new_m_verts).squeeze(0)

            # gen_verts = self._norm_dict['mean'].to(self._device) + offset_verts
            # gen_verts = gen_verts.unsqueeze(0)
            # lnd_loss = self._manager.compute_mse_loss(
            #     gen_verts[:, self.template_landmarks_idx, :],
            #     aligned_new_m_lnd)
            # ch_loss = pytorch3d.loss.chamfer_distance(
            #     gen_verts, aligned_new_m_verts)[0]
            # lapl_loss = self._manager._compute_laplacian_regularizer(gen_verts)
            #
            # total_loss = lnd_loss + ch_loss + lapl_loss
            total_loss = 10 * lnd_loss + ch_loss

            total_loss.backward()
            optimizer.step()

            ch_losses.append(ch_loss.item() * self._manager.to_mm_const)
            # geom_losses.append(geom_loss.item() * self._manager.to_mm_const)
            lnd_losses.append(lnd_loss.item() * self._manager.to_mm_const)

        errors_shape = pytorch3d.loss.chamfer_distance(
            gen_verts, aligned_new_m_verts.expand(16, -1, -1),
            batch_reduction=None)[0]
        errors_lnd = torch.mean(self._manager.compute_mse_loss(
            gen_verts[:, self.template_landmarks_idx, :],
            aligned_new_m_lnd.expand(16, -1, -1),
            reduction='none').view(gen_verts.shape[0], -1), dim=1)

        # errors_shape = pytorch3d.loss.chamfer_distance(
        #     gen_verts, aligned_new_m_verts,
        #     batch_reduction=None)[0]
        # errors_lnd = torch.mean(self._manager.compute_mse_loss(
        #     gen_verts[:, self.template_landmarks_idx, :],
        #     aligned_new_m_lnd,
        #     reduction='none').view(gen_verts.shape[0], -1), dim=1)
        errors = 10 * errors_lnd + errors_lnd
        min_error_idx = torch.argmin(errors)
        # min_error_idx = 0

        v_p = (gen_verts - self._norm_dict['mean'].to(self._device)) / \
            self._norm_dict['std'].to(self._device)
        z_p = self._manager.encode(v_p.to(self._device))
        pred_class = self._manager.classify_latent(
            z_p[min_error_idx, ::].unsqueeze(0), 'qda')
        print(f"class_enc: {pred_class}")

        # local QDAs:
        tr_z, tr_l = self._manager.encode_all(self._train_loader, True)
        tr_z_np = torch.cat(tr_z, dim=0).numpy()
        for key, z_region in self._manager.latent_regions.items():
            z_loc = z[min_error_idx, z_region[0]:z_region[1]]
            pred_r_qda = self._region_qdas[key].predict(
                z_loc.unsqueeze(0).cpu().detach().numpy())
            print(f"{colour2attribute_dict[key]} -> "
                  f"{self._manager.idx2class(pred_r_qda)}")

        print(f"ch: {errors_shape[min_error_idx] * self._manager.to_mm_const}. "
              f"lnd: {errors_lnd[min_error_idx] * self._manager.to_mm_const}")
        pred_class = self._manager.classify_latent(
            z[min_error_idx, ::].unsqueeze(0), 'qda')
        print(f"class: {pred_class}")

        plt.plot(ch_losses)
        # plt.plot(geom_losses)
        plt.plot(lnd_losses)
        plt.show()

        # SHOW MESHES
        scene = trimesh.scene.scene.Scene()
        new_m_mesh.vertices = aligned_new_m_verts.squeeze(0).cpu().numpy()
        new_m_mesh.export(new_m_path[:-4] + "_aligned.obj")

        out_verts = gen_verts[min_error_idx, ::].detach().cpu().numpy()
        gen_m = trimesh.Trimesh(
            out_verts, self._manager.template.face.t().cpu().numpy())
        gen_m.export(new_m_path[:-4] + "_fit.obj")
        gen_landmarks = trimesh.PointCloud(
            out_verts[self.template_landmarks_idx, :])

        scene.add_geometry(gen_m)
        # scene.add_geometry(new_m_mesh)
        scene.add_geometry(gen_landmarks)
        scene.show()

        # delete from here
        fig_entire_z_name = os.path.join(self._out_dir,
                                         'lda_emb_distributions.pkl')
        with open(fig_entire_z_name, 'rb') as f:
            fig_entire_z = pickle.load(f)

        z_proj = self._manager.lda_project_latents_in_2d(
            z[min_error_idx, ::].unsqueeze(0).detach().cpu().numpy())

        ax = fig_entire_z.gca()
        sns.scatterplot(x=z_proj[:, 0], y=z_proj[:, 1], ax=ax, c=['#e881a7'])
        plt.show()

        fig_regions_z_name = os.path.join(self._out_dir,
                                          'emb_all_train_dist.pkl')
        with open(fig_regions_z_name, 'rb') as f:
            fig_fgrid_regions_z = pickle.load(f)

        z_p_np = z[min_error_idx, ::].unsqueeze(0).detach().cpu().numpy()
        r_proj = {}
        for key, z_region in self._manager.latent_regions.items():
            z_p_region = z_p_np[:, z_region[0]:z_region[1]]
            z_r_embeddings = self._region_ldas[key].transform(z_p_region)
            r_proj[key] = z_r_embeddings
            x1, x2 = z_r_embeddings[:, 0], z_r_embeddings[:, 1]
            fig_fgrid_regions_z.axes_dict[colour2attribute_dict[key]].scatter(
                x1, x2, c=['#e881a7'], s=2)
        plt.show()

        return out_verts, z[min_error_idx, ::].detach()

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

    def interpolate(self):
        with open(os.path.join('precomputed_craniofacial',
                               'data_split.json'), 'r') as fp:
            data = json.load(fp)
        test_list = data['test']
        test_list = [n for n in test_list if 'aug' not in n]
        meshes_root = self._test_loader.dataset.root

        # Pick first test mesh and find most different mesh in test set
        v_1 = None
        distances = [0]
        for i, fname in enumerate(test_list):
            mesh_path = os.path.join(meshes_root, fname)
            mesh = trimesh.load_mesh(mesh_path, 'obj', process=False)
            mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                                      requires_grad=False, device='cpu')
            if i == 0:
                v_1 = mesh_verts
            else:
                distances.append(
                    self._manager.compute_mse_loss(v_1, mesh_verts).item())

        m_2_path = os.path.join(
            meshes_root, test_list[np.asarray(distances).argmax()])
        m_2 = trimesh.load_mesh(m_2_path, 'obj', process=False)
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

    def _load_and_encode(self, patient_fname=None, mesh_path=None):
        if mesh_path is None:
            assert patient_fname is not None
            meshes_root = self._test_loader.dataset.root
            mesh_path = os.path.join(meshes_root, patient_fname)

        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                                  requires_grad=False, device='cpu')
        v_p = (mesh_verts - self._norm_dict['mean']) / self._norm_dict['std']
        z_p = self._manager.encode(v_p.unsqueeze(0).to(self._device))
        return z_p

    def interpolate_syndrome_to_normal(self, patient_fname):
        z_p = self._load_and_encode(patient_fname)
        if 'augmented/' in patient_fname:
            patient_fname = patient_fname[len('augmented/'):]

        # Find normal patients latent vectors
        normal_p_index = self._manager.class2idx('n')
        normal_p_mean = self._manager.qda.means_[normal_p_index]

        # Move from mean of distribution to 1std in direction of z_p.
        # Eigenvalues of covariance matrix are diagonal of covariance of aligned
        # distribution -> use them to find pdf at 1 std

        normal_p_covariance = self._manager.qda.covariance_[normal_p_index]
        multi_normal_dist = multivariate_normal(mean=normal_p_mean,
                                                cov=normal_p_covariance)
        eigenval, eigenvec = eigh(normal_p_covariance)
        reference_dist = multivariate_normal(mean=np.zeros_like(normal_p_mean),
                                             cov=np.diag(eigenval))
        reference_std_on_x1 = np.sqrt(reference_dist.cov[0, 0])
        reference_std_vec_on_x1 = np.zeros_like(normal_p_mean)
        reference_std_vec_on_x1[0] = reference_std_on_x1

        reference_pdf_mean = -multi_normal_dist.logpdf(normal_p_mean)
        reference_pdf_1std = -reference_dist.logpdf(reference_std_vec_on_x1)
        reference_pdf_2std = -reference_dist.logpdf(2 * reference_std_vec_on_x1)
        reference_pdf_3std = -reference_dist.logpdf(3 * reference_std_vec_on_x1)

        # print(-multi_normal_dist.logpdf(z_p.detach().cpu().numpy()))
        # print(reference_pdf_mean)
        # print(reference_pdf_1std)  # close to mean
        # print(reference_pdf_2std)
        # print(reference_pdf_3std)  # far from mean

        z_mean_target = torch.tensor(normal_p_mean).unsqueeze(0)
        z_interp_full = self.vector_linspace(z_p, z_mean_target, 5000)

        # find z vectors with correct pdf
        pdf_intermediate = [-multi_normal_dist.logpdf(z.detach().cpu().numpy())
                            for z in z_interp_full]
        pdf_lt_3std = [p <= reference_pdf_3std for p in pdf_intermediate]
        pdf_lt_2std = [p <= reference_pdf_2std for p in pdf_intermediate]
        pdf_lt_1std = [p <= reference_pdf_1std for p in pdf_intermediate]

        z_3std_target = z_interp_full[pdf_lt_3std.index(True), :].unsqueeze(0)
        z_2std_target = z_interp_full[pdf_lt_2std.index(True), :].unsqueeze(0)
        z_1std_target = z_interp_full[pdf_lt_1std.index(True), :].unsqueeze(0)

        n_p_to_3std = 8
        # Iterpolate all attributes ############################################
        z_interp_pto3std = self.vector_linspace(z_p, z_3std_target, n_p_to_3std)
        z_interp = torch.cat([z_interp_pto3std, z_2std_target,
                              z_1std_target, z_mean_target], dim=0)
        self._render_embed_save_z_interpolations(
            z_interp, patient_fname[:-4] + '_all_attributes')

        # Interpolate subsets of attributes ####################################
        proc_z_distances = pd.DataFrame(
            columns=['procedure', 'd3', 'd2', 'd1', 'dm'])
        for key, attributes in procedures2attributes_dict.items():
            z_interp = z_p.repeat(n_p_to_3std + 3, 1)
            for attr in attributes:
                zf_idxs = self._manager.latent_regions[attr]
                z_pf = z_p[:, zf_idxs[0]:zf_idxs[1]].to(self._device)
                z_3f = z_3std_target[:, zf_idxs[0]:zf_idxs[1]].to(self._device)
                z_interp[:n_p_to_3std, zf_idxs[0]:zf_idxs[1]] = \
                    self.vector_linspace(z_pf, z_3f, n_p_to_3std)
                z_2f = z_2std_target[:, zf_idxs[0]:zf_idxs[1]].to(self._device)
                z_1f = z_1std_target[:, zf_idxs[0]:zf_idxs[1]].to(self._device)
                z_mf = z_mean_target[:, zf_idxs[0]:zf_idxs[1]].to(self._device)
                z_interp[n_p_to_3std, zf_idxs[0]:zf_idxs[1]] = z_2f
                z_interp[n_p_to_3std + 1, zf_idxs[0]:zf_idxs[1]] = z_1f
                z_interp[n_p_to_3std + 2, zf_idxs[0]:zf_idxs[1]] = z_mf

            z_mean_target_dist = z_mean_target.squeeze().to(self._device)
            d3 = self._manager.compute_mse_loss(z_interp[n_p_to_3std - 1, :],
                                                z_mean_target_dist)
            d2 = self._manager.compute_mse_loss(z_interp[n_p_to_3std, :],
                                                z_mean_target_dist)
            d1 = self._manager.compute_mse_loss(z_interp[n_p_to_3std + 1, :],
                                                z_mean_target_dist)
            dm = self._manager.compute_mse_loss(z_interp[n_p_to_3std + 2, :],
                                                z_mean_target_dist)
            proc_z_distances = proc_z_distances.append(
                {'procedure': key, 'd3': d3.item(), 'd2': d2.item(),
                 'd1': d1.item(), 'dm': dm.item()}, ignore_index=True)
            self._render_embed_save_z_interpolations(
                z_interp, patient_fname[:-4] + '_' + key)
        proc_z_distances.to_csv(os.path.join(
            self._out_dir, 'interpolations',
            patient_fname[:-4] + '_procedure_distances.csv'))

    def _render_embed_save_z_interpolations(self, z_interp, save_id):
        out_interp_dir = os.path.join(self._out_dir, 'interpolations', save_id)
        if not os.path.isdir(out_interp_dir):
            os.mkdir(out_interp_dir)

        fig_entire_z_name = os.path.join(self._out_dir,
                                         'lda_emb_distributions.pkl')
        fig_regions_z_name = os.path.join(self._out_dir,
                                          'emb_all_train_dist.pkl')
        try:
            with open(fig_entire_z_name, 'rb') as f:
                fig_entire_z = pickle.load(f)
            with open(fig_regions_z_name, 'rb') as f:
                fig_fgrid_regions_z = pickle.load(f)
        except FileNotFoundError:
            self.plot_embeddings(embedding_mode='lda')
            with open(fig_entire_z_name, 'rb') as f:
                fig_entire_z = pickle.load(f)
            with open(fig_regions_z_name, 'rb') as f:
                fig_fgrid_regions_z = pickle.load(f)

        # project entire latents in 2D and save figure
        z_interp_proj = self._manager.lda_project_latents_in_2d(
            z_interp.detach().cpu().numpy())

        ax = fig_entire_z.gca()
        sns.scatterplot(x=z_interp_proj[:, 0], y=z_interp_proj[:, 1],
                        ax=ax, c=['#e881a7'])
        fig_entire_z.savefig(
            os.path.join(out_interp_dir, save_id + '_emb_interpolate.svg'))

        emb_images = []
        for zproj in z_interp_proj:
            with open(fig_entire_z_name, 'rb') as f:
                fig_entire_z = pickle.load(f)
            canvas = FigureCanvas(fig_entire_z)
            ax = fig_entire_z.gca()
            ax.scatter(x=zproj[0], y=zproj[1], c=['#e881a7'])
            # rasterize plot
            shape = list(canvas.get_width_height()[::-1]) + [3]
            canvas.draw()
            img = np.frombuffer(fig_entire_z.canvas.tostring_rgb(),
                                dtype='uint8').reshape(shape)
            emb_images.append(torch.tensor(np.array(img)))
        write_video(
            os.path.join(out_interp_dir, save_id + '_emb_interpolate.mp4'),
            torch.stack(emb_images, dim=0), fps=4)

        # project attribute latents in their 2D plots and save figure
        # per_region_dfs_list = [fig_fgrid_regions_z.data]
        z_interp_np = z_interp.detach().cpu().numpy()
        r_proj = {}
        for key, z_region in self._manager.latent_regions.items():
            z_interp_region = z_interp_np[:, z_region[0]:z_region[1]]
            z_r_embeddings = self._region_ldas[key].transform(z_interp_region)
            r_proj[key] = z_r_embeddings
            x1, x2 = z_r_embeddings[:, 0], z_r_embeddings[:, 1]
            fig_fgrid_regions_z.axes_dict[colour2attribute_dict[key]].scatter(
                x1, x2, c=['#e881a7'], s=2)
        fig_fgrid_regions_z.fig.savefig(
            os.path.join(out_interp_dir, save_id + '_emb_r_interpolate.svg'))

        emb_images = []
        for point_idx in range(z_interp_np.shape[0]):
            with open(fig_regions_z_name, 'rb') as f:
                fig_fgrid_regions_z = pickle.load(f)
            canvas = FigureCanvas(fig_fgrid_regions_z.fig)
            shape = list(canvas.get_width_height()[::-1]) + [3]
            for key, z_region in self._manager.latent_regions.items():
                x1, x2 = r_proj[key][point_idx]
                keyname = colour2attribute_dict[key]
                fig_fgrid_regions_z.axes_dict[keyname].scatter(
                    x1, x2, c=['#e881a7'], s=2)
            # rasterize plot
            canvas.draw()
            img = np.frombuffer(canvas.tostring_rgb(),
                                dtype='uint8').reshape(shape)
            emb_images.append(torch.tensor(np.array(img)))
        video_frames = torch.stack(emb_images, dim=0)
        pad_shape = [video_frames.shape[0], video_frames.shape[1], 1, 3]
        video_frames = torch.cat([video_frames, torch.ones(pad_shape)], dim=2)
        write_video(
            os.path.join(out_interp_dir, save_id + '_emb_r_interpolate.mp4'),
            video_frames, fps=4)

        # render and save video + image of interpolation in mesh space
        v_interp = self._manager.generate(z_interp.to(self._device))
        v_interp = self._unnormalize_verts(v_interp) if self._normalized_data \
            else v_interp

        out_mesh_dir = os.path.join(out_interp_dir, 'meshes')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)
        self.save_batch(v_interp, out_mesh_dir)

        source_dist = self._manager.compute_vertex_errors(
            v_interp, v_interp[0, ::].expand(v_interp.shape[0], -1, -1))

        out_mesh_dir = os.path.join(out_interp_dir, 'meshes_colormap')
        if not os.path.isdir(out_mesh_dir):
            os.mkdir(out_mesh_dir)
        source_colours = utils.errors_to_colors(
                source_dist, min_value=0,
                max_value=10, cmap='plasma') / 255
        self.save_batch(v_interp, out_mesh_dir, v_colours=source_colours)

        self.set_renderings_size(512)
        self.set_rendering_background_color([1, 1, 1])
        renderings = self._manager.render(v_interp).cpu()
        renderings_dist = self._manager.render(v_interp, source_dist,
                                               error_max_scale=10).cpu()

        im = make_grid(torch.cat([renderings, renderings_dist], dim=-2),
                       padding=10, pad_value=1, nrow=v_interp.shape[0])

        rend_comb = torch.cat([renderings, renderings_dist], dim=-1)
        write_video(
            os.path.join(out_interp_dir, save_id + '_interpolate.mp4'),
            rend_comb.permute(0, 2, 3, 1) * 255, fps=4)
        save_image(im, os.path.join(out_interp_dir,
                                    save_id + '_interpolate.png'))

    def classify_and_project(self, patient_fname):
        z_p = self._load_and_encode(patient_fname)
        print(self._manager.classify_latent(z_p, 'qda'))

        fig_entire_z_name = os.path.join(self._out_dir,
                                         'lda_emb_distributions.pkl')
        with open(fig_entire_z_name, 'rb') as f:
            fig_entire_z = pickle.load(f)

        z_proj = self._manager.lda_project_latents_in_2d(
            z_p.detach().cpu().numpy())

        ax = fig_entire_z.gca()
        sns.scatterplot(x=z_proj[:, 0], y=z_proj[:, 1], ax=ax, c=['#e881a7'])
        out_interp_dir = os.path.join(self._out_dir, 'interpolations')
        fig_entire_z.savefig(
            os.path.join(out_interp_dir, patient_fname[:-4] + '_emb.svg'))

        fig_regions_z_name = os.path.join(self._out_dir,
                                          'emb_all_train_dist.pkl')
        with open(fig_regions_z_name, 'rb') as f:
            fig_fgrid_regions_z = pickle.load(f)
        z_p_np = z_p.detach().cpu().numpy()
        r_proj = {}
        for key, z_region in self._manager.latent_regions.items():
            z_p_region = z_p_np[:, z_region[0]:z_region[1]]
            z_r_embeddings = self._region_ldas[key].transform(z_p_region)
            r_proj[key] = z_r_embeddings
            x1, x2 = z_r_embeddings[:, 0], z_r_embeddings[:, 1]
            fig_fgrid_regions_z.axes_dict[colour2attribute_dict[key]].scatter(
                x1, x2, c=['#e881a7'], s=2)
        fig_fgrid_regions_z.fig.savefig(
            os.path.join(out_interp_dir, patient_fname[:-4] + '_emb_r.svg'))

    def evaluate_all_pre_post_pairs_in_excel(self, pairs_root,
                                             pairs_excel_path):
        pairs_df = pd.read_excel(pairs_excel_path)
        for r_idx, row in pairs_df.iterrows():
            pid, procedure = str(row["PID"]), row["Surgery regions"]
            pre_path = os.path.join(pairs_root, row["Pre name"])
            post_path = os.path.join(pairs_root, row["Post name"])
            metrics = self.evaluate_pre_post_pair(pre_path, post_path,
                                                  pid, procedure)
            for k, metric in metrics.items():
                pairs_df.loc[r_idx, k] = metric
        pairs_df.to_excel(pairs_excel_path[:-5] + "_with_results.xlsx")

    def evaluate_pre_post_pair(self, pre_path, post_path,
                               patient_id, procedure='monobloc'):
        z_pre = self._load_and_encode(mesh_path=pre_path)
        z_post = self._load_and_encode(mesh_path=post_path)

        self._project_pre_post_pair(z_pre, z_post, patient_id)

        pre_class = self._manager.classify_latent(z_pre, 'qda')
        post_class = self._manager.classify_latent(z_post, 'qda')

        pre_posteriors = self._manager.qda.predict_proba(
            z_pre.detach().cpu().numpy())
        post_posteriors = self._manager.qda.predict_proba(
            z_post.detach().cpu().numpy())
        print(f"pre_posteriors: {pre_posteriors}, "
              f"post_posteriors: {post_posteriors}")

        d_pre_g = self._manager.mahalanobis_dist_to_qda_distribution(z_pre)
        d_post_g = self._manager.mahalanobis_dist_to_qda_distribution(z_post)
        metric_global = (d_pre_g - d_post_g) / d_post_g

        region_classification_metrics_path = os.path.join(
            self._out_dir, 'classification_report_regions.json')
        try:
            with open(region_classification_metrics_path) as f:
                region_c_reports = json.load(f)
        except FileNotFoundError:
            print("procedure specific metric not weighted according to local "
                  "QDA performance. Test classifiers first!")
            region_c_reports = None

        metric_affected_regions = 0
        affected_regions = procedures2attributes_dict[procedure]
        for key in affected_regions:
            z_region = self._manager.latent_regions[key]
            z_pre_region = z_pre[:, z_region[0]:z_region[1]]
            z_post_region = z_post[:, z_region[0]:z_region[1]]
            d_pre_r = self._manager.mahalanobis_dist_to_qda_distribution(
                z_pre_region, region=key)
            d_post_r = self._manager.mahalanobis_dist_to_qda_distribution(
                z_post_region, region=key)

            if region_c_reports is not None:
                w = region_c_reports[key]['accuracy']
            else:
                w = 1
            metric_affected_regions += w * ((d_pre_r - d_post_r) / d_post_r)
        metric_affected_regions /= len(affected_regions)

        return {"pre_class": pre_class, "post_class": post_class,
                "global_metric": metric_global,
                "procedure_metric": metric_affected_regions}

    def _project_pre_post_pair(self, z_pre, z_post, patient_id):
        out_plots_dir = os.path.join(self._out_dir, 'pre_post_eval_plots')
        if not os.path.isdir(out_plots_dir):
            os.mkdir(out_plots_dir)

        fig_entire_z_name = os.path.join(self._out_dir,
                                         'lda_emb_distributions.pkl')
        with open(fig_entire_z_name, 'rb') as f:
            fig_entire_z = pickle.load(f)

        z_pre_np = z_pre.detach().cpu().numpy()
        z_post_np = z_post.detach().cpu().numpy()

        z_pre_proj = self._manager.lda_project_latents_in_2d(z_pre_np)
        z_post_proj = self._manager.lda_project_latents_in_2d(z_post_np)

        ax = fig_entire_z.gca()
        sns.scatterplot(x=z_pre_proj[:, 0], y=z_pre_proj[:, 1],
                        ax=ax, c=['#e881a7'])
        sns.scatterplot(x=z_post_proj[:, 0], y=z_post_proj[:, 1],
                        ax=ax, c=['#a34D7a'])
        plot_2d_arrow(tail_coords=z_pre_proj, head_coords=z_post_proj, ax=ax)
        fig_entire_z.savefig(
            os.path.join(out_plots_dir, patient_id + "_emb.svg"))

        fig_regions_z_name = os.path.join(self._out_dir,
                                          'emb_all_train_dist.pkl')
        with open(fig_regions_z_name, 'rb') as f:
            fig_fgrid_regions_z = pickle.load(f)

        for key, z_region in self._manager.latent_regions.items():
            z_pre_region = z_pre_np[:, z_region[0]:z_region[1]]
            z_post_region = z_post_np[:, z_region[0]:z_region[1]]
            z_pre_embeddings = self._region_ldas[key].transform(z_pre_region)
            z_post_embeddings = self._region_ldas[key].transform(z_post_region)
            fig_fgrid_regions_z.axes_dict[colour2attribute_dict[key]].scatter(
                z_pre_embeddings[:, 0], z_pre_embeddings[:, 1],
                c=['#e881a7'], s=2)
            fig_fgrid_regions_z.axes_dict[colour2attribute_dict[key]].scatter(
                z_post_embeddings[:, 0], z_post_embeddings[:, 1],
                c=['#a34D7a'], s=2)
            plot_2d_arrow(
                tail_coords=z_pre_embeddings, head_coords=z_post_embeddings,
                ax=fig_fgrid_regions_z.axes_dict[colour2attribute_dict[key]],
                scale=1)
        fig_fgrid_regions_z.fig.savefig(
            os.path.join(out_plots_dir, patient_id + "_emb_r.svg"))

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
                                  np.concatenate(ts_l['augmented'])]),
            'gender': np.concatenate([np.concatenate(tr_l['gender']),
                                      np.concatenate(ts_l['gender'])]),
            'age': np.concatenate([np.concatenate(tr_l['age']),
                                   np.concatenate(ts_l['age'])]),
        })

        colours = ['#ed6e5d', '#74bfc2', '#eecd4a', '#124d81']
        hue_order = ['n', 'a', 'c', 'm']
        # TRAIN vs TEST
        plt.clf()
        sns.scatterplot(data=df, x='x1', y='x2', hue='class', style='type',
                        hue_order=hue_order, palette=colours)
        plt.savefig(os.path.join(self._out_dir,
                                 embedding_mode + '_emb_train_vs_test.svg'))

        # TRAIN REAL vs TRAIN AUG
        plt.clf()
        sns.scatterplot(data=df.loc[df['type'] == 'train'],
                        x='x1', y='x2', hue='class', style='aug',
                        hue_order=hue_order, palette=colours)
        plt.savefig(os.path.join(self._out_dir,
                                 embedding_mode + '_emb_real_vs_aug.svg'))

        plt.clf()
        sns.scatterplot(data=df.loc[df['aug']],
                        x='x1', y='x2', hue='class', marker='x',
                        hue_order=hue_order, palette=colours,)
        sns.scatterplot(data=df.loc[~df['aug']],
                        x='x1', y='x2', hue='class',
                        hue_order=hue_order, palette=colours)
        plt.savefig(os.path.join(self._out_dir,
                                 embedding_mode + '_emb_real_vs_aug2.svg'))

        # TRAIN REAL vs TRAIN AUG, distributions on real
        plt.clf()
        sns.kdeplot(data=df.loc[(df['type'] == 'train') & (~df['aug'])],
                    x='x1', y='x2', hue='class', fill=True,
                    hue_order=hue_order, palette=colours)
        sns.scatterplot(data=df.loc[df['aug']],
                        x='x1', y='x2', hue='class',
                        hue_order=hue_order, palette=colours)
        plt.savefig(
            os.path.join(self._out_dir,
                         embedding_mode + '_emb_real_dist_vs_sc_aug.svg'))

        # TRAIN REAL, trying to shade and blend distributions
        plt.clf()
        fig_handle = plt.figure()
        cmaps = [create_alpha_cmap(c) for c in colours]
        handles = []
        for c, cmap, col in zip(hue_order, cmaps, colours):
            sns.kdeplot(
                data=df.loc[(df['type'] == 'train') & (~df['aug']) &
                            (df['class'] == c)],
                x='x1', y='x2', fill=True, levels=5, cmap=cmap, alpha=0.8
            )
            sns.kdeplot(
                data=df.loc[(df['type'] == 'train') & (~df['aug']) &
                            (df['class'] == c)],
                x='x1', y='x2', levels=5, color=col, linewidths=0.5, alpha=0.5
            )
            handles.append(mpatches.Patch(facecolor=col, label=c))
        plt.legend(handles=handles)

        fig_name = os.path.join(self._out_dir,
                                embedding_mode + '_emb_distributions')

        # pickle figure to use it later in other plots
        with open(fig_name + '.pkl', 'wb') as f:
            pickle.dump(fig_handle, f)

        plt.savefig(fig_name + '.svg')

        normalize_color = Normalize(vmin=-60.0, vmax=240.0)
        ax = fig_handle.gca()
        ax.scatter(df.loc[df['gender'] == 'M']['x1'],
                   df.loc[df['gender'] == 'M']['x2'],
                   c=df.loc[df['gender'] == 'M']['age'], s=4,
                   cmap=get_cmap('Blues'), norm=normalize_color)
        ax.scatter(df.loc[df['gender'] == 'F']['x1'],
                   df.loc[df['gender'] == 'F']['x2'],
                   c=df.loc[df['gender'] == 'F']['age'], s=4,
                   cmap=get_cmap('Reds'), norm=normalize_color)
        plt.savefig(os.path.join(self._out_dir,
                                 embedding_mode + '_emb_distributions_a_g.svg'))

        self.plot_embeddings_per_region(tr_z_np, tr_y, tr_l)

    def plot_embeddings_per_region(self, tr_z_np, tr_y, tr_l):
        plt.clf()
        per_region_dfs_list = []
        for key, z_region in self._manager.latent_regions.items():
            if z_region[1] - z_region[0] > 2:
                tr_z_np_region = tr_z_np[:, z_region[0]:z_region[1]]
                z_r_embeddings = self._region_ldas[key].transform(
                    tr_z_np_region)
                x1, x2 = z_r_embeddings[:, 0], z_r_embeddings[:, 1]
            else:
                x1 = tr_z_np[:, z_region[0]]
                x2 = tr_z_np[:, z_region[1] - 1]

            per_region_dfs_list.append(pd.DataFrame({
                'x1': x1, 'x2': x2,
                'class': self._manager.idx2class(tr_y),
                'aug': np.concatenate(tr_l['augmented']),
                'region': np.array([colour2attribute_dict[key]] * tr_y.shape[0])
            }))
        df = pd.concat(per_region_dfs_list)
        df.drop(df[df['aug']].index, inplace=True)

        colours = ['#ed6e5d', '#74bfc2', '#eecd4a', '#124d81', '#dbcbbe']
        hue_order = ['n', 'a', 'c', 'm']
        # also augmented data are scattered
        g = sns.FacetGrid(df, col='region', hue='class', palette=colours,
                          hue_order=hue_order, col_wrap=5, height=2)
        g.map(sns.scatterplot, 'x1', 'x2', s=10)
        g.set_titles(col_template="{col_name}")
        g.add_legend()
        plt.savefig(os.path.join(self._out_dir, 'emb_all_train.svg'))

        # plot distribs
        g = sns.FacetGrid(df, col='region', hue='class', palette=colours,
                          hue_order=hue_order, col_wrap=5, height=2)
        g.map(sns.kdeplot, 'x1', 'x2', fill=False, levels=1, alpha=0.8,
              thresh=0.68)
        g.set_titles(col_template="{col_name}")
        g.add_legend()

        fig_name = os.path.join(self._out_dir, 'emb_all_train_dist')
        with open(fig_name + '.pkl', 'wb') as f:
            pickle.dump(g, f)
        plt.savefig(fig_name + '.svg')

    def test_classifiers(self):
        plt.clf()
        plt.close()
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

        report_mlp = classification_report(ts_ly,
                                           self._manager.idx2class(pred_mlp),
                                           output_dict=True)
        report_svm = classification_report(ts_ly,
                                           self._manager.idx2class(pred_svm),
                                           output_dict=True)
        report_lda = classification_report(ts_ly,
                                           self._manager.idx2class(pred_lda),
                                           output_dict=True)
        report_qda = classification_report(ts_ly,
                                           self._manager.idx2class(pred_qda),
                                           output_dict=True)
        reports = {'mlp': report_mlp, 'svm': report_svm,
                   'lda': report_lda, 'qda': report_qda}
        outfile_path = os.path.join(self._out_dir, 'classification_report.json')
        with open(outfile_path, 'w') as outfile:
            json.dump(reports, outfile)

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
        region_reports = {}
        for key, z_region in self._manager.latent_regions.items():
            pred_r_qda = self._region_qdas[key].predict(
                ts_z_np[:, z_region[0]:z_region[1]])
            region_reports[key] = classification_report(
                ts_ly, self._manager.idx2class(pred_r_qda), output_dict=True)
            region_reports[key]["accuracy"] = self._region_qdas[key].score(
                ts_z_np[:, z_region[0]:z_region[1]], ts_y)

        outfile_path = os.path.join(self._out_dir,
                                    'classification_report_regions.json')
        with open(outfile_path, 'w') as outfile:
            json.dump(region_reports, outfile)

    def confusion_matrices_per_region(self, ts_z_np, ts_ly):
        confusion_matrices_lda = {}
        for key, z_region in self._manager.latent_regions.items():
            pred_r_lda = self._region_ldas[key].predict(
                ts_z_np[:, z_region[0]:z_region[1]])
            confmat_r_lda = confusion_matrix(
                ts_ly, self._manager.idx2class(pred_r_lda), normalize='true')
            confusion_matrices_lda[key] = confmat_r_lda

        confusion_matrices_qda = {}
        for key, z_region in self._manager.latent_regions.items():
            pred_r_qda = self._region_qdas[key].predict(
                ts_z_np[:, z_region[0]:z_region[1]])
            confmat_r_qda = confusion_matrix(
                ts_ly, self._manager.idx2class(pred_r_qda), normalize='true')
            confusion_matrices_qda[key] = confmat_r_qda

        cms = [confusion_matrices_lda, confusion_matrices_qda]
        for m, confusion_matrices in zip(['lda', 'qda'], cms):
            plt.clf()
            sns.set(color_codes=True)
            labels = unique_labels(ts_ly)
            n_cols = 5
            n_regions = len(confusion_matrices.keys())
            n_rows = n_regions // n_cols + (n_regions % n_cols > 0)
            plt.figure(figsize=(7.5 * n_cols, 6 * n_rows))
            for n, (region, cf) in enumerate(confusion_matrices.items()):
                ax = plt.subplot(n_rows, n_cols, n + 1)
                g = sns.heatmap(cf, annot=True, cmap="YlGnBu", ax=ax,
                                vmin=0., vmax=1., annot_kws={"fontsize": 26})
                g.set_title(colour2attribute_dict[region], fontsize=18)
                g.set_xticklabels(labels, fontsize=18)
                g.set_yticklabels(labels, fontsize=18)
                g.set(ylabel="True Label", xlabel="Predicted Label")
            plt.tight_layout()
            plt.savefig(os.path.join(self._out_dir, f"region_confmats_{m}.svg"),
                        bbox_inches='tight')


if __name__ == '__main__':
    import argparse
    import utils
    from data_loading import get_data_loaders
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
    # tester.set_renderings_size(256)
    # tester.set_rendering_background_color()
    # tester.fit_mesh(
    #     new_m_path="/media/simo/DATASHURPRO/for_simone/atypical_cruzon/atypical_head_face.obj",
    #     new_m_landmarks_path="/media/simo/DATASHURPRO/for_simone/atypical_cruzon/atypical_head_face_lnd.txt",
    #     lr=0.01, iterations=200)
    # tester.fit_mesh(
    #     new_m_path="/home/simo/Desktop/NEW_MODELS/Crouzon/3043070/c_3043070_18-12-2019_dummy_cleaned.obj",
    #     new_m_landmarks_path="/home/simo/Desktop/NEW_MODELS/Crouzon/3043070/c_3043070_18-12-2019_dummy_20_lnd.txt",
    #     lr=0.01, iterations=200)
    # tester.plot_embeddings(embedding_mode='lda')
    # tester.test_classifiers()
    tester.evaluate_all_pre_post_pairs_in_excel(
        pairs_root="/media/simo/DATASHURPRO/pre_post_fitted_meshes",
        pairs_excel_path="/media/simo/DATASHURPRO/pre_post_fitted_meshes/pre_post.xlsx")
    # tester.evaluate_pre_post_pair(
    #     pre_path="/media/simo/DATASHURPRO/pre_post_fitted_meshes/Missing_Apert_Mesh_out/A102_Pre_Op.ply",
    #     post_path="/media/simo/DATASHURPRO/pre_post_fitted_meshes/Apert/GOSH/642788_Post_Op.ply",
    # )
    # tester.interpolate_syndrome_to_normal(patient_fname='a_7.obj')
    # tester.interpolate_syndrome_to_normal(patient_fname='c_104.obj')
    # tester.interpolate_syndrome_to_normal(
    #     patient_fname='c_atypical_head_face.obj')
    # tester.classify_and_project(patient_fname='c_atypical_head_face.obj')
    # tester.interpolate()
    # tester.latent_traversals()
    # tester.random_generation_and_rendering(n_samples=16)
    # tester.random_generation_and_save(n_samples=16)
    # print(tester.reconstruction_errors(test_loader))
    # print(tester.compute_diversity_train_set())
    # print(tester.compute_diversity())
