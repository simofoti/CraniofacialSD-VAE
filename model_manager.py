import os
import pickle
import torch.nn
import trimesh
import tqdm

import numpy as np

from sklearn import svm, discriminant_analysis
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import hard_rgb_blend
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    BlendParams,
    HardGouraudShader
)

import utils
from mesh_simplification import MeshSimplifier
from compute_spirals import preprocess_spiral
from model import Model, MLPClassifier


class ModelManager(torch.nn.Module):
    def __init__(self, configurations, device, rendering_device=None,
                 precomputed_storage_path='precomputed'):
        super(ModelManager, self).__init__()
        self._model_params = configurations['model']
        self._optimization_params = configurations['optimization']
        self._precomputed_storage_path = precomputed_storage_path
        self._normalized_data = configurations['data']['normalize_data']

        self.to_mm_const = configurations['data']['to_mm_constant']
        self.device = device
        self.template = utils.load_template(
            configurations['data']['template_path'])

        low_res_templates, down_transforms, up_transforms = \
            self._precompute_transformations(show_meshes=False)
        meshes_all_resolutions = [self.template] + low_res_templates
        spirals_indices = self._precompute_spirals(meshes_all_resolutions)

        self._losses = None
        self._w_latent_cons_loss = float(
            self._optimization_params['latent_consistency_weight'])
        self._w_laplacian_loss = float(
            self._optimization_params['laplacian_weight'])
        self._w_kl_loss = float(self._optimization_params['kl_weight'])

        self._net = Model(in_channels=self._model_params['in_channels'],
                          out_channels=self._model_params['out_channels'],
                          latent_size=self._model_params['latent_size'],
                          spiral_indices=spirals_indices,
                          down_transform=down_transforms,
                          up_transform=up_transforms,
                          pre_z_sigmoid=self._model_params['pre_z_sigmoid'],
                          is_vae=self._w_kl_loss > 0).to(device)

        self._optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=float(self._optimization_params['lr']),
            weight_decay=self._optimization_params['weight_decay'])

        self._latent_regions = self._compute_latent_regions()

        self._rend_device = rendering_device if rendering_device else device
        self.default_shader = HardGouraudShader(
            cameras=FoVPerspectiveCameras(),
            blend_params=BlendParams(background_color=[0, 0, 0]))
        self.simple_shader = ShadelessShader(
            blend_params=BlendParams(background_color=[0, 0, 0]))
        self.renderer = self._create_renderer()

        self._swap_features = configurations['data']['swap_features']
        if self._swap_features:
            self._out_grid_size = self._optimization_params['batch_size']
        else:
            self._out_grid_size = 4

        bs = self._optimization_params['batch_size']
        self._batch_diagonal_idx = [(bs + 1) * i for i in range(bs)]

        if self._w_latent_cons_loss > 0:
            assert self._swap_features

        if 'classifier' in configurations:
            self._classifier_params = configurations['classifier']

            if self._classifier_params['mlp_training_type'] == 'end2end':
                self._mlp_classifier_end2end = True
                self._w_classifier_loss = \
                    self._classifier_params['mlp_loss_weight']
            else:
                self._mlp_classifier_end2end = False
                self._w_classifier_loss = 0

            fnames = os.listdir(configurations['data']['dataset_path'])
            n_classes = len(set([n[0] for n in fnames if n.endswith('.obj')]))
            self._class2idx_dict = None
            self._class_weights = None

            self._main_classifier = None

            self.classifier_mlp = MLPClassifier(
                self._model_params['latent_size'],
                self._classifier_params['mlp_hidden_features'],
                n_classes).to(device)
            self._classifier_optimizer = torch.optim.Adam(
                self.classifier_mlp.parameters(),
                lr=float(self._classifier_params['mlp_lr']),
                weight_decay=self._optimization_params['weight_decay'])

            self.classifier_svm = svm.LinearSVC(class_weight='balanced')
            self.lda = discriminant_analysis.LinearDiscriminantAnalysis(
                n_components=2, store_covariance=True)
            self.qda = discriminant_analysis.QuadraticDiscriminantAnalysis(
                store_covariance=True)

            if self._w_latent_cons_loss > 0:
                self.region_ldas = {
                    key: discriminant_analysis.LinearDiscriminantAnalysis(
                        n_components=2, store_covariance=True)
                    for key in self.latent_regions.keys()
                }
                self.region_qdas = {
                    key: discriminant_analysis.QuadraticDiscriminantAnalysis(
                        store_covariance=True)
                    for key in self.latent_regions.keys()
                }
            else:
                self.region_ldas, self.region_qdas = None, None
        else:
            self._classifier_params = None

        # If latents are used for other purposes avoids embedding all training
        # samples multiple times
        self._train_latents_list = None
        self._train_dict_labels_lists = None

    @property
    def loss_keys(self):
        return ['reconstruction', 'kl',
                'latent_consistency', 'laplacian',
                'classification', 'classification_acc', 'tot']

    @property
    def latent_regions(self):
        return self._latent_regions

    @property
    def is_vae(self):
        return self._w_kl_loss > 0

    @property
    def model_latent_size(self):
        return self._model_params['latent_size']

    @property
    def batch_diagonal_idx(self):
        return self._batch_diagonal_idx

    @property
    def train_latents_and_labels(self):
        return self._train_latents_list, self._train_dict_labels_lists

    def _precompute_transformations(self, show_meshes=False):
        storage_path = os.path.join(self._precomputed_storage_path,
                                    'transforms.pkl')
        try:
            with open(storage_path, 'rb') as file:
                low_res_templates, down_transforms, up_transforms = \
                    pickle.load(file)
        except FileNotFoundError:
            print("Computing Down- and Up- sampling transformations ")
            if not os.path.isdir(self._precomputed_storage_path):
                os.mkdir(self._precomputed_storage_path)

            sampling_params = self._model_params['sampling']
            m = self.template

            r_weighted = False if sampling_params['type'] == 'basic' else True

            low_res_templates = []
            down_transforms = []
            up_transforms = []
            for sampling_factor in sampling_params['sampling_factors']:
                simplifier = MeshSimplifier(in_mesh=m, debug=show_meshes)
                m, down, up = simplifier(sampling_factor, r_weighted)
                low_res_templates.append(m)
                down_transforms.append(down)
                up_transforms.append(up)

            with open(storage_path, 'wb') as file:
                pickle.dump(
                    [low_res_templates, down_transforms, up_transforms], file)

        down_transforms = [d.to(self.device) for d in down_transforms]
        up_transforms = [u.to(self.device) for u in up_transforms]
        return low_res_templates, down_transforms, up_transforms

    def _precompute_spirals(self, templates):
        storage_path = os.path.join(self._precomputed_storage_path,
                                    'spirals.pkl')
        try:
            with open(storage_path, 'rb') as file:
                spiral_indices_list = pickle.load(file)
        except FileNotFoundError:
            print("Computing Spirals")
            spirals_params = self._model_params['spirals']
            spiral_indices_list = []
            for i in range(len(templates) - 1):
                spiral_indices_list.append(
                    preprocess_spiral(templates[i].face.t().cpu().numpy(),
                                      spirals_params['length'][i],
                                      templates[i].pos.cpu().numpy(),
                                      spirals_params['dilation'][i]))
            with open(storage_path, 'wb') as file:
                pickle.dump(spiral_indices_list, file)
        spiral_indices_list = [s.to(self.device) for s in spiral_indices_list]
        return spiral_indices_list

    def _compute_latent_regions(self):
        region_names = list(self.template.feat_and_cont.keys())
        latent_size = self._model_params['latent_size']
        assert latent_size % len(region_names) == 0
        region_size = latent_size // len(region_names)
        return {k: [i * region_size, (i + 1) * region_size]
                for i, k in enumerate(region_names)}

    def forward(self, data):
        return self._net(data.x)

    @torch.no_grad()
    def encode(self, data):
        self._net.eval()
        return self._net.encode(data)[0]

    @torch.no_grad()
    def generate(self, z):
        self._net.eval()
        return self._net.decode(z)

    def generate_for_opt(self, z):
        self._net.train()
        return self._net.decode(z)

    def run_epoch(self, data_loader, device, train=True):
        if train:
            self._net.train()
        else:
            self._net.eval()

        self._reset_losses()
        it = 0
        for it, data in enumerate(data_loader):
            if train:
                losses = self._do_iteration(data, device, train=True)
            else:
                with torch.no_grad():
                    losses = self._do_iteration(data, device, train=False)
            self._add_losses(losses)
        self._divide_losses(it + 1)

    def _do_iteration(self, data, device='cpu', train=True):
        if train:
            self._optimizer.zero_grad()
            if self._classifier_params and self._mlp_classifier_end2end:
                self._classifier_optimizer.zero_grad()

        data = data.to(device)
        reconstructed, z, mu, logvar = self.forward(data)
        loss_recon = self.compute_mse_loss(reconstructed, data.x)
        loss_laplacian = self._compute_laplacian_regularizer(reconstructed)

        if self._w_kl_loss > 0:
            loss_kl = self._compute_kl_divergence_loss(mu, logvar)
        else:
            loss_kl = torch.tensor(0, device=device)

        if self._swap_features:
            loss_z_cons = self._compute_latent_consistency(z, data.swapped)
        else:
            loss_z_cons = torch.tensor(0, device=device)

        if self._classifier_params and self._mlp_classifier_end2end:
            if self._swap_features:
                z = z[self._batch_diagonal_idx, ::]
                y_gt = list(np.array(data.y)[self._batch_diagonal_idx])
            else:
                y_gt = data.y
            y_pred, y_pred_label = self.classifier_mlp(z)
            loss_class, acc_class = self.compute_classification_loss_and_acc(
                y_pred, y_pred_label, y_gt)
        else:
            loss_class = torch.tensor(0, device=device)
            acc_class = torch.tensor(0, device=device)

        loss_tot = loss_recon + \
            self._w_kl_loss * loss_kl + \
            self._w_latent_cons_loss * loss_z_cons + \
            self._w_laplacian_loss * loss_laplacian + \
            self._w_classifier_loss * loss_class

        if train:
            loss_tot.backward()
            self._optimizer.step()
            if self._classifier_params and self._mlp_classifier_end2end:
                self._classifier_optimizer.step()

        return {'reconstruction': loss_recon.item(),
                'kl': loss_kl.item(),
                'latent_consistency': loss_z_cons.item(),
                'laplacian': loss_laplacian.item(),
                'classification': loss_class.item(),
                'classification_acc': acc_class.item(),
                'tot': loss_tot.item()}

    @staticmethod
    def _compute_l1_loss(prediction, gt, reduction='mean'):
        return torch.nn.L1Loss(reduction=reduction)(prediction, gt)

    @staticmethod
    def compute_mse_loss(prediction, gt, reduction='mean'):
        return torch.nn.MSELoss(reduction=reduction)(prediction, gt)

    def compute_classification_loss_and_acc(self, y_pred, y_pred_label, y_gt):
        y_gt_tens = torch.tensor(self.class2idx(y_gt), device=y_pred.device)
        class_weights = self._class_weights.to(y_gt_tens.device)
        loss = torch.nn.CrossEntropyLoss(class_weights)(y_pred, y_gt_tens)
        acc = 100 * torch.sum(y_pred_label == y_gt_tens) / len(y_gt)
        return loss, acc

    def _compute_laplacian_regularizer(self, prediction):
        bs = prediction.shape[0]
        n_verts = prediction.shape[1]
        laplacian = self.template.laplacian.to(prediction.device)
        prediction_laplacian = utils.batch_mm(laplacian, prediction)
        loss = prediction_laplacian.norm(dim=-1) / n_verts
        return loss.sum() / bs

    @staticmethod
    def _compute_kl_divergence_loss(mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(kl, dim=0)

    @staticmethod
    def _compute_embedding_loss(z):
        return (z ** 2).mean(dim=1)

    def _compute_latent_consistency(self, z, swapped_feature):
        bs = self._optimization_params['batch_size']
        eta1 = self._optimization_params['latent_consistency_eta1']
        eta2 = self._optimization_params['latent_consistency_eta2']
        latent_region = self._latent_regions[swapped_feature]
        z_feature = z[:, latent_region[0]:latent_region[1]].view(bs, bs, -1)
        z_else = torch.cat([z[:, :latent_region[0]],
                            z[:, latent_region[1]:]], dim=1).view(bs, bs, -1)
        triu_indices = torch.triu_indices(
            z_feature.shape[0], z_feature.shape[0], 1)

        lg = z_feature.unsqueeze(0) - z_feature.unsqueeze(1)
        lg = lg[triu_indices[0], triu_indices[1], :, :].reshape(-1,
                                                                lg.shape[-1])
        lg = torch.sum(lg ** 2, dim=-1)

        dg = z_feature.permute(1, 2, 0).unsqueeze(0) - \
            z_feature.permute(1, 2, 0).unsqueeze(1)
        dg = dg[triu_indices[0], triu_indices[1], :, :].permute(0, 2, 1)
        dg = torch.sum(dg.reshape(-1, dg.shape[-1]) ** 2, dim=-1)

        dr = z_else.unsqueeze(0) - z_else.unsqueeze(1)
        dr = dr[triu_indices[0], triu_indices[1], :, :].reshape(-1,
                                                                dr.shape[-1])
        dr = torch.sum(dr ** 2, dim=-1)

        lr = z_else.permute(1, 2, 0).unsqueeze(0) - \
            z_else.permute(1, 2, 0).unsqueeze(1)
        lr = lr[triu_indices[0], triu_indices[1], :, :].permute(0, 2, 1)
        lr = torch.sum(lr.reshape(-1, lr.shape[-1]) ** 2, dim=-1)
        zero = torch.tensor(0, device=z.device)
        return (1 / (bs ** 3 - bs ** 2)) * \
               (torch.sum(torch.max(zero, lr - dr + eta2)) +
                torch.sum(torch.max(zero, lg - dg + eta1)))

    def compute_vertex_errors(self, out_verts, gt_verts):
        vertex_errors = self.compute_mse_loss(
            out_verts, gt_verts, reduction='none')
        vertex_errors = torch.sqrt(torch.sum(vertex_errors, dim=-1))
        vertex_errors *= self.to_mm_const
        return vertex_errors

    @torch.no_grad()
    def encode_all(self, data_loader, is_train_loader=True):
        latents_list = []
        dict_labels_lists = {'y': [], 'age': [], 'gender': [], 'augmented': []}
        for data in data_loader:
            if self._swap_features:
                b_ids = self._batch_diagonal_idx
                x = data.x[self._batch_diagonal_idx, ::]
                labels = {'y': np.array(data.y)[b_ids],
                          'age': data.age[b_ids, 0],
                          'gender': np.array(data.gender)[b_ids],
                          'augmented': data.augmented[b_ids, 0]}
            else:
                x = data.x
                labels = {'y': data.y,
                          'age': data.age[:, 0],
                          'gender': data.gender,
                          'augmented': data.augmented[:, 0]}
            latents_list.append(self.encode(x.to(self.device)).detach().cpu())
            for k, lbs in labels.items():
                dict_labels_lists[k].append(lbs)
        if is_train_loader:
            self._train_latents_list = latents_list
            self._train_dict_labels_lists = dict_labels_lists
        return latents_list, dict_labels_lists

    def mlp_classifier_epoch(self, latents_list, labels_list, train=True):
        epoch_loss = 0
        epoch_acc = 0

        for z, label in zip(latents_list, labels_list['y']):
            if train:
                self._classifier_optimizer.zero_grad()

            y_pred, y_pred_label = self.classifier_mlp(z.to(self.device))
            loss_class, acc_class = self.compute_classification_loss_and_acc(
                y_pred, y_pred_label, label)

            if train:
                loss_class.backward()
                self._classifier_optimizer.step()

            epoch_loss += loss_class.item()
            epoch_acc += acc_class.item()
        return epoch_loss / len(latents_list), epoch_acc / len(latents_list)

    def train_and_validate_classifiers(self, train_loader, validation_loader,
                                       writer, checkpoint_dir):
        if self._train_latents_list is None:
            self.encode_all(train_loader, is_train_loader=True)
        val_latents_list, val_l_list = self.encode_all(validation_loader, False)

        print("Training classifiers")

        # MLP
        if not self._mlp_classifier_end2end:
            tot_epochs_mlp = self._classifier_params['mlp_epochs']
            for epoch in tqdm.tqdm(range(tot_epochs_mlp)):
                tr_loss, tr_acc = self.mlp_classifier_epoch(
                    self._train_latents_list, self._train_dict_labels_lists)
                val_loss, val_acc = self.mlp_classifier_epoch(
                    val_latents_list, val_l_list, False)
                writer.add_scalar("train/class_loss", tr_loss, epoch + 1)
                writer.add_scalar("train/class_acc", tr_acc, epoch + 1)
                writer.add_scalar("validation/class_loss", val_loss, epoch + 1)
                writer.add_scalar("validation/class_acc", val_acc, epoch + 1)
            accuracy_mlp = self.mlp_classifier_epoch(val_latents_list,
                                                     val_l_list, False)[1]
            print(f"MLP validation accuracy = {accuracy_mlp}")
            self._save_classifier(checkpoint_dir, 'mlp')

        latents = torch.cat(self._train_latents_list, dim=0).numpy()
        y_gt = self.class2idx(
            np.concatenate(self._train_dict_labels_lists['y']))
        latents_val = torch.cat(val_latents_list, dim=0).numpy()
        y_gt_val = self.class2idx(np.concatenate(val_l_list['y']))

        # SVM
        self.classifier_svm.fit(latents, y_gt)
        accuracy_svm = self.classifier_svm.score(latents_val, y_gt_val)
        print(f"SVM validation accuracy = {accuracy_svm}")
        self._save_classifier(checkpoint_dir, 'svm')

        # LDA
        self.lda.fit(latents, y_gt)
        accuracy_lda = self.lda.score(latents_val, y_gt_val)
        print(f"LDA validation accuracy = {accuracy_lda}")
        self._save_classifier(checkpoint_dir, 'lda')

        # QDA
        self.qda.fit(latents, y_gt)
        accuracy_qda = self.qda.score(latents_val, y_gt_val)
        print(f"QDA validation accuracy = {accuracy_qda}")
        self._save_classifier(checkpoint_dir, 'qda')

        # region LDAs and QDAs
        if self._w_latent_cons_loss > 0:
            for key, z_region in self.latent_regions.items():
                z_np_region = latents[:, z_region[0]:z_region[1]]
                self.region_ldas[key].fit(z_np_region, y_gt)
                self.region_qdas[key].fit(z_np_region, y_gt)
            self._save_region_classifiers(checkpoint_dir, 'lda')
            self._save_region_classifiers(checkpoint_dir, 'qda')

    def lda_project_latents_in_2d(self, latents):
        return self.lda.transform(latents)

    def qda_sample(self, sample_class='a', n_samples=1):
        if isinstance(sample_class, str):
            sample_class = self.class2idx(sample_class)
        mean = self.qda.means_[sample_class]
        cov = self.qda.covariance_[sample_class]
        return np.random.multivariate_normal(mean, cov, n_samples)

    @torch.no_grad()
    def classify_latent(self, z, model='main'):
        if model == 'main':
            model = self._classifier_params['main_model_type']

        if model == 'mlp':
            y_pred = self.classifier_mlp(z.to(self.device)).detach().numpy()
        elif model == 'svm':
            y_pred = self.classifier_svm.predict(z.detach().cpu().numpy())
        elif model == 'lda':
            y_pred = self.lda.predict(z.detach().cpu().numpy())
        elif model == 'qda':
            y_pred = self.qda.predict(z.detach().cpu().numpy())
        else:
            raise NotImplementedError
        return self.idx2class(y_pred)

    def set_class_conversions_and_weights(self, data_c_and_w):
        data_c_and_w['b'] = data_c_and_w.pop('b')  # b always last
        self._class2idx_dict = {k: i for i, k in enumerate(data_c_and_w.keys())}
        idx2class = {v: k for k, v in self._class2idx_dict.items()}
        n_classes = len(data_c_and_w.keys())
        list_weights = [data_c_and_w[idx2class[i]] for i in range(n_classes)]
        self._class_weights = torch.tensor(list_weights)
        self._class_weights /= self._class_weights.sum()

    def set_class_conversions(self, class2idx_dict):
        self._class2idx_dict = class2idx_dict

    def class2idx(self, data_class):
        if isinstance(data_class, list) or isinstance(data_class, np.ndarray):
            idxs = [self._class2idx_dict[d] for d in data_class]
        else:
            idxs = self._class2idx_dict[data_class]
        return idxs

    def idx2class(self, idx):
        idx2class_dict = {v: k for k, v in self._class2idx_dict.items()}
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            c = [idx2class_dict[i] for i in idx]
        else:
            c = idx2class_dict[idx]
        return c

    def _reset_losses(self):
        self._losses = {k: 0 for k in self.loss_keys}

    def _add_losses(self, additive_losses):
        for k in self.loss_keys:
            loss = additive_losses[k]
            self._losses[k] += loss.item() if torch.is_tensor(loss) else loss

    def _divide_losses(self, value):
        for k in self.loss_keys:
            self._losses[k] /= value

    def log_losses(self, writer, epoch, phase='train'):
        for k in self.loss_keys:
            loss = self._losses[k]
            loss = loss.item() if torch.is_tensor(loss) else loss
            writer.add_scalar(
                phase + '/' + str(k), loss, epoch + 1)

    def log_images(self, in_data, writer, epoch, normalization_dict=None,
                   phase='train', error_max_scale=5):
        gt_meshes = in_data.x.to(self._rend_device)
        out_meshes = self.forward(in_data.to(self.device))[0]
        out_meshes = out_meshes.to(self._rend_device)

        if self._normalized_data:
            mean_mesh = normalization_dict['mean'].to(self._rend_device)
            std_mesh = normalization_dict['std'].to(self._rend_device)
            gt_meshes = gt_meshes * std_mesh + mean_mesh
            out_meshes = out_meshes * std_mesh + mean_mesh

        vertex_errors = self.compute_vertex_errors(out_meshes, gt_meshes)

        gt_renders = self.render(gt_meshes)
        out_renders = self.render(out_meshes)
        errors_renders = self.render(out_meshes, vertex_errors,
                                     error_max_scale)
        log = torch.cat([gt_renders, out_renders, errors_renders], dim=-1)
        log = make_grid(log, padding=10, pad_value=1, nrow=self._out_grid_size)
        writer.add_image(tag=phase, global_step=epoch + 1, img_tensor=log)

    def _create_renderer(self, img_size=256):
        raster_settings = RasterizationSettings(image_size=img_size)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings,
                                      cameras=FoVPerspectiveCameras()),
            shader=self.default_shader)
        renderer.to(self._rend_device)
        return renderer

    @torch.no_grad()
    def render(self, batched_data, vertex_errors=None, error_max_scale=None):
        batch_size = batched_data.shape[0]
        batched_verts = batched_data.detach().to(self._rend_device)
        template = self.template.to(self._rend_device)

        if vertex_errors is not None:
            self.renderer.shader = self.simple_shader
            textures = TexturesVertex(utils.errors_to_colors(
                vertex_errors, min_value=0,
                max_value=error_max_scale, cmap='plasma') / 255)
        else:
            self.renderer.shader = self.default_shader
            textures = TexturesVertex(torch.ones_like(batched_verts) * 0.5)

        meshes = Meshes(
            verts=batched_verts,
            faces=template.face.t().expand(batch_size, -1, -1),
            textures=textures)

        rotation, translation = look_at_view_transform(
            dist=2.5, elev=0, azim=15)
        cameras = FoVPerspectiveCameras(R=rotation, T=translation,
                                        device=self._rend_device)

        lights = PointLights(location=[[0.0, 0.0, 3.0]],
                             diffuse_color=[[1., 1., 1.]],
                             device=self._rend_device)

        materials = Materials(shininess=0.5, device=self._rend_device)

        images = self.renderer(meshes, cameras=cameras, lights=lights,
                               materials=materials).permute(0, 3, 1, 2)
        return images[:, :3, ::]

    def render_and_show_batch(self, data, normalization_dict):
        verts = data.x.to(self._rend_device)
        if self._normalized_data:
            mean_mesh = normalization_dict['mean'].to(self._rend_device)
            std_mesh = normalization_dict['std'].to(self._rend_device)
            verts = verts * std_mesh + mean_mesh
        rend = self.render(verts)
        grid = make_grid(rend, padding=10, pad_value=1,
                         nrow=self._out_grid_size)
        img = ToPILImage()(grid)
        img.show()

    def show_mesh(self, vertices, normalization_dict=None):
        vertices = torch.squeeze(vertices)
        if self._normalized_data:
            mean_verts = normalization_dict['mean'].to(vertices.device)
            std_verts = normalization_dict['std'].to(vertices.device)
            vertices = vertices * std_verts + mean_verts
        mesh = trimesh.Trimesh(vertices.cpu().detach().numpy(),
                               self.template.face.t().cpu().numpy())
        mesh.show()

    def save_weights(self, checkpoint_dir, epoch):
        net_name = os.path.join(checkpoint_dir, 'model_%08d.pt' % (epoch + 1))
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save({'model': self._net.state_dict()}, net_name)
        torch.save({'optimizer': self._optimizer.state_dict()}, opt_name)
        if self._classifier_params and self._mlp_classifier_end2end:
            self._save_classifier(checkpoint_dir, 'mlp')

    def resume(self, checkpoint_dir):
        last_model_name = utils.get_model_list(checkpoint_dir, 'model')
        state_dict = torch.load(last_model_name)
        self._net.load_state_dict(state_dict['model'])
        epochs = int(last_model_name[-11:-3])
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self._optimizer.load_state_dict(state_dict['optimizer'])
        if self._classifier_params is not None:
            self._resume_classifier(checkpoint_dir, 'mlp')
            self._resume_classifier(checkpoint_dir, 'svm')
            self._resume_classifier(checkpoint_dir, 'lda')
            self._resume_classifier(checkpoint_dir, 'qda')
            if self._w_latent_cons_loss > 0:
                self._resume_region_classifiers(checkpoint_dir, 'lda')
                self._resume_region_classifiers(checkpoint_dir, 'qda')
        print(f"Resume from epoch {epochs}")
        return epochs

    def _save_classifier(self, checkpoint_dir, classifier_type='mlp'):
        if classifier_type == 'mlp':
            net_name = os.path.join(checkpoint_dir, 'mlp_classifier.pt')
            torch.save({'model': self.classifier_mlp.state_dict()}, net_name)
        elif classifier_type == 'svm':
            svm_name = os.path.join(checkpoint_dir, 'svm_classifier.pkl')
            with open(svm_name, 'wb') as f:
                pickle.dump(self.classifier_svm, f)
        elif classifier_type == 'lda':
            lda_name = os.path.join(checkpoint_dir, 'lda_classifier.pkl')
            with open(lda_name, 'wb') as f:
                pickle.dump(self.lda, f)
        elif classifier_type == 'qda':
            qda_name = os.path.join(checkpoint_dir, 'qda_classifier.pkl')
            with open(qda_name, 'wb') as f:
                pickle.dump(self.qda, f)
        else:
            raise NotImplementedError

    def _resume_classifier(self, checkpoint_dir, classifier_type='mlp'):
        try:
            if classifier_type == 'mlp':
                net_name = os.path.join(checkpoint_dir, 'mlp_classifier.pt')
                state_dict = torch.load(net_name)
                self.classifier_mlp.load_state_dict(state_dict['model'])
            elif classifier_type == 'svm':
                svm_name = os.path.join(checkpoint_dir, 'svm_classifier.pkl')
                with open(svm_name, 'rb') as f:
                    self.classifier_svm = pickle.load(f)
            elif classifier_type == 'lda':
                lda_name = os.path.join(checkpoint_dir, 'lda_classifier.pkl')
                with open(lda_name, 'rb') as f:
                    self.lda = pickle.load(f)
            elif classifier_type == 'qda':
                qda_name = os.path.join(checkpoint_dir, 'qda_classifier.pkl')
                with open(qda_name, 'rb') as f:
                    self.qda = pickle.load(f)
            else:
                raise NotImplementedError
        except FileNotFoundError:
            print(f"Can't load classifier {classifier_type}: not trained yet.")

    def _save_region_classifiers(self, checkpoint_dir, classifier_type='lda'):
        if classifier_type == 'lda':
            r_ldas_name = os.path.join(checkpoint_dir, 'region_ldas.pkl')
            with open(r_ldas_name, 'wb') as f:
                pickle.dump(self.region_ldas, f)
        elif classifier_type == 'qda':
            r_qdas_name = os.path.join(checkpoint_dir, 'region_qdas.pkl')
            with open(r_qdas_name, 'wb') as f:
                pickle.dump(self.region_qdas, f)
        else:
            raise NotImplementedError

    def _resume_region_classifiers(self, checkpoint_dir, classifier_type='lda'):
        try:
            if classifier_type == 'lda':
                r_ldas_name = os.path.join(checkpoint_dir, 'region_ldas.pkl')
                with open(r_ldas_name, 'rb') as f:
                    self.region_ldas = pickle.load(f)
            elif classifier_type == 'qda':
                r_qdas_name = os.path.join(checkpoint_dir, 'region_qdas.pkl')
                with open(r_qdas_name, 'rb') as f:
                    self.region_qdas = pickle.load(f)
            else:
                raise NotImplementedError
        except FileNotFoundError:
            print(f"Can't load region classifiers {classifier_type}: "
                  f"not trained yet.")


class ShadelessShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = \
            blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        pixel_colors = meshes.sample_textures(fragments)
        images = hard_rgb_blend(pixel_colors, fragments, self.blend_params)
        return images
