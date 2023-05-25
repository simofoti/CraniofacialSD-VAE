import json
import os
import random
import tqdm
import trimesh
import torch

import numpy as np
import pandas as pd

from collections import Counter
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split

from swap_batch_transform import SwapFeatures
from utils import (get_dataset_summary, get_age_and_gender_from_summary,
                   find_data_used_from_summary, interpolate,
                   compute_laplacian_eigendecomposition,
                   spectral_combination, spectral_interpolation)


def get_data_loaders(config, template=None):
    data_config = config['data']
    batch_size = config['optimization']['batch_size']

    train_set = MeshInMemoryDataset(
        data_config['dataset_path'], data_config,
        dataset_type='train', template=template)
    validation_set = MeshInMemoryDataset(
        data_config['dataset_path'], data_config,
        dataset_type='val', template=template)
    test_set = MeshInMemoryDataset(
        data_config['dataset_path'], data_config,
        dataset_type='test', template=template)
    normalization_dict = train_set.normalization_dict

    swapper = SwapFeatures(template) if data_config['swap_features'] else None

    train_loader = MeshLoader(train_set, batch_size, shuffle=True,
                              drop_last=True, feature_swapper=swapper,
                              num_workers=data_config['number_of_workers'])
    validation_loader = MeshLoader(validation_set, batch_size, shuffle=True,
                                   drop_last=True, feature_swapper=swapper,
                                   num_workers=data_config['number_of_workers'])
    test_loader = MeshLoader(test_set, batch_size, shuffle=False,
                             drop_last=True, feature_swapper=swapper,
                             num_workers=data_config['number_of_workers'])
    data_classes_and_weights = train_set.classes_weights
    return train_loader, validation_loader, test_loader, \
        normalization_dict, data_classes_and_weights


class MeshLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 feature_swapper=None, **kwargs):
        collater = MeshCollater(feature_swapper)
        super(MeshLoader, self).__init__(dataset, batch_size, shuffle,
                                         collate_fn=collater, **kwargs)


class MeshCollater:
    def __init__(self, feature_swapper=None):
        self._swapper = feature_swapper

    def __call__(self, data_list):
        return self.collate(data_list)

    def collate(self, data_list):
        if not isinstance(data_list[0], Data):
            raise TypeError(
                f"DataLoader found invalid type: {type(data_list[0])}. "
                f"Expected torch_geometric.data.Data instead")

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        batched_data = Data()
        for key in keys:
            attribute_list = [data[key] for data in data_list]
            batched_data[key] = default_collate(attribute_list)
        if self._swapper is not None:
            batched_data = self._swapper(batched_data)
        return batched_data


class MeshInMemoryDataset(InMemoryDataset):
    def __init__(self, root, data_config, dataset_type='train',
                 transform=None, pre_transform=None, template=None):
        self._root = root
        self._data_config = data_config
        self._data_type = data_config['data_type']
        self._dataset_summary = get_dataset_summary(
            data_config, self._data_type)
        self._data_to_use = find_data_used_from_summary(
            self._dataset_summary, self._data_type)

        self._precomputed_storage_path = data_config['precomputed_path']
        if not os.path.isdir(self._precomputed_storage_path):
            os.mkdir(self._precomputed_storage_path)

        if 'stratified_split' in data_config:
            self._stratified_split = data_config['stratified_split']
        else:
            self._stratified_split = False

        self._dataset_type = dataset_type
        self._normalize = data_config['normalize_data']
        self._template = template

        self._train_names, self._test_names, self._val_names = self.split_data(
            os.path.join(self._precomputed_storage_path, 'data_split.json'))

        normalization_dict = self.compute_mean_and_std()
        self._normalization_dict = normalization_dict
        self.mean = normalization_dict['mean']
        self.std = normalization_dict['std']

        super(MeshInMemoryDataset, self).__init__(
            root, transform, pre_transform)

        self.classes_weights = self.compute_classes_and_weights(dataset_type)

        if dataset_type == 'train':
            data_path = self.processed_paths[0]
        elif dataset_type == 'test':
            data_path = self.processed_paths[1]
        elif dataset_type == 'val':
            data_path = self.processed_paths[2]
        else:
            raise Exception("train, val and test are supported data types")

        self.data, self.slices = torch.load(data_path)
        if self.transform:
            self.data = [self.transform(td) for td in self.data]

    @property
    def raw_file_names(self):
        return 'mesh_data.zip'

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt', 'val.pt']

    @property
    def normalization_dict(self):
        return self._normalization_dict

    @property
    def data_classes(self):
        return set([name[0] for name in self._train_names])

    def compute_classes_and_weights(self, dataset_type='train'):
        if dataset_type == 'train':
            all_names = self._train_names
        elif dataset_type == 'test':
            all_names = self._test_names
        else:
            all_names = self._val_names
        all_cls = [n.split('/')[1][0] if '/' in n else n[0] for n in all_names]
        n_aug_per_class = Counter(all_cls)
        return {k: 1 / v for k, v in n_aug_per_class.items()}

    def download(self):
        pass

    def find_filenames(self, find_augmented=True):
        files = []
        for dirpath, _, fnames in os.walk(self._root):
            for f in fnames:
                if f.endswith('.ply') or f.endswith('.obj'):
                    if 'aug' not in dirpath:
                        if self._data_to_use is None:
                            files.append(f)
                        elif f[:-4] in self._data_to_use:
                            files.append(f)
                    elif find_augmented:  # data augmented only with data_to_use
                        files.append(os.path.join('augmented', f))
        return files

    def split_data(self, data_split_list_path):
        try:
            with open(data_split_list_path, 'r') as fp:
                data = json.load(fp)
            train_list = data['train']
            test_list = data['test']
            val_list = data['val']
        except FileNotFoundError:
            all_file_names = self.find_filenames(find_augmented=False)
            all_file_names.sort()

            if self._stratified_split:
                y = [name[0] for name in all_file_names]
                train_list, test, _, test_y = train_test_split(
                    all_file_names, y, stratify=y, test_size=0.2)
                test_list, val_list, _, _ = train_test_split(
                    test, test_y, stratify=test_y, test_size=0.5)
            else:
                train_list, test_list, val_list = [], [], []
                for i, fname in enumerate(all_file_names):
                    if i % 100 <= 5:
                        test_list.append(fname)
                    elif i % 100 <= 10:
                        val_list.append(fname)
                    else:
                        train_list.append(fname)

            data_config = self._data_config
            augmentation_factor = data_config['augmentation_factor']
            if augmentation_factor > 0:
                train_list = self._augment(
                    train_list, mode=data_config['augmentation_mode'],
                    aug_factor=augmentation_factor,
                    balanced=data_config['augmentation_balanced'])

            data = {'train': train_list, 'test': test_list, 'val': val_list}
            with open(data_split_list_path, 'w') as fp:
                json.dump(data, fp)
        return train_list, test_list, val_list

    def load_mesh(self, filename, show=False):
        mesh_path = os.path.join(self._root, filename)
        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                                  requires_grad=False)
        if show:
            tm = trimesh.Trimesh(vertices=mesh.vertices,
                                 faces=self._template.face.t().cpu().numpy())
            tm.show()
        return mesh_verts

    def compute_mean_and_std(self):
        normalization_dict_path = os.path.join(
            self._precomputed_storage_path, 'norm.pt')
        try:
            normalization_dict = torch.load(normalization_dict_path)
        except FileNotFoundError:
            assert self._dataset_type == 'train'
            train_verts = None
            for i, fname in tqdm.tqdm(enumerate(self._train_names)):
                mesh_verts = self.load_mesh(fname)
                if i == 0:
                    train_verts = torch.zeros(
                        [len(self._train_names), mesh_verts.shape[0], 3],
                        requires_grad=False)
                train_verts[i, ::] = mesh_verts

            mean = torch.mean(train_verts, dim=0)
            std = torch.std(train_verts, dim=0)
            std = torch.where(std > 0, std, torch.tensor(1e-8))
            normalization_dict = {'mean': mean, 'std': std}
            torch.save(normalization_dict, normalization_dict_path)
        return normalization_dict

    def _process_set(self, files_list):
        dataset = []
        for fname in tqdm.tqdm(files_list):
            mesh_verts = self.load_mesh(fname)

            if self._normalize:
                mesh_verts = (mesh_verts - self.mean) / self.std

            age, gender = get_age_and_gender_from_summary(
                self._dataset_summary, fname[:-4])

            y = fname.split('/')[1][0] if '/' in fname else fname[0]
            y = 'n' if y == 'b' else y
            data = Data(x=mesh_verts, y=y,
                        augmented=('aug' in fname),
                        age=age, gender=gender)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            dataset.append(data)
        return dataset

    def process(self):
        train_data = self._process_set(self._train_names)
        torch.save(self.collate(train_data), self.processed_paths[0])
        test_data = self._process_set(self._test_names)
        torch.save(self.collate(test_data), self.processed_paths[1])
        val_data = self._process_set(self._val_names)
        torch.save(self.collate(val_data), self.processed_paths[2])

    def save_mean_mesh(self):
        first_mesh_path = os.path.join(self._root, self._train_names[0])
        first_mesh = trimesh.load_mesh(first_mesh_path, process=False)
        first_mesh.vertices = self.mean.detach().cpu().numpy()
        first_mesh.export(
            os.path.join(self._precomputed_storage_path, 'mean.ply'))

    def _augment(self, train_list, mode='interpolate', aug_factor=10,
                 balanced=True, split_3years=True):
        augmented_dir = os.path.join(self._root, 'augmented')
        if os.path.isdir(augmented_dir) and os.listdir(augmented_dir):
            aug_names = os.listdir(augmented_dir)
            n_aug_per_class = {cl: 0 for cl in set([n[0] for n in aug_names])}
            if self._dataset_type == 'train':
                for name in aug_names:
                    if name.endswith('.obj') or name.endswith('.ply'):
                        n_aug_per_class[name[0]] += 1
                        train_list.append(os.path.join('augmented', name))
                print(f"Found data previously augmented -> {n_aug_per_class}")
        elif self._dataset_type == 'train':
            initial_list = train_list.copy()

            if mode == 'spectral_comb' or mode == 'spectral_interp':
                # self._spectral_projections_analysis(initial_list, k=30)
                eigd = compute_laplacian_eigendecomposition(
                    self._template, k=1000)
            else:
                eigd = None

            data_classes = set([name[0] for name in initial_list])
            paths_age_gender_per_class = {cl: [] for cl in data_classes}
            for name in initial_list:
                age, gender = get_age_and_gender_from_summary(
                    self._dataset_summary, name[:-4])
                info = {'name': name, 'gender': gender, 'age': age}
                paths_age_gender_per_class[name[0]].append(info)

            # Merge paediatric and normal
            paths_age_gender_per_class['n'] += paths_age_gender_per_class['b']
            del paths_age_gender_per_class['b']

            if not os.path.isdir(augmented_dir):
                os.mkdir(augmented_dir)

            for c, info in paths_age_gender_per_class.items():
                if balanced:
                    n_aug_data = aug_factor * len(initial_list)
                    target_per_class = n_aug_data // len(data_classes)
                    n_aug_this_class = target_per_class - len(info)
                else:
                    n_aug_this_class = (aug_factor - 1) * len(info)

                info_df = pd.DataFrame(info).convert_dtypes()
                # NB: kids are 3 years old until their birthday (so 48 is used)
                less_3y = info_df.loc[info_df['age'] < 48].to_dict('records')
                more_3y = info_df.loc[info_df['age'] >= 48].to_dict('records')

                for i in range(n_aug_this_class):
                    if split_3years:
                        age_group_info = random.choice([less_3y, more_3y])
                    else:
                        age_group_info = info

                    selector = np.random.choice(len(age_group_info), 2,
                                                replace=False)
                    name1 = age_group_info[selector[0]]['name']
                    name2 = age_group_info[selector[1]]['name']
                    path1 = os.path.join(self._root, name1)
                    path2 = os.path.join(self._root, name2)
                    mesh1 = trimesh.load_mesh(path1, process=False)
                    mesh2 = trimesh.load_mesh(path2, process=False)
                    x1 = np.array(mesh1.vertices)
                    x2 = np.array(mesh2.vertices)

                    if mode == 'spectral_comb':
                        aug = '_spectral_comb' + str(i)
                        x_aug = spectral_combination(x1, x2, eigd)
                    elif mode == 'spectral_interp':
                        aug = '_spectral_interp' + str(i)
                        x_aug = spectral_interpolation(x1, x2, eigd)
                    else:
                        interpolation_value = np.random.uniform(size=1).item()
                        aug = '_interp' + f'{interpolation_value:.2f}'
                        x_aug = interpolate(x1, x2, interpolation_value)
                    mesh1.vertices = x_aug

                    aug_name = name1[:-4] + '_' + name2[2:-4] + aug + name1[-4:]
                    mesh1.export(os.path.join(augmented_dir, aug_name))
                    train_list.append(os.path.join('augmented', aug_name))
        return train_list

    def _spectral_projections_analysis(self, filenames, k=200,
                                       plot_type='scatter'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import Normalize
        from matplotlib.cm import get_cmap

        # normalize colors considering that maximum age is 20y (240m)
        # also 0m would be white, use negative ages to shift colours
        normalize_color = Normalize(vmin=-60.0, vmax=240.0)
        cmap_blue = get_cmap('Blues')
        cmap_red = get_cmap('Reds')

        s, u = compute_laplacian_eigendecomposition(self._template, k=k)
        spectral_proj_template = u.T @ self._template.pos.detach().cpu().numpy()
        data_classes = set([name[0] for name in filenames])
        data_classes = [c for c in data_classes if c != 'b']
        # data_classes = ['n', 'a', 'c', 'm']  # for consistent order in plots

        sns.set_theme(style="ticks")
        fig, axs = plt.subplots(len(data_classes), 3, figsize=(10, 10))

        for c_i, c in enumerate(data_classes):
            for name in filenames:
                if name[0] == c or (name[0] == 'b' and c == 'n'):
                    path = os.path.join(self._root, name)
                    age, gender = get_age_and_gender_from_summary(
                        self._dataset_summary, name[:-4])

                    mesh = trimesh.load_mesh(path, process=False)
                    spectral_proj = u.T @ np.array(mesh.vertices)
                    spectral_proj -= spectral_proj_template

                    x = np.arange(1, k + 1)
                    if gender == 'M':
                        line_colour = list(cmap_blue(normalize_color(age)))
                    else:
                        line_colour = list(cmap_red(normalize_color(age)))
                    line_colour[-1] = 0.7

                    # line_colour = 'b' if gender == 'M' else 'r'
                    line_colour = 'darkslategrey' if age > 12 * 4 else 'coral'

                    n = 'H' if c == 'n' else c.upper()
                    if plot_type == 'line':
                        axs[c_i, 0].set_title(f"{n}x")
                        axs[c_i, 0].plot(x, spectral_proj[:, 0],
                                         color=line_colour, linewidth=0.5)
                        axs[c_i, 1].set_title(f"{n}y")
                        axs[c_i, 1].plot(x, spectral_proj[:, 1],
                                         color=line_colour, linewidth=0.5)
                        axs[c_i, 2].set_title(f"{n}z")
                        axs[c_i, 2].plot(x, spectral_proj[:, 2],
                                         color=line_colour, linewidth=0.5)
                    else:
                        axs[c_i, 0].set_title(f"{n}x")
                        axs[c_i, 0].scatter(x, spectral_proj[:, 0], s=1,
                                            color=line_colour, linewidth=0.5)
                        axs[c_i, 1].set_title(f"{n}y")
                        axs[c_i, 1].scatter(x, spectral_proj[:, 1], s=1,
                                            color=line_colour, linewidth=0.5)
                        axs[c_i, 2].set_title(f"{n}z")
                        axs[c_i, 2].scatter(x, spectral_proj[:, 2], s=1,
                                            color=line_colour, linewidth=0.5)

        for ax in axs.flat:
            ax.set(xlabel='spectral components', ylabel='value')
        for ax in axs.flat:
            ax.label_outer()
        plt.savefig(os.path.join(self._root, "processed",
                                 "spectral_proj_analysis.svg"))


if __name__ == '__main__':
    import argparse
    import utils

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configurations/default.yaml',
                        help="Path to the configuration file.")
    opts = parser.parse_args()
    conf = utils.get_config(opts.config)

    tpl = utils.load_template(conf['data']['template_path'])
    # BodyGenerator('/home/simo/Desktop').save_mean_mesh()
    tr_loader, val_loader, te_loader, norm_d, _ = get_data_loaders(conf, tpl)
    # tr_loader.dataset.save_mean_mesh()
    # fnames = tr_loader.dataset.find_filenames(find_augmented=False)
    # tr_loader.dataset._spectral_projections_analysis(fnames, k=30)
