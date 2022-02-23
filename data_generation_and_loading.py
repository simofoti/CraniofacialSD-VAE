import json
import os
import random
import pickle
import tqdm
import trimesh
import torch

import numpy as np
import pandas as pd

from abc import abstractmethod
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Dataset, InMemoryDataset, Data
from sklearn.model_selection import train_test_split

from swap_batch_transform import SwapFeatures
from utils import (get_dataset_summary, get_age_and_gender_from_summary,
                   find_data_used_from_summary, interpolate,
                   compute_laplacian_eigendecomposition,
                   spectral_combination, spectral_interpolation)


class DataGenerator:
    def __init__(self, model_dir, data_dir='./data'):
        self._model_dir = model_dir
        self._data_dir = data_dir

    def __call__(self, number_of_meshes, weight=1., overwrite_data=False):
        if not os.path.isdir(self._data_dir):
            os.mkdir(self._data_dir)

        if not os.listdir(self._data_dir) or overwrite_data:  # directory empty
            print("Generating Data from PCA")
            for i in tqdm.tqdm(range(number_of_meshes)):
                v = self.generate_random_vertices(weight)
                self.save_vertices(v, str(i))

    def save_vertices(self, vertices, name):
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        m = trimesh.Trimesh(vertices, process=False)
        m.export(os.path.join(self._data_dir, name + '.ply'))

    @abstractmethod
    def generate_random_vertices(self, weight):
        pass


class FaceGenerator(DataGenerator):
    def __init__(self, uhm_path, data_dir='./data'):
        super(FaceGenerator, self).__init__(uhm_path, data_dir)
        infile = open(self._model_dir, 'rb')
        uhm_dict = pickle.load(infile)
        infile.close()

        self._components = uhm_dict['Eigenvectors'].shape[1]
        self._mu = uhm_dict['Mean']
        self._eigenvectors = uhm_dict['Eigenvectors']
        self._eigenvalues = uhm_dict['EigenValues']

    def generate_random_vertices(self, weight):
        w = weight * np.random.normal(size=self._components) * \
            self._eigenvalues ** 0.5
        w = np.expand_dims(w, axis=1)
        vertices = self._mu + self._eigenvectors @ w
        return vertices.reshape(-1, 3)


class BodyGenerator(DataGenerator):
    """ To install star model see https://github.com/ahmedosman/STAR"""
    def __init__(self, data_dir='./data'):
        super(BodyGenerator, self).__init__(None, data_dir)
        from star.pytorch.star import STAR
        self._star = STAR(gender='neutral', num_betas=10)

    def generate_random_vertices(self, weight=3):
        poses = torch.zeros([1, 72])
        betas = torch.rand([1, self._star.num_betas]) * 2 * weight - weight

        trans = torch.zeros([1, 3])
        verts = self._star.forward(poses.cuda(), betas.cuda(), trans.cuda())[-1]
        # Normalize verts in -1, 1 wrt height
        y_min = torch.min(verts[:, 1])
        scale = 2 / (torch.max(verts[:, 1]) - y_min)
        verts[:, 1] -= y_min
        verts *= scale
        verts[:, 1] -= 1
        return verts

    def save_mean_mesh(self):
        t = trimesh.Trimesh(self._star.v_template.cpu().numpy(),
                            self._star.f, process=False)
        t.export(os.path.join(self._data_dir, 'template.ply'))


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
    data_classes = train_set.data_classes
    return train_loader, validation_loader, test_loader, \
        normalization_dict, data_classes


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


class MeshDataset(Dataset):
    def __init__(self, root, data_config, dataset_type='train',
                 transform=None, pre_transform=None, template=None):
        self._root = root
        self._dataset_summary = get_dataset_summary(data_config)
        self._data_to_use = find_data_used_from_summary(
            self._dataset_summary, data_config['data_type'])

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

        self._processed_files = [f + '.pt' for f in self.raw_file_names]

        normalization_dict = self.compute_mean_and_std()
        self._normalization_dict = normalization_dict
        self.mean = normalization_dict['mean']
        self.std = normalization_dict['std']
        super(MeshDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        if self._dataset_type == 'train':
            file_names = self._train_names
        elif self._dataset_type == 'test':
            file_names = self._test_names
        elif self._dataset_type == 'val':
            file_names = self._val_names
        else:
            raise Exception("train, val and test are supported dataset types")
        return file_names

    @property
    def processed_file_names(self):
        return self._processed_files

    @property
    def normalization_dict(self):
        return self._normalization_dict

    def download(self):
        pass

    def find_filenames(self):
        files = []
        for dirpath, _, fnames in os.walk(self._root):
            for f in fnames:
                if f.endswith('.ply') or f.endswith('.obj') and 'aug' not in f:
                    if self._data_to_use is None:
                        files.append(f)
                    elif f[2:-4] in self._data_to_use:
                        files.append(f)
        return files

    def split_data(self, data_split_list_path):
        try:
            with open(data_split_list_path, 'r') as fp:
                data = json.load(fp)
            train_list = data['train']
            test_list = data['test']
            val_list = data['val']
        except FileNotFoundError:
            all_file_names = self.find_filenames()
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

            data = {'train': train_list, 'test': test_list, 'val': val_list}
            with open(data_split_list_path, 'w') as fp:
                json.dump(data, fp)
        return train_list, test_list, val_list

    def load_mesh(self, filename):
        mesh_path = os.path.join(self._root, filename)
        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                                  requires_grad=False)
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

    def process(self):
        for i, fname in tqdm.tqdm(enumerate(self.raw_file_names)):
            mesh_verts = self.load_mesh(fname)

            if self._normalize:
                mesh_verts = (mesh_verts - self.mean) / self.std

            data = Data(x=mesh_verts, y=fname[0], augmented=('aug' in fname))

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, os.path.join(self.processed_dir,
                                          fname[:-4] + '.pt'))

    def get(self, idx):
        filename = self.raw_file_names[idx]
        return torch.load(os.path.join(self.processed_dir,
                                       filename[:-4] + '.pt'))

    def len(self):
        return len(self.processed_file_names)

    def save_mean_mesh(self):
        first_mesh_path = os.path.join(self._root, self._train_names[0])
        first_mesh = trimesh.load_mesh(first_mesh_path, process=False)
        first_mesh.vertices = self.mean.detach().cpu().numpy()
        first_mesh.export(
            os.path.join(self._precomputed_storage_path, 'mean.ply'))


class MeshInMemoryDataset(InMemoryDataset):
    def __init__(self, root, data_config, dataset_type='train',
                 transform=None, pre_transform=None, template=None):
        self._root = root
        self._data_type = data_config['data_type']
        self._dataset_summary = get_dataset_summary(data_config)
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

        if dataset_type == 'train':
            self._augment(mode=data_config['augmentation_mode'],
                          aug_factor=data_config['augmentation_factor'],
                          balanced=data_config['augmentation_balanced'])

        super(MeshInMemoryDataset, self).__init__(
            root, transform, pre_transform)

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

    def download(self):
        pass

    def find_filenames(self):
        files = []
        for dirpath, _, fnames in os.walk(self._root):
            for f in fnames:
                if f.endswith('.ply') or f.endswith('.obj') and 'aug' not in f:
                    if self._data_to_use is None:
                        files.append(f)
                    elif f[2:-4] in self._data_to_use:
                        files.append(f)
        return files

    def split_data(self, data_split_list_path):
        try:
            with open(data_split_list_path, 'r') as fp:
                data = json.load(fp)
            train_list = data['train']
            test_list = data['test']
            val_list = data['val']
        except FileNotFoundError:
            all_file_names = self.find_filenames()
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
                self._dataset_summary, fname[2:-4], self._data_type)

            data = Data(x=mesh_verts, y=fname[0],
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

    def _augment(self, mode='interpolate', aug_factor=10, balanced=True,
                 split_3years=True):
        augmented_dir = os.path.join(self._root, 'augmented')
        if os.path.isdir(augmented_dir) and os.listdir(augmented_dir):
            aug_names = os.listdir(augmented_dir)
            n_aug_per_class = {cl: 0 for cl in set([n[0] for n in aug_names])}
            for name in aug_names:
                if name.endswith('.obj') or name.endswith('.ply'):
                    self._train_names.append(os.path.join('augmented', name))
                    n_aug_per_class[name[0]] += 1
            print(f"Found data previously augmented. Using {n_aug_per_class}")
        else:
            if mode == 'spectral_comb' or mode == 'spectral_interp':
                self._spectral_projections_analysis(k=30)
                eigd = compute_laplacian_eigendecomposition(
                    self._template, k=1000)
            else:
                eigd = None

            initial_list = self._train_names.copy()
            data_classes = set([name[0] for name in initial_list])
            paths_age_gender_per_class = {cl: [] for cl in data_classes}
            for name in initial_list:
                age, gender = get_age_and_gender_from_summary(
                    self._dataset_summary, name[2:-4], self._data_type)
                info = {'name': name, 'gender': gender, 'age': age}
                paths_age_gender_per_class[name[0]].append(info)

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
                    self._train_names.append(
                        os.path.join('augmented', aug_name))

    def _spectral_projections_analysis(self, k=200, plot_type='scatter'):
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import get_cmap

        # normalize colors considering that maximum age is 20y (240m)
        # also 0m would be white, use negative ages to shift colours
        normalize_color = Normalize(vmin=-60.0, vmax=240.0)
        cmap_blue = get_cmap('Blues')
        cmap_red = get_cmap('Reds')

        s, u = compute_laplacian_eigendecomposition(self._template, k=k)
        spectral_proj_template = u.T @ self._template.pos.detach().cpu().numpy()
        data_classes = set([name[0] for name in self._train_names])

        fig, axs = plt.subplots(len(data_classes), 3, figsize=(10, 10))

        for c_i, c in enumerate(data_classes):
            for name in self._train_names:
                if name[0] == c:
                    path = os.path.join(self._root, name)
                    age, gender = get_age_and_gender_from_summary(
                        self._dataset_summary, name[2:-4], self._data_type)

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
                    # line_colour = 'darkslategrey' if age > 12 * 3 else 'coral'

                    if plot_type == 'line':
                        axs[c_i, 0].set_title(f"{c}_s1")
                        axs[c_i, 0].plot(x, spectral_proj[:, 0],
                                         color=line_colour, linewidth=0.5)
                        axs[c_i, 1].set_title(f"{c}_s2")
                        axs[c_i, 1].plot(x, spectral_proj[:, 1],
                                         color=line_colour, linewidth=0.5)
                        axs[c_i, 2].set_title(f"{c}_s3")
                        axs[c_i, 2].plot(x, spectral_proj[:, 2],
                                         color=line_colour, linewidth=0.5)
                    else:
                        axs[c_i, 0].set_title(f"{c}_s1")
                        axs[c_i, 0].scatter(x, spectral_proj[:, 0], s=5,
                                            color=line_colour, linewidth=0.5)
                        axs[c_i, 1].set_title(f"{c}_s2")
                        axs[c_i, 1].scatter(x, spectral_proj[:, 1], s=5,
                                            color=line_colour, linewidth=0.5)
                        axs[c_i, 2].set_title(f"{c}_s3")
                        axs[c_i, 2].scatter(x, spectral_proj[:, 2], s=5,
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

    # BodyGenerator('/home/simo/Desktop').save_mean_mesh()
    tr_loader, val_loader, te_loader, norm_dict = get_data_loaders(conf, None)
    tr_loader.dataset.save_mean_mesh()
