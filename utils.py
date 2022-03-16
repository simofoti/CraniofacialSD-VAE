import os
import yaml
import trimesh
import torch

import matplotlib.cm
import torch_geometric.transforms

import networkx as nx
import numpy as np
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, from_trimesh
from pandas import read_excel
from scipy.sparse.linalg import eigsh


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def prepare_sub_folder(output_directory):
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print(f"Creating directory: {checkpoint_directory}")
        os.makedirs(checkpoint_directory)
    return checkpoint_directory


def load_template(mesh_path):
    mesh = trimesh.load_mesh(mesh_path, 'ply', process=False)
    feat_and_cont = extract_feature_and_contour_from_colour(mesh)
    mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float,
                              requires_grad=False)
    face = torch.from_numpy(mesh.faces).t().to(torch.long).contiguous()
    mesh_colors = torch.tensor(mesh.visual.vertex_colors,
                               dtype=torch.float, requires_grad=False)
    data = Data(pos=mesh_verts, face=face, colors=mesh_colors,
                feat_and_cont=feat_and_cont)
    data = torch_geometric.transforms.FaceToEdge(False)(data)
    data.laplacian = torch.sparse_coo_tensor(
        *get_laplacian(data.edge_index, normalization='rw'))
    return data


def extract_feature_and_contour_from_colour(colored):
    # assuming that the feature is colored in red and its contour in black
    if isinstance(colored, torch_geometric.data.Data):
        assert hasattr(colored, 'colors')
        colored_trimesh = torch_geometric.utils.to_trimesh(colored)
        colors = colored.colors.to(torch.long).numpy()
    elif isinstance(colored, trimesh.Trimesh):
        colored_trimesh = colored
        colors = colored_trimesh.visual.vertex_colors
    else:
        raise NotImplementedError

    graph = nx.from_edgelist(colored_trimesh.edges_unique)
    one_rings_indices = [list(graph[i].keys()) for i in range(len(colors))]

    features = {}
    for index, (v_col, i_ring) in enumerate(zip(colors, one_rings_indices)):
        if str(v_col) not in features:
            features[str(v_col)] = {'feature': [], 'contour': []}

        if is_contour(colors, index, i_ring):
            features[str(v_col)]['contour'].append(index)
        else:
            features[str(v_col)]['feature'].append(index)

    # certain vertices on the contour have interpolated colours ->
    # assign them to adjacent region
    elem_to_remove = []
    for key, feat in features.items():
        if len(feat['feature']) < 3:
            elem_to_remove.append(key)
            for idx in feat['feature']:
                counts = Counter([str(colors[ri])
                                  for ri in one_rings_indices[idx]])
                most_common = counts.most_common(1)[0][0]
                if most_common == key:
                    break
                features[most_common]['feature'].append(idx)
                features[most_common]['contour'].append(idx)
    for e in elem_to_remove:
        features.pop(e, None)

    # with b map
    # 0=eyes, 1=ears, 2=sides, 3=neck, 4=back, 5=mouth, 6=forehead,
    # 7=cheeks 8=cheekbones, 9=forehead, 10=jaw, 11=nose
    # key = list(features.keys())[11]
    # feature_idx = features[key]['feature']
    # contour_idx = features[key]['contour']

    # find surroundings
    # all_distances = self.compute_minimum_distances(
    #     colored.vertices, colored.vertices[contour_idx]
    # )
    # max_distance = max(all_distances)
    # all_distances[feature_idx] = max_distance
    # all_distances[contour_idx] = max_distance
    # threshold = 0.005
    # surrounding_idx = np.squeeze(np.argwhere(all_distances < threshold))
    # colored.visual.vertex_colors[surrounding_idx] = [0, 0, 0, 255]
    # colored.show()
    return features


def is_contour(colors, center_index, ring_indices):
    center_color = colors[center_index]
    ring_colors = [colors[ri] for ri in ring_indices]
    for r in ring_colors:
        if not np.array_equal(center_color, r):
            return True
    return False


def to_torch_sparse(spmat):
    return torch.sparse_coo_tensor(
        torch.LongTensor([spmat.tocoo().row, spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def batch_mm(sparse, matrix_batch):
    """
    :param sparse: Sparse matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns (b, n, k) -> (n, b, k) -> (n, b*k)
    matrix = matrix_batch.transpose(0, 1).reshape(sparse.shape[1], -1)

    # And then reverse the reshaping.
    return sparse.mm(matrix).reshape(sparse.shape[0],
                                     batch_size, -1).transpose(1, 0)


def errors_to_colors(values, min_value=None, max_value=None, cmap=None):
    device = values.device
    min_value = values.min() if min_value is None else min_value
    max_value = values.max() if max_value is None else max_value
    if min_value != max_value:
        values = (values - min_value) / (max_value - min_value)

    cmapper = matplotlib.cm.get_cmap(cmap)
    values = cmapper(values.cpu().detach().numpy(), bytes=True)
    return torch.tensor(values[:, :, :3]).to(device)


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(
                      os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_dataset_summary(data_config):
    if 'dataset_summary_path' in data_config:
        dataset_summary = read_excel(data_config['dataset_summary_path'])
    else:
        dataset_summary = None
    return dataset_summary


def find_data_used_from_summary(summary, data_type):
    if summary is None:
        return None
    else:
        cond_column = "Head Used" if data_type == 'heads' else "Face Used"
        id_column = 'ID' if data_type == 'heads' else 'PID'
        ids_to_use = summary.loc[summary[cond_column] == 'y'][id_column]
        return list(ids_to_use.astype(int).astype(str))


def interpolate(x1, x2, value=0.5):
    return x1 + value * (x2 - x1)


def compute_laplacian_eigendecomposition(template, k=500):
    lapl = to_scipy_sparse_matrix(
        *get_laplacian(template.edge_index, normalization=None))
    return eigsh(lapl, k=k, which='SM')


def spectral_combination(x1, x2, eigendec):
    s, u = eigendec
    s1 = u.T @ x1
    s2 = u.T @ x2

    swap_until = 30  # one third of them are swapped    vvv
    selector = np.random.choice(swap_until, swap_until // 3, replace=False)
    s3 = s1.copy()
    s3[selector] = s2[selector]
    return u @ s3


def spectral_interpolation(x1, x2, eigendec):
    s, u = eigendec
    s1 = u.T @ x1
    s2 = u.T @ x2

    values = np.random.uniform(size=[s1.shape[0], 1])
    s3 = s1 + values * (s2 - s1)

    interp_until = 30
    s4 = s1.copy()
    s4[:interp_until] = s3[:interp_until]
    return u @ s4
