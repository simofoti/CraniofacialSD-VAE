import os
import yaml
import trimesh
import torch

import matplotlib.cm
import torch_geometric.transforms

import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from pandas import read_excel
from scipy.linalg import eigh, norm
from scipy.sparse.linalg import eigsh
from matplotlib.colors import Normalize, ListedColormap, to_rgba_array
from matplotlib.patches import Ellipse


procedures2attributes_dict = {
    'foar': ['[238 206  74 255]', '[116 192 194 255]'],
    'genoplasty': ['[194 109  97 255]'],
    'le_front_I': ['[232 129 166 255]', '[ 89  51 139 255]'],
    'le_front_II': ['[232 129 166 255]', '[133 169 172 255]',
                    '[237 109  93 255]'],
    'le_front_III': ['[232 129 166 255]', '[133 169 172 255]',
                     '[237 109  93 255]', '[ 89  51 139 255]',
                     '[245 158  40 255]'],
    'mandibular_ost': ['[194 109  97 255]', '[164  78 123 255]'],
    'monobloc': ['[232 129 166 255]', '[133 169 172 255]',
                 '[237 109  93 255]', '[ 89  51 139 255]',
                 '[245 158  40 255]', '[ 26  81  82 255]',
                 '[238 206  74 255]', '[116 192 194 255]'],
    'orbital_ost': ['[133 169 172 255]', '[245 158  40 255]',
                    '[ 26  81  82 255]', '[238 206  74 255]'],
    'rhinoplasty': ['[237 109  93 255]'],
    'zygomatic_ost': ['[ 89  51 139 255]', '[245 158  40 255]']
}

colour2attribute_dict = {
    '[232 129 166 255]': 'upper lip',
    '[194 109  97 255]': 'chin',
    '[133 169 172 255]': 'nasolabial',
    '[237 109  93 255]': 'nose',
    '[ 89  51 139 255]': 'cheeks',
    '[245 158  40 255]': 'zygomatic',
    '[ 26  81  82 255]': 'eyes',
    '[164  78 123 255]': 'jaw',
    '[238 206  74 255]': 'supraorbital',
    '[ 18  78 129 255]': 'neck',
    '[245 160 106 255]': 'ears',
    '[116 192 194 255]': 'frontal',
    '[ 90  97 115 255]': 'occipital',
    '[164 184 207 255]': 'temporal',
    '[219 203 190 255]': 'parietal'
}


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


def get_dataset_summary(data_config, data_type):
    if 'dataset_summary_path' in data_config:
        d_summary = read_excel(data_config['dataset_summary_path'])
        d_summary['mesh_name'] = 'nan'
        d_summary.loc[d_summary['Dataset'] == 'Paeds', 'mesh_name'] = 'b'
        d_summary.loc[d_summary['Dataset'] == 'Apert', 'mesh_name'] = 'a'
        d_summary.loc[d_summary['Dataset'] == 'Crouzon', 'mesh_name'] = 'c'
        d_summary.loc[d_summary['Dataset'] == 'Muenke', 'mesh_name'] = 'm'
        d_summary.loc[d_summary['Dataset'] == 'LSFM', 'mesh_name'] = 'n'
        d_summary.loc[d_summary['Dataset'] == 'LYHM', 'mesh_name'] = 'n'
        id_column = 'ID' if data_type == 'heads' else 'PID'
        d_summary['mesh_name'] = d_summary['mesh_name'] + '_' + \
            d_summary[id_column].fillna(-1).astype(int).astype(str)
    else:
        d_summary = None
    return d_summary


def find_data_used_from_summary(summary, data_type):
    if summary is None:
        return None
    else:
        cond_column = "Head Used" if data_type == 'heads' else "Face Used"
        ids_to_use = summary.loc[summary[cond_column] == 'y']['mesh_name']
        return list(ids_to_use)


def get_age_and_gender_from_summary(summary, mesh_id):
    try:
        id_column = summary['mesh_name']
        age = summary.loc[id_column == mesh_id]['AgeMonths'].values[0]
        if np.isnan(age):
            age = summary.loc[id_column == mesh_id]['AgeYears'].values[0] * 12
            # Add half of a year. It is unlikely they just had their birthday...
            age += 6
        gender = summary.loc[id_column == mesh_id]['Gender'].values[0]
    except IndexError:  # should be triggered by augmented meshes
        age, gender = -1, 'n/a'
    return age, gender


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

    values = np.random.normal(loc=0.5, scale=0.5, size=[s1.shape[0], 1])
    s3 = s1 + values * (s2 - s1)

    interp_until = 30
    s4 = s1.copy()
    s4[:interp_until] = s3[:interp_until]
    return u @ s4


def get_per_vertex_eigenvector_color(eigenvec_matrix, eigenvec_n):
    cmap = matplotlib.cm.get_cmap("bwr")
    e_vec = eigenvec_matrix[:, eigenvec_n]
    colors = cmap(Normalize(vmin=e_vec.min(), vmax=e_vec.max())(e_vec))
    return colors


def create_alpha_cmap(base_color_name):
    vals = np.ones((256, 4))
    base_color = to_rgba_array(base_color_name)
    vals[:, 0] = np.linspace(1, base_color[0, 0], 256)
    vals[:, 1] = np.linspace(1, base_color[0, 1], 256)
    vals[:, 2] = np.linspace(1, base_color[0, 2], 256)
    vals[:10, 3] = np.linspace(0, 1, 10)
    return ListedColormap(vals)


def get_gaussian_ellipse(mean, covariance, color, n_sigma=3):
    v, w = eigh(covariance)
    u = w[0] / norm(w[0])
    angle = 180 * np.arctan(u[1] / u[0]) / np.pi

    ell = Ellipse(mean, n_sigma * v[0] ** 0.5, n_sigma * v[1] ** 0.5,
                  180 + angle, facecolor=color, edgecolor=color, linewidth=2)
    ell.set_alpha(0.2)
    return ell


def plot_confusion_matrix(data, labels, output_filename):
    sns.set(color_codes=True)
    ax = sns.heatmap(data, annot=True, cmap="YlGnBu", vmin=0., vmax=1.)

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
