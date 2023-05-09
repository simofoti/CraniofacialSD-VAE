import os
import random
import json
import trimesh
import torch

from torch_geometric.data import Data

import utils
from model_manager import ModelManager
from swap_batch_transform import OriginalSwapFeatures


def add_vert_colors(verts, color):
    v_colors = torch.ones_like(verts) * torch.tensor(color)
    return torch.cat([verts, v_colors], dim=-1)


data_path = '/media/simo/DATASHURPRO/for_simone/head_face_meshes'
data_split_list_path = './precomputed_craniofacial_new/data_split.json'
with open(data_split_list_path, 'r') as fp:
    data = json.load(fp)
train_list = data['train']
# select only healthy and not augmented
train_list = [n for n in train_list if 'n_' in n and 'aug' not in n]

vertices = []
for i in range(4):
    path = os.path.join(data_path, random.choice(train_list))
    mesh = trimesh.load_mesh(path, process=False)
    vertices.append(torch.tensor(mesh.vertices, dtype=torch.float,
                                 requires_grad=False))

vertices[0] = add_vert_colors(vertices[0], [248, 161, 105])
vertices[1] = add_vert_colors(vertices[1], [243, 141, 155])
vertices[2] = add_vert_colors(vertices[2], [253, 230, 133])
vertices[3] = add_vert_colors(vertices[3], [114, 191, 193])

batch = Data(x=torch.stack(vertices))

config = utils.get_config('configurations/craniofacial.yaml')
manager = ModelManager(configurations=config, device='cpu')

swapped_batch = OriginalSwapFeatures(manager.template)(batch)

out_mesh_dir = 'outputs/craniofacial/in_batch'
if not os.path.isdir(out_mesh_dir):
    os.mkdir(out_mesh_dir)

for i in range(swapped_batch.x.shape[0]):
    mesh = trimesh.Trimesh(
        swapped_batch.x[i, :, :3].cpu().detach().numpy(),
        manager.template.face.t().cpu().numpy(),
        vertex_colors=swapped_batch.x[i, :, 3:].cpu().detach().numpy())
    mesh.export(os.path.join(out_mesh_dir, str(i) + '.ply'))

