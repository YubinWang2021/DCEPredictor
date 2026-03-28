from __future__ import division

import torch
import torch.nn as nn
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle
from renderer import Renderer
import os
import cv2

os.environ['PYOPENGL_PLATFORM'] = 'egl'
# import config as cfg
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat2mat(quat)


class SMPL(nn.Module):

    def __init__(self, model_file='../data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'):
        super(SMPL, self).__init__()
        with open(model_file, 'rb') as f:
            smpl_model = pickle.load(f, encoding="iso-8859-1")
        self.model = smpl_model
        J_regressor = smpl_model['J_regressor'].tocoo()
        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data
        i = torch.LongTensor([row, col])
        v = torch.FloatTensor(data)
        v = torch.FloatTensor(data)
        J_regressor_shape = [24, 6890]
        self.register_buffer('J_regressor', torch.sparse.FloatTensor(i, v, J_regressor_shape).to_dense())
        self.register_buffer('weights', torch.FloatTensor(smpl_model['weights']))
        self.register_buffer('posedirs', torch.FloatTensor(smpl_model['posedirs']))
        self.register_buffer('v_template', torch.FloatTensor(smpl_model['v_template']))
        self.register_buffer('shapedirs', torch.FloatTensor(np.array(smpl_model['shapedirs'])))
        self.register_buffer('faces', torch.from_numpy(smpl_model['f'].astype(np.int64)))
        self.register_buffer('kintree_table', torch.from_numpy(smpl_model['kintree_table'].astype(np.int64)))
        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]))

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.translation_shape = [3]

        self.pose = torch.zeros(self.pose_shape)
        self.beta = torch.zeros(self.beta_shape)
        self.translation = torch.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None

        J_regressor_extra = torch.from_numpy(np.load('../data/J_regressor_extra.npy')).float()
        semantic_prior = np.load('../data/semantic_prior.npy', mmap_mode='r')
        #self.prior_best_match = np.argmax(semantic_prior, axis=1)
        self.prior_best_match = np.load('../data/prior_best_match.npy', mmap_mode='r')
        self.register_buffer('J_regressor_extra', J_regressor_extra)
        self.joints_idx = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]

    def forward(self, pose, beta):
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1, 10)[None, :].expand(batch_size, -1, -1)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3)  # (batch_size * 24, 1, 3)
            R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1, 207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890, 3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights, G.permute(1, 0, 2, 3).contiguous().view(24, -1)).view(6890, batch_size, 4,
                                                                                             4).transpose(0, 1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        self.render(v.view(6890, 3), self.faces)
        v = v.reshape(6890, 3)
        print(v.shape)
        np.save('../data/v_0101_0105.npy', v)
        return v

    def render(self, verts, mesh_face):
        verts1 = torch.zeros(verts.shape)
        verts2 = torch.zeros(verts.shape)
        verts1[:len(verts) // 2] = verts[:len(verts) // 2]
        verts2[len(verts) // 2:] = verts[len(verts) // 2:]
        orig_img = np.zeros((224, 224, 3))
        # Setup renderer for visualization
        mesh_face0 = mesh_face[np.where(self.prior_best_match == 0)]
        mesh_face1 = mesh_face[np.where(self.prior_best_match == 1)]

        mesh_face2 = mesh_face[np.where(self.prior_best_match == 2)]

        mesh_face3 = mesh_face[np.where(self.prior_best_match == 3)]

        mesh_face4 = mesh_face[np.where(self.prior_best_match == 4)]

        mesh_face5 = mesh_face[np.where(self.prior_best_match == 5)]

        mesh_face6 = mesh_face[np.where(self.prior_best_match == 6)]

        mesh_face7 = mesh_face[np.where(self.prior_best_match == 7)]

        mesh_face8 = mesh_face[np.where(self.prior_best_match == 8)]

        mesh_face9 = mesh_face[np.where(self.prior_best_match == 9)]

        mesh_face10 = mesh_face[np.where(self.prior_best_match == 10)]

        mesh_face11 = mesh_face[np.where(self.prior_best_match == 11)]
        mesh_face12 = mesh_face[np.where(self.prior_best_match == 12)]
        mesh_face13 = mesh_face[np.where(self.prior_best_match == 13)]
        mesh_face14 = mesh_face[np.where(self.prior_best_match == 14)]
        mesh_face15 = mesh_face[np.where(self.prior_best_match == 15)]
        mesh_face16 = mesh_face[np.where(self.prior_best_match == 16)]
        mesh_face17 = mesh_face[np.where(self.prior_best_match == 17)]
        mesh_face18 = mesh_face[np.where(self.prior_best_match == 18)]
        mesh_face19 = mesh_face[np.where(self.prior_best_match == 19)]
        renderer2 = Renderer(mesh_face2, resolution=(224, 224), orig_img=True, wireframe=False)
        renderer5 = Renderer(mesh_face5, resolution=(224, 224), orig_img=True, wireframe=False)
        renderer9 = Renderer(mesh_face9, resolution=(224, 224), orig_img=True, wireframe=False)
        renderer10 = Renderer(mesh_face10, resolution=(224, 224), orig_img=True, wireframe=False)
        renderer13 = Renderer(mesh_face13, resolution=(224, 224), orig_img=True, wireframe=False)
        renderer14 = Renderer(mesh_face14, resolution=(224, 224), orig_img=True, wireframe=False)
        renderer15 = Renderer(mesh_face15, resolution=(224, 224), orig_img=True, wireframe=False)
        renderer18 = Renderer(mesh_face18, resolution=(224, 224), orig_img=True, wireframe=False)
        renderer19 = Renderer(mesh_face19, resolution=(224, 224), orig_img=True, wireframe=False)
        #hair
        rendered_img = renderer2.render(
            orig_img,
            verts,
            cam=(1.0, 1.0, 0.0, 0.0),
            color=(255 / 255, 0 / 255, 0 / 255),
            mesh_filename=None,
            rotate=False
        )


        rendered_img = renderer19.render(
            rendered_img,
            verts,
            cam=(1.0, 1.0, 0.0, 0.0),
            color=(255 / 255, 170 / 255, 0 / 255),
            mesh_filename=None,
            rotate=False
        )
        rendered_img = renderer18.render(
            rendered_img,
            verts,
            cam=(1.0, 1.0, 0.0, 0.0),
            color=(255 / 255, 255 / 255, 0 / 255),
            mesh_filename=None,
            rotate=False
        )


        rendered_img = renderer9.render(
            rendered_img,
            verts,
            cam=(1.0, 1.0, 0.0, 0.0),
            color=(0 / 255, 25 / 255, 25 / 255),
            mesh_filename=None,
            rotate=False
        )
        #neck
        rendered_img = renderer10.render(
            rendered_img,
            verts,
            cam=(1.0, 1.0, 0.0, 0.0),
            color=(85 / 255, 51 / 255, 0 / 255),
            mesh_filename=None,
            rotate=False
        )

        #right arm
        rendered_img = renderer15.render(
            rendered_img,
            verts,
            cam=(1.0, 1.0, 0.0, 0.0),
            color=(0 / 255, 255 / 255, 255 / 255),
            mesh_filename=None,
            rotate=False
        )
        # upper clothes
        rendered_img = renderer5.render(
            rendered_img,
            verts,
            cam=(1.0, 1.0, 0.0, 0.0),
            color=(255 / 255, 13 / 255, 0 / 255),
            mesh_filename=None,
            rotate=False
        )

        #left arm
        rendered_img = renderer14.render(
            rendered_img,
            verts,
            cam=(1.0, 1.0, 0.0, 0.0),
            color=(51 / 255, 170 / 255, 221 / 255),
            mesh_filename=None,
            rotate=False
        )
        #face
        rendered_img = renderer13.render(
            rendered_img,
            verts,
            cam=(1.0, 1.0, 0.0, 0.0),
            color=(0 / 255, 0 / 255, 255 / 255),
            mesh_filename=None,
            rotate=False
        )
        cv2.imwrite('./rendered_img.png', rendered_img[:, :, ::-1])
        return rendered_img
    def get_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
        joints_extra = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_extra])
        joints = torch.cat((joints, joints_extra), dim=1)
        joints = joints[:, self.joints_idx]
        return joints


if __name__ == '__main__':
    smpl = SMPL()
    rotmat = [[[[ 8.8316e-01, -1.7435e-01, -4.3547e-01],
                [-1.4457e-02, -9.3803e-01,  3.4624e-01],
                [-4.6885e-01, -2.9949e-01, -8.3096e-01]],

               [[ 9.9037e-01, -1.2769e-01,  5.3539e-02],
                [ 1.3107e-01,  9.8923e-01, -6.5177e-02],
                [-4.4639e-02,  7.1567e-02,  9.9644e-01]],

               [[ 9.9694e-01,  7.5739e-02,  1.9593e-02],
                [-7.8230e-02,  9.6684e-01,  2.4312e-01],
                [-5.2949e-04, -2.4391e-01,  9.6980e-01]],

               [[ 9.9829e-01,  3.7266e-03, -5.8383e-02],
                [-9.2723e-03,  9.9543e-01, -9.5007e-02],
                [ 5.7762e-02,  9.5385e-02,  9.9376e-01]],

               [[ 9.7825e-01, -1.4253e-02, -2.0694e-01],
                [-1.5912e-01,  5.8845e-01, -7.9272e-01],
                [ 1.3307e-01,  8.0841e-01,  5.7338e-01]],

               [[ 9.9423e-01, -9.7982e-02,  4.3654e-02],
                [ 1.0672e-01,  8.6265e-01, -4.9442e-01],
                [ 1.0786e-02,  4.9623e-01,  8.6813e-01]],

               [[ 9.9989e-01,  4.1306e-04,  1.4797e-02],
                [-1.5796e-04,  9.9985e-01, -1.7237e-02],
                [-1.4802e-02,  1.7233e-02,  9.9974e-01]],

               [[ 9.8707e-01,  7.1213e-02,  1.4363e-01],
                [-9.7939e-02,  9.7716e-01,  1.8858e-01],
                [-1.2692e-01, -2.0021e-01,  9.7150e-01]],

               [[ 9.6918e-01, -1.3599e-01, -2.0542e-01],
                [ 1.5803e-01,  9.8286e-01,  9.4932e-02],
                [ 1.8899e-01, -1.2447e-01,  9.7406e-01]],

               [[ 9.9968e-01, -4.1455e-03, -2.4925e-02],
                [ 2.6640e-03,  9.9824e-01, -5.9180e-02],
                [ 2.5127e-02,  5.9095e-02,  9.9794e-01]],

               [[ 9.8271e-01, -1.7395e-01,  6.3482e-02],
                [ 1.4719e-01,  9.4182e-01,  3.0219e-01],
                [-1.1235e-01, -2.8762e-01,  9.5113e-01]],

               [[ 9.3842e-01,  3.4208e-01,  4.8466e-02],
                [-3.4489e-01,  9.1922e-01,  1.8998e-01],
                [ 2.0437e-02, -1.9500e-01,  9.8059e-01]],

               [[ 9.9998e-01, -4.1370e-03,  5.3857e-03],
                [ 3.5619e-03,  9.9470e-01,  1.0273e-01],
                [-5.7821e-03, -1.0270e-01,  9.9470e-01]],

               [[ 9.1520e-01,  3.9569e-01,  7.6346e-02],
                [-3.9180e-01,  9.1801e-01, -6.1253e-02],
                [-9.4324e-02,  2.6147e-02,  9.9520e-01]],

               [[ 8.9073e-01, -4.3749e-01, -1.2329e-01],
                [ 4.2988e-01,  8.9896e-01, -8.4109e-02],
                [ 1.4763e-01,  2.1920e-02,  9.8880e-01]],

               [[ 9.9968e-01, -1.6306e-02, -1.9334e-02],
                [ 1.2042e-02,  9.7909e-01, -2.0309e-01],
                [ 2.2241e-02,  2.0279e-01,  9.7897e-01]],

               [[ 6.1531e-01,  7.4265e-01, -2.6434e-01],
                [-7.8492e-01,  6.0818e-01, -1.1842e-01],
                [ 7.2822e-02,  2.8035e-01,  9.5713e-01]],

               [[ 5.6729e-01, -7.8237e-01,  2.5706e-01],
                [ 8.2253e-01,  5.2300e-01, -2.2342e-01],
                [ 4.0358e-02,  3.3818e-01,  9.4022e-01]],

               [[ 1.9287e-01, -3.5493e-01, -9.1478e-01],
                [ 5.7317e-02,  9.3477e-01, -3.5061e-01],
                [ 9.7955e-01,  1.5190e-02,  2.0063e-01]],

               [[ 2.2024e-01,  3.9169e-01,  8.9335e-01],
                [-1.2658e-02,  9.1691e-01, -3.9889e-01],
                [-9.7536e-01,  7.6546e-02,  2.0690e-01]],

               [[ 9.9633e-01, -4.3941e-02, -7.3411e-02],
                [ 4.4991e-02,  9.9891e-01,  1.2713e-02],
                [ 7.2773e-02, -1.5969e-02,  9.9722e-01]],

               [[ 9.9609e-01,  1.2921e-02,  8.7367e-02],
                [-1.3884e-02,  9.9985e-01,  1.0430e-02],
                [-8.7219e-02, -1.1602e-02,  9.9612e-01]],

               [[ 9.7347e-01,  2.1969e-01, -6.3964e-02],
                [-2.0781e-01,  9.6587e-01,  1.5465e-01],
                [ 9.5755e-02, -1.3725e-01,  9.8590e-01]],

               [[ 9.7142e-01, -2.2502e-01,  7.5604e-02],
                [ 2.1155e-01,  9.6512e-01,  1.5426e-01],
                [-1.0768e-01, -1.3385e-01,  9.8513e-01]]]]

    beta = [[ 1.1966,  0.0516,  0.1071, -0.6911, -0.1227, -0.1355, -0.1933,  0.0890,
              -0.0418,  0.1257]]
    print(torch.Tensor(beta))
    semantic_prior = np.load('../data/semantic_prior.npy', mmap_mode='r')
    # print(semantic_prior)
    # print(semantic_prior.shape)
    # print(semantic_prior[0])
    best_match =  np.load('../data/prior_best_match.npz', mmap_mode='r')
    v = smpl(torch.Tensor(rotmat), torch.Tensor(beta))
    print(v)
    print(v.shape)

    # print(best_match.shape)
    # print(best_match)
