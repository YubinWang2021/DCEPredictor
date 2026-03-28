from serialization import load_model
from tqdm import tqdm
import h5py
import numpy as np
import trimesh
import pyrender
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import torch
import os
import torch.nn as nn
from sklearn.decomposition import PCA
#os.environ['PYOPENGL_PLATFORM'] = 'egl'



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
    def __init__(self, model_file='/data/dce3d/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'):
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

        J_regressor_extra = torch.from_numpy(np.load('/data/dce3d/J_regressor_extra.npy')).float()
        #semantic_prior = np.load('/data/dce3d/semantic_prior.npy', mmap_mode='r')
        # self.prior_match = np.argmax(semantic_prior, axis=1)
        #self.prior_best_match = np.load('/data/dce3d/prior_match.npy', mmap_mode='r')
        self.register_buffer('J_regressor_extra', J_regressor_extra)
        self.joints_idx = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]

    def show_skeleton(self, vals):
        connect = [
            (0,1),(0,2),(0,3),(1,4), (2,5),
            (3,6),(4,7),(5,8),(6,9),(7,10),
            (8,11), (9,12), (9,13), (9,14), (12,15), (13, 16),
            (14,17), (16,18), (17,19), (18,20),(19,21),
            (20,22), (21,23)
        ]
        LR =[
            False, True, False, False, True,
            False, False, True, False, False,
            True, False, False, True, False,
            False, True, False, True, False,
            True,False,True,False]

        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
        I   = np.array([touple[0] for touple in connect])
        J   = np.array([touple[1] for touple in connect])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(0, -90)
        ax.axis('off')
        for i in np.arange( len(I) ):
            x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
            z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
            y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
            ax.plot(x, y, z, lw=2,linestyle='-' ,c=lcolor if LR[i] else rcolor)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #plt.show()
        #exit(0)
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
        #self.render(v.view(6890, 3), self.faces)
        v = v.reshape(6890, 3)
        joint_x = torch.matmul(self.J_regressor, v[:, 0])
        joint_y = torch.matmul(self.J_regressor, v[:, 1])
        joint_z = torch.matmul(self.J_regressor, v[:, 2])
        print(joint_x.shape)
        joints = torch.stack([joint_x, joint_y, joint_z])
        print(joints.shape)
        self.show_skeleton(joints.T.numpy())

        print(v.shape)
        return v


m = load_model('/data/dce3d/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
mean_params = h5py.File('/data/dce3d/neutral_smpl_mean_params.h5')
smpl = SMPL()
pose1 = [[ 3.13400769e+00, -5.28170764e-02, -1.68974012e-01, -7.34391212e-02,
           -5.65518327e-02, -1.39350137e-02, -2.29389849e-03,  4.98376600e-02,
           6.88722208e-02,  5.21789193e-02,  1.31186210e-02, -1.01115378e-02,
           2.24503919e-01,-5.02349921e-02, -3.80691607e-03,  2.10271835e-01,
           4.01574709e-02, -5.10532735e-03, -1.81149654e-02, -6.38265163e-03,
           -2.08786912e-02,  5.97914271e-02,  1.07164562e-01, -1.01480663e-01,
           3.49301212e-02,-1.07443728e-01, -1.21068336e-01,  1.55887362e-02,
           1.26785412e-01, -9.30622146e-02,  5.17247571e-03, -1.34127185e-01,
           -1.48413330e-01,  6.19624415e-03, -2.08951053e-04,  7.47227222e-02,
           -4.09233086e-02, -3.70299011e-01,  7.81559944e-02,  4.62782606e-02,
           3.40863705e-01,  1.25406934e-02,  5.37197990e-03,  3.08812764e-02,
           2.18552366e-01, -1.93223089e-01, -1.08494198e+00,  2.02882141e-01,
           1.93090603e-01,  1.11366820e+00,  9.89959389e-02, -3.42678964e-01,
           -1.00072517e-04,  9.82326120e-02,  3.31835657e-01,  4.93594259e-03,
           4.52058688e-02, -5.80894947e-02,  3.11855283e-02, 1.78251360e-02,
           5.66842332e-02, -1.90531928e-02, -1.63560525e-01, -7.82037750e-02,
           -1.74610585e-01, -1.46583453e-01,  7.81062394e-02,  1.74330190e-01]]
beta1 = [[-0.20579515,  0.8578984,   0.6952866,   0.77491874, -0.07468977,  0.04534215,
          -0.4854895,   0.09801093,  0.5780989,  -0.06435902]]

pose2 = [[ 3.13384986e+00, -3.62787023e-02, -9.97439772e-02, -6.63519800e-02,
           -4.34159487e-02, -1.10430587e-02,  1.85286254e-02,  5.54462038e-02,
           5.67415021e-02, 5.99865243e-02,  7.14684371e-03, -3.46427504e-03,
           2.10307166e-01, -3.68881412e-02, -2.64929445e-03,  1.88005939e-01,
           6.06654473e-02, -1.38916518e-03, -1.17857261e-02, -1.00030480e-02,
           -1.53904734e-02,  5.76937273e-02,  1.00571938e-01, -1.04983144e-01,
           2.95077693e-02, -9.73880738e-02,  8.86274949e-02,  1.43257389e-02,
           5.04233222e-03, -2.10346945e-04, -1.13272525e-01,  1.05721112e-02,
           1.27242073e-01, -8.83958042e-02,  5.41350292e-03, -1.29875615e-01,
           -1.41689256e-01, -1.03142916e-03, -7.69265578e-04,  7.47656897e-02,
           -5.08539304e-02, -3.65715742e-01,  7.61656314e-02,  5.59825860e-02,
           3.35477442e-01,  2.56500822e-02,  5.02478378e-03,  2.72483062e-02,
           1.98162004e-01, -1.99134797e-01, -1.09508693e+00,  1.81959882e-01,
           2.02946350e-01,  1.11684966e+00,  6.04090057e-02, -2.90210903e-01,
           -2.31108442e-03,  7.12517202e-02,  2.79502690e-01,  1.20088803e-02,
           3.46019343e-02, -5.05187884e-02,  2.61938665e-02,  6.03051204e-03,
           4.95265573e-02, -1.25128366e-02, -1.58430830e-01, -7.57456571e-02,
           -1.68992266e-01, -1.42264307e-01,  7.37869889e-02,  1.65977269e-01]]

beta2 = [[-0.13225749,  0.79805356,  0.66986734,  0.7196756,   0.01085477, -0.00726128,
          -0.5154457,   0.1319955,   0.5059309,  -0.13220522]]


pose3 = [[ 3.12633777e+00, -2.91826725e-02,  6.29244745e-02, -2.44424939e-02,
           -3.32548469e-02, -1.79166475e-03, -4.93947789e-02,  3.08964308e-02,
           3.12612578e-02,  5.06187230e-02,  1.82378793e-03, -3.78239714e-03,
           1.92183077e-01, -6.11335747e-02,  2.07091845e-03,  2.31667444e-01,
           2.98443511e-02, -1.68635976e-03, -1.39272576e-02,  1.52002368e-03,
           1.98224280e-03,  5.76786734e-02,  1.13163196e-01, -1.10995561e-01,
           4.39319313e-02, -1.03373699e-01,  1.01222999e-01,  1.49005186e-02,
           5.41154109e-03,  2.97820498e-03, -1.16040781e-01,  8.56067333e-03,
           1.38249680e-01, -9.16752368e-02, -2.95686116e-03, -1.36594057e-01,
           -1.15945980e-01, -6.27051219e-02,  2.03363337e-02,  7.42639974e-02,
           -3.88234891e-02, -3.47355902e-01,  7.19380602e-02,  5.84479049e-02,
           3.41185987e-01,  4.43198420e-02, -6.57944083e-02,  4.09439579e-02,
           2.02202544e-01, -2.14733392e-01, -1.08377552e+00,  1.83507487e-01,
           1.81665793e-01,  1.09270012e+00,  8.87923241e-02, -3.45965236e-01,
           -4.47294238e-04,  4.91326563e-02,  2.92789429e-01,  2.34983768e-03,
           3.53270695e-02, -5.64139336e-02,  3.79547402e-02, -5.04378870e-04,
           4.82321978e-02, -2.45663375e-02, -1.56561777e-01, -7.45511800e-02,
           -1.71840042e-01, -1.42150238e-01,  7.45768026e-02,  1.63606316e-01]]

beta3= [[-0.12020759,  0.9121984,   0.651128,    0.56286985, -0.0828072,  -0.01807698,
         -0.5691441,   0.13559215,  0.5042928, -0.050696  ]]


import pickle
f = open('/data/dce3d/0350-C07-W01-T000-F006_pose.pkl', 'rb')
pose_1 = pickle.load(f)
f = open('/data/dce3d/0350-C07-W01-T000-F009_pose.pkl', 'rb')
pose_2 = pickle.load(f)
f = open('/data/dce3d/0350-C07-W01-T000-F015_pose.pkl', 'rb')
pose_3 = pickle.load(f)
f = open('/data/dce3d/0350-C07-W01-T000-F018_pose.pkl', 'rb')
pose_4 = pickle.load(f)
f = open('/data/dce3d/0350-C07-W01-T000-F021_pose.pkl', 'rb')
pose_5 = pickle.load(f)
f = open('/data/dce3d/0350-C07-W01-T000-F006_shape.pkl', 'rb')
beta_1 = pickle.load(f)
f = open('/data/dce3d/0350-C07-W01-T000-F009_shape.pkl', 'rb')
beta_2 = pickle.load(f)
f = open('/data/dce3d/0350-C07-W01-T000-F015_shape.pkl', 'rb')
beta_3 = pickle.load(f)
f = open('/data/dce3d/0350-C07-W01-T000-F018_shape.pkl', 'rb')
beta_4 = pickle.load(f)
f = open('/data/dce3d/0350-C07-W01-T000-F021_shape.pkl', 'rb')
beta_5 = pickle.load(f)


# pose_1 = [[2.17, 0, 0, -0.22, 0.02, 0.09, -0.24, -0.05, -0.08, 0.28, 0.01, 0.01, 0.43, -0.06, -0.1, 0.5, 0.0034512917694683523, 0.06, 0.02, -0.03, 0.01, 0.01, 0.13, -0.05, -0.06, -0.18, 0.14, 0.01, -0.003655812969241232, 0.01, -0.21, 0.16, 0.11, -0.04, 0.08, -0.2, -0.01, -0.01, -0.003478249897550943, -0.02, -0.12, -0.27, -0.05, 0.2, 0.24, 0.07, 0.83, 0.03, 0.5, -0.5, -0.83, 0.17, 0.17, 1.0, 0.67, -0.33, 0.0, -1.33, 0.5, -0.33, -0.07, -0.1, 0.05, -0.04, 0.09, -0.07, -0.14, -0.06, -0.18, -0.09, 0.11, 0.2]]
# pose_2 = [[2.17, 0, 0, -0.22, 0.02, 0.09, -0.24, -0.05, -0.08, 0.28, 0.01, 0.01, 0.43, -0.06, -0.1, 0.5, 0.0034512917694683523, 0.06, 0.02, -0.03, 0.01, 0.01, 0.13, -0.05, -0.06, -0.18, 0.14, 0.01, -0.003655812969241232, 0.01, -0.21, 0.16, 0.11, -0.04, 0.08, -0.2, -0.01, -0.01, -0.003478249897550943, -0.02, -0.12, -0.27, -0.05, 0.2, 0.24, 0.07, 0.83, 0.03, 0.5, -0.5, -0.83, 0.33, 0.17, 1.0, 0.67, -0.33, 0.0, -0.83, 0.5, 0.0, -0.07, -0.1, 0.05, -0.04, 0.09, -0.07, -0.14, -0.06, -0.18, -0.09, 0.11, 0.2]]
# pose_3 = [[2.17, 0, 0, -0.22, 0.02, 0.09, -0.24, -0.05, -0.08, 0.28, 0.01, 0.01, 0.43, -0.06, -0.1, 0.5, 0.0034512917694683523, 0.06, 0.02, -0.03, 0.01, 0.01, 0.13, -0.05, -0.06, -0.18, 0.14, 0.01, -0.003655812969241232, 0.01, -0.21, 0.16, 0.11, -0.04, 0.08, -0.2, -0.01, -0.01, -0.003478249897550943, -0.02, -0.12, -0.27, -0.05, 0.2, 0.24, 0.07, 0.83, 0.03, 0.5, -0.5, -0.83, 0.5, 0.33, 1.0, 0.67, -0.33, 0.0, -0.5, 0.33, -0.33, -0.07, -0.1, 0.05, 0.0, 0.09, -0.07, -0.14, -0.06, -0.18, -0.09, 0.11, 0.2]]
# beta_1 = [[0.21, 0.34, -0.35, 0.36, 0.42, 0.03, 0.3, 0.24, 0.21, 0.31]]
# beta_2 = [[0.21, 0.34, -0.35, 0.36, 0.42, 0.03, 0.3, 0.24, 0.21, 0.31]]
# beta_3 = [[0.21, 0.34, -0.35, 0.36, 0.42, 0.03, 0.3, 0.24, 0.21, 0.31]]


# pose4 = [[2.17, 0, 0, -0.22, 0.02, 0.09, -0.24, -0.05, -0.08, 0.28, 0.01, 0.01, 0.43, -0.06, -0.1, 0.5, 0.0034512917694683523, 0.06, 0.02, -0.03, 0.01, 0.01, 0.13, -0.05, -0.06, -0.18, 0.14, 0.01, -0.003655812969241232, 0.01, -0.21, 0.16, 0.11, -0.04, 0.08, -0.2, -0.01, -0.01, -0.003478249897550943, -0.02, -0.12, -0.27, -0.05, 0.2, 0.24, 0.07, 0.83, 0.03, 0.5, -0.5, -0.83, 0.5, 0.33, 1.0, 0.67, -0.33, 0.0, -0.5, 0.33, -0.33, -0.07, -0.1, 0.05, 0.0, 0.09, -0.07, -0.14, -0.06, -0.18, -0.09, 0.11, 0.2]]
# beta4 = [[0.21, 0.34, -0.35, 0.36, 0.42, 0.03, 0.3, 0.24, 0.21, 0.31]]
# pose = 
v = smpl(torch.Tensor(pose_3), torch.Tensor(beta_3))
#v = v.numpy()


vertex_colors = np.load('vert_colors_3D.npy')
vertex_colors = vertex_colors[0:6890].round().astype(np.uint8)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(90, -90)
max_range = 0.55
ax.set_xlim(- max_range, max_range)
ax.set_ylim(- max_range, max_range)
ax.set_zlim(-0.2 - max_range, -0.2 + max_range)
ax.axis('off')
'''
torso_indices = np.load('./torso_indices.npy')
torso_back_indices = np.load('torso_back_indices.npy')
vertex_img = np.load('./pred_vertices_img.npy')

img_ori = Image.open("./img_back.png")
img_ori = np.array(img_ori)
for idx in torso_back_indices:
    h, w = int(vertex_img[idx][0].round()), int(vertex_img[idx][1].round())
    print(h, w)
    color = img_ori[w, h]
    if color[0] != 255 and color[1] != 255 and color[2] != 255:
        vertex_colors[idx] = color
np.save('vertex_color_with_front_back.npy', vertex_colors)
'''
all_test_colors = []
for i in range(6890):
    all_test_colors.append([0.0,0.0,0.0])
ax.scatter(m[:,0], m[:,1], m[:,2], c=all_test_colors, marker='o', s=-1)


#exit(0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#plt.show()


msh = trimesh.Trimesh(vertices=v, faces=m.f, vertex_colors=vertex_colors.squeeze())
scene_center = np.mean(msh.vertices, axis=0)

mesh_node = pyrender.Mesh.from_trimesh(msh)
#mesh_node.primitives[0].material.baseColorFactor = [0.8, 0.2, 0.2, 1.0]  
mesh_node.primitives[0].material.roughnessFactor = 0.5  
scene = pyrender.Scene(ambient_light=[0.20,0.20,0.20])
scene.add(mesh_node)


point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=8.0)
point_light_pose = np.eye(4)  
point_light_pose[:3, 3] = [2.0, 0.0, 5.0]  
scene.add(point_light, pose=point_light_pose)


'''
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=4 / 3.0)
camera_pose = np.eye(4)  # Identity matrix for the camera pose
scene.add(camera, pose=camera_pose)
'''
camera_pose = np.eye(4)
camera_pose[:3, 3] = scene_center + np.array([0.0, 0.0, 2.0])  
scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=4 / 3.0), pose=camera_pose)

#light = pyrender.SpotLight(color=np.ones(3), intensity=100.0,
#                           innerConeAngle=np.pi/16.0,
#                           outerConeAngle=np.pi/6.0)
#scene.add(light, pose=camera_pose)

#renderer = pyrender.OffscreenRenderer(viewport_width=1200, viewport_height=900)
#color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
#color_image = Image.fromarray(color)
#color_image.save("color_image.png")


viewer = pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(800, 600))


