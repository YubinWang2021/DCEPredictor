import numpy as np

from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight
from opendr.renderer import ColoredRenderer
from matplotlib.pyplot import MultipleLocator
from serialization import load_model
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import torch
import h5py
import cv2
import matplotlib.colors as clr


# def color_map(part_index):
#     color_dict = {
#         2: (255 / 255, 0 / 255, 0 / 255),
#         5: (255 / 255, 85 / 255, 0 / 255),
#         9: (0 / 255, 85 / 255, 85 / 255),
#         10: (85 / 255, 51 / 255, 0 / 255),
#         13: (0 / 255, 0 / 255, 255 / 255),
#         14: (51 / 255, 170 / 255, 221 / 255),
#         15: (0 / 255, 255 / 255, 255 / 255),
#         18: (255 / 255, 255 / 255, 0 / 255),
#         19: (255 / 255, 170 / 255, 0 / 255)
#     }
#     if part_index in color_dict:
#         return color_dict[part_index]

#     else:
#         return (245 / 255, 245 / 255, 245 / 255)


# def find_index(prior, part):
#     idx_list = list()
#     for i in range(len(prior)):
#         if prior[i] == part:
#             idx_list.append(i)
#     return idx_list


# # Load SMPL model
m = load_model('/media/data3/wangyubin/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

# # mean_params = np.load('../data/smpl_mean_params.npz')
#mean_params = h5py.File('/media/dat3/wangyubin/neutral_smpl_mean_params.h5')

# semantic_prior = np.load('../data/semantic_prior.npy', mmap_mode='r')
# f_head = np.load('../data/head_indices.npy', mmap_mode='r')
# f = open('../data/head.obj', 'r')
# lines = f.readlines()

# #aroundy = cv2.Rodrigues(np.array([0, np.radians(180), 0]))[0]



# # print(len(semantic_prior))
# prior_best_match = np.argmax(semantic_prior, axis=1)
# hair_set = set()
# for i in range(len(prior_best_match)):
#     if prior_best_match[i] == 2 and (
#             np.array(m[m.f[i][0]])[1] > 0.4 and np.array(m[m.f[i][1]])[1] > 0.4 and np.array(m[m.f[i][2]])[1] > 0.4) \
#             and (
#             np.array(m[m.f[i][0]])[2] < 0.49 and np.array(m[m.f[i][1]])[2] < 0.49 and np.array(m[m.f[i][2]])[2] < 0.49):

#         hair_set.add(m.f[i][0])
#         hair_set.add(m.f[i][1])
#         hair_set.add(m.f[i][2])

#     elif prior_best_match[i] == 2:
#         prior_best_match[i] = 10
# np.save('../data/prior_best_match.npy', prior_best_match)
# exit(0)
class Visualizer(object):
    def __init__(self):
        ## Create OpenDR renderer
        super(Visualizer, self).__init__()
        self.rn = ColoredRenderer()
        self.w, self.h = (640, 480)


## Assign attributes to renderer
rn = ColoredRenderer()
w, h = (640, 480)
rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w, w]) / 2., c=np.array([w, h]) / 2.,
                          k=np.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m, f=m.f, bgcolor=np.zeros(3))
## Construct point light source


rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,
    num_verts=len(m),
    light_pos=np.array([-1000, -1000, -2000]),
    vc=np.ones_like(m) * .9,
    light_color=np.array([1., 1., 1.]))

## Show it using OpenCV
import cv2
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 20))

ax = plt.axes(projection="3d")
ax.view_init(90, -90)
#ax.view_init(90, -90)
#ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 0.2, 1, 1]))
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.75, 1, 1, 1]))
# ax.pbaspect = [0.15, 1.0, 1]


x_3d = m[:, 0]
y_3d = m[:, 1]
z_3d = m[:, 2]
'''
m_v = np.load('../data/v_0101.npy')
center = m_v.mean(axis=0)

#m_v = np.dot((m_v - center), aroundy) + center
x_3d = m_v[:, 0]
y_3d = m_v[:, 1]
z_3d = m_v[:, 2]
'''
# draw vertices
#ax.scatter(x_3d, y_3d, z_3d, zdir='z', c='#2E8B57', s=1)
# ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.75, 1, 1, 1]))

# print(np.array(hair_set))
# ax.scatter(np.array(x_3d)[np.array(list(hair_set))], np.array(y_3d)[np.array(list(hair_set))],
#         np.array(z_3d)[np.array(list(hair_set))], zdir='z', c='#0000FF', s=4)
# print(list(np.array(y_3d)[np.array(list(hair_set))]))

# draw faces
ax.set(xlabel="X", ylabel="Y", zlabel="Z")
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')

for i in tqdm(range(len(m.f))):
    vertex_lst = m.f[i]
    x_list = list()
    y_list = list()
    z_list = list()

    #v = np.load('../data/v_0101.npy')
    x_list.append(float(m[vertex_lst[0]][0]))
    y_list.append(float(m[vertex_lst[0]][1]))
    z_list.append(float(m[vertex_lst[0]][2]))
    x_list.append(float(m[vertex_lst[1]][0]))
    y_list.append(float(m[vertex_lst[1]][1]))
    z_list.append(float(m[vertex_lst[1]][2]))
    x_list.append(float(m[vertex_lst[2]][0]))
    y_list.append(float(m[vertex_lst[2]][1]))
    z_list.append(float(m[vertex_lst[2]][2]))
    '''
    # for side or top-down view
    np.dot((m - center), aroundy) + center
    v_0 = np.dot((m[vertex_lst[0]] - center), aroundy) + center
    v_1 = np.dot((m[vertex_lst[1]] - center), aroundy) + center
    v_2 = np.dot((m[vertex_lst[2]] - center), aroundy) + center

    x_list.append(float(v_0[0]))
    y_list.append(float(v_0[1]))
    z_list.append(float(v_0[2]))
    x_list.append(float(v_1[0]))
    y_list.append(float(v_1[1]))
    z_list.append(float(v_1[2]))
    x_list.append(float(v_2[0]))
    y_list.append(float(v_2[1]))
    z_list.append(float(v_2[2]))
    
    # do not swin
    
    x_list.append(float(m[vertex_lst[0]][0]))
    y_list.append(float(m[vertex_lst[0]][1]))
    z_list.append(float(m[vertex_lst[0]][2]))
    x_list.append(float(m[vertex_lst[1]][0]))
    y_list.append(float(m[vertex_lst[1]][1]))
    z_list.append(float(m[vertex_lst[1]][2]))
    x_list.append(float(m[vertex_lst[2]][0]))
    y_list.append(float(m[vertex_lst[2]][1]))
    z_list.append(float(m[vertex_lst[2]][2]))
    '''
    # draw triangle

    #ax.plot_trisurf(x_list, y_list, z_list, color=(245 / 255, 245 / 255, 245 / 255), edgecolor='black', linewidth=0.15)

    # draw part triangle
    ax.plot_trisurf(x_list, y_list, z_list, color=color_map(prior_best_match[i]), edgecolor='black', linewidth=0.15)
    # draw lines
    # x_list.append(float(m[vertex_lst[0]][0])
    # y_list.append(float(m[vertex_lst[0]][1]))
    # z_list.append(float(m[vertex_lst[0]][2]))
    # ax.plot(x_list, y_list, z_list , ls='-',linewidth=0.3, color='#A0522D')

# plt.show()

f = plt.gcf()  # 获取当前图像
f.savefig('/media/data3/wangyubin/DCEReID/utils/Paper_SMPL_T3.png')
f.clear()

#plt.imsave('/media/data3/wangyubin/DCEReID/utils/Paper_SMPL_T1.png', fig)
# cv2.imshow('render_SMPL', rn.r)
# print('..Print any key while on the display window')
# cv2.waitKey(0)
# cv2.destroyAllWindows()


## Could also use matplotlib to display
# import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
# import pdb; pdb.set_trace()
