import os
import sys
import pickle
import numpy as np
import PIL.Image
from tqdm import tqdm
import dnnlib
import dnnlib.tflib as tflib
import matplotlib.pyplot as plt
plt.ion()

import config
from time import time
from training.misc import adjust_dynamic_range

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()
_G_cache = dict()

def load_Gs(path, and_G=False):
    if path not in _Gs_cache:
        with open(path, 'rb') as f:
           _G, _D, Gs = pickle.load(f)
        _Gs_cache[path] = Gs
        if and_G:
            _G_cache[path] = _G
    return _Gs_cache[path], _G_cache[path]

#path = '/media/romain/data/stylegan/results/00007-sgan-feuilles256px_100k-1gpu/network-snapshot-007726.pkl'
#path = '/media/romain/data/stylegan/results/00012-sgan-feuilles256px_100k-1gpu/network-snapshot-007326.pkl'
#path = '/media/romain/data/stylegan/results/00015-sgan-feuilles256px_100k_labeled-1gpu/'
#path = '/media/romain/data/stylegan/results/00016-sgan-feuilles256px_100k_labeled-1gpu/'
path = '/media/romain/windows/Users/Romain/Desktop/stylegan/results/00067-sgan-feuilles1024_labels-1gpu/'
# ckpt = 'network-snapshot-006526.pkl'
# ckpt = 'network-snapshot-006521.pkl'
ckpt = 'network-snapshot-006521.pkl'
figpath = '/media/romain/windows/Users/Romain/Desktop/grid_exploration/00067-sgan-feuilles1024_labels-1gpu/'

tflib.init_tf()

# Gs = load_Gs(path)
Gs, G = load_Gs(path + ckpt, and_G=True)


plt.rcParams['toolbar']='none'
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['figure.facecolor']='black'

xpos, ypos = 24, 6#14, 5 #21, 24 #10, 12
zpos = 8 #12

for xpos, ypos, zpos in [[1, 2, 3], [24, 6, 8], [14, 5, 12]]:

    rot = np.pi/2
    #gstart, gstop = 0.5, 1e-2
    grid_size = 0.5
    fps, d = 30, 1000 * 1
    w, h = 512, 1024
    xgrid, ygrid = 16, 4

    seed = 51 #46, 51, 12, 121]
    # for seed in [12, 46, 51]:
    figdir = 'x%iy%iz%i_seed%i_dur%i_grid%ix%i_size%.3f' %(xpos, ypos, zpos, seed, d, xgrid, ygrid, grid_size)
    if not os.path.exists(os.path.join(figpath, figdir)):
        os.makedirs(os.path.join(figpath, figdir))

    rand = np.random.RandomState(seed)
    n_dim = Gs.input_shape[1]
    dlatent_avg = Gs.get_var('dlatent_avg')

    latent_start = np.array([rand.randn(n_dim)]) / 10
    # latent_stop = -latent_start
    # latent_traj = np.linspace(latent_start[0], dlatent_avg, d//2)
    # latent_traj = np.vstack((latent_traj, np.linspace(dlatent_avg, latent_stop[0], d//2)))

    # truncation trick
    psis = np.linspace(-1, 1, d+1)
    latent_traj = (latent_start - dlatent_avg) * np.reshape(psis, [-1, 1, 1]) + dlatent_avg
    rot_traj = np.linspace(-rot, rot, d+1)

    labels = np.zeros([1, 16])
    # labels[:, 8] = 1
    # labels[:, 12] = 1

    dgrid = np.linspace(-grid_size/2, -grid_size/2, d//2)
    dgrid = np.hstack((dgrid, dgrid[::-1]))
    img_grid = np.zeros([ygrid * h, xgrid * w, 3], dtype=np.uint8)
    postfix = {'norm': '0', 'dlatent': '0'}

    fig, ax = plt.subplots()
    stop = False
    def stop_play(event):
        global stop
        if event.key == 'q':
            stop = True

    fig.canvas.mpl_connect('key_press_event', stop_play)
    im = ax.imshow(img_grid)
    plt.tight_layout()
    for t in tqdm(range(d), total=d, unit='img'):
        tstart = time()
        latent = latent_traj[t]
        if t < d:
            dlat = np.diff([latent_traj[t], latent_traj[t+1]], axis=0)[0][0]
            dl = np.linalg.norm(dlat)

        x = np.linspace(-dgrid[t], dgrid[t], xgrid) / 4
        y = np.linspace(-dgrid[t], dgrid[t], ygrid)

        xx, yy = np.meshgrid(x, y)
        # if rot > 0:
        #     coord = np.vstack((xx.reshape(1, -1), yy.reshape(1, -1))).T
        #     r = np.array([[np.cos(rot_traj[t]), np.sin(rot_traj[t])],
        #                   [-np.sin(rot_traj[t]), np.cos(rot_traj[t])]])
        #     new_xx, new_yy = np.dot(coord, r).T
        #     xx = new_xx.reshape(xx.shape)
        #     yy = new_yy.reshape(yy.shape)

        normal = dlat[[xpos, ypos, zpos]]
        c = -latent[0][[xpos, ypos, zpos]].dot(normal)
        zz = (normal[0] * xx - normal[1] * yy - c) * 1. / normal[2]

        blatent = latent[0].repeat(xgrid * ygrid).reshape(n_dim, ygrid, xgrid)
        blatent[xpos] += xx
        blatent[ypos] += yy
        blatent[zpos] += zz

        images = Gs.run(blatent.reshape(n_dim, -1).T, None, is_validation=True, minibatch_size=1)
        images = images.reshape(ygrid, xgrid, 3, h, w)

        for i in range(xgrid):
            for j in range(ygrid):
                image = images[j, i].transpose(1, 2, 0)
                image = adjust_dynamic_range(image, [-1, 1], [0, 255])
                image = np.rint(image).clip(0, 255).astype(np.uint8)
                img_grid[h*j:h*(j+1), w*i:w*(i+1)] = image

        fps = 1 / (time() - tstart)
        im.set_data(img_grid)
        n = np.linalg.norm(latent-dlatent_avg)
        mspeed = np.round(1/dl)
        postfix = 'index=%i/%i, fps=%.2f, norm=%.3f, slow=x%i' %(t, d, fps, n, mspeed)
        postfix += ' x=%i, y=%i, z=%i, seed=%i' %(xpos, ypos, zpos, seed)
        # postfix.update({'norm': n, 'dlatent': dl})
        ax.text(x=0, y=-50, s=postfix, fontsize=8,
                bbox=dict(facecolor='black'), fontdict=dict(color='green'))
        plt.pause(0.00001)
        pil_im = PIL.Image.fromarray(img_grid)
        pil_im.save(os.path.join(figpath, figdir, '%06d.jpg' %t), quality=95)

    plt.close()
