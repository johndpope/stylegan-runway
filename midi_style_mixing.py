import os
import sys
import mido
import pickle

import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['toolbar']='none'
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['figure.facecolor']='black'
plt.ion()
plt.axis('off')

name = '00098-stylegan2-feuilles1k-1gpu-config-e'
Writer = animation.writers['ffmpeg']#avconv
writer = Writer(fps=15, metadata=dict(artist=name), bitrate=1800)

network_pkl = '/home/romain/win_desk/stylegan2/results/00098-stylegan2-feuilles1k-1gpu-config-e/network-snapshot-000600.pkl'
G, _D, Gs = pretrained_networks.load_networks(network_pkl)

truncation_psi = 1.0
w, h = 256, 256
seed = 45
rand = np.random.RandomState(seed)
# src_seeds=[639,701,687,615,2268]
# dst_seeds=[888,829,1898,1733]
# style_ranges=[range(0,4)]*3+[range(4,8)]*2+[range(8,14)]
loop, reset = True, False
midi_name = 'Midi Fighter Twister:Midi Fighter Twister MIDI 1 28:0'
assert midi_name in mido.get_input_names()

Gs_kwargs = dnnlib.EasyDict()
Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False
if truncation_psi is not None:
    Gs_kwargs.truncation_psi = truncation_psi

latent = np.array([rand.randn(Gs.input_shape[1])])
dlatent = Gs.components.mapping.run(latent, None) # [seed, layer, component]
images = Gs.components.synthesis.run(dlatent, **Gs_kwargs)

def press(event):
    global loop, reset
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'q':
        loop = False
        plt.close()
    elif event.key == 'r':
        reset = True
    elif event.key == 'e':
        seed = np.random.randint(100)
        print('setting seed to %i' %seed)
        rand = np.random.RandomState(seed)

fig, ax = plt.subplots()
fig.set_size_inches([6.61, 10.41])
fig.canvas.mpl_connect('key_press_event', press)
im = ax.imshow(images[0])

n_styles = dlatent.shape[1]
style_mix = np.ones(n_styles)

writer.setup(fig, name + '.mp4', 300)
with mido.open_input(midi_name) as inport:
    while loop:
        for m in inport.iter_pending():
            if m.control < n_styles:
                style_mix[m.control] = 1 - m.value / 128
                print(m.control, m.value)
        if reset:
            latent = np.array([rand.randn(Gs.input_shape[1])])
            dlatent = Gs.components.mapping.run(latent, None) # [seed, layer, component]
            print('reset the image')
            reset = False

        mixed_dlatent = dlatent.copy()
        for i in range(n_styles):
            mixed_dlatent[0][i] *= style_mix[i]

        images = Gs.components.synthesis.run(mixed_dlatent, **Gs_kwargs)
        im.set_data(images[0])
        writer.grab_frame()
        plt.pause(0.00001)

writer.finish()

# latent = np.array([rand.randn(Gs.input_shape[1])])
# dlatent = Gs.components.mapping.run(latent, None)
# images = Gs.components.synthesis.run(dlatent, randomize_noise=False, **synthesis_kwargs)
#
# from time import time
# from training.misc import adjust_dynamic_range
#
# n_dim = Gs.input_shape[1]
# latent_start = np.array([rand.randn(n_dim)])
# latent_stop  = -latent_start #np.array([rand.randn(n_dim)])
# fps, d = 30, 500
# latent_traj = np.linspace(latent_start[0], latent_stop[0], d)
#
# img_grid = np.zeros([4 * 256, 4 * 256, 3], dtype=np.uint8)
#
# fig, ax = plt.subplots()
# stop = False
# def stop_play(event):
#     global stop
#     if event.key == 'q':
#         stop = True
#
# fig.canvas.mpl_connect('key_press_event', stop_play)
# im = ax.imshow(img_grid)
# for t in range(d):
#     tstart = time()
#     latent = np.array([latent_traj[t]])
#     for i in range(4):
#         for j in range(4):
#             labels = np.zeros([1, 4, 4])
#             labels[0, i, j] = 1/(i+1)
#             labels = labels.reshape(1, 16)
#             images = Gs.run(latent, labels, is_validation=True, minibatch_size=8)
#
#             image = images[0].transpose(1, 2, 0)
#             image = adjust_dynamic_range(image, [-1, 1], [0,255])
#             image = np.rint(image).clip(0, 255).astype(np.uint8)
#             img_grid[256*i:256*(i+1), 256*j:256*(j+1)] = image
#
#     fps = 1 / (time() - tstart)
#     im.set_data(img_grid)
#     n = np.linalg.norm(latent)
#     ax.set_title('frame=%i/%i, fps=%.2f, norm=%.3f' %(t, d, fps, n))
#     plt.pause(0.00001)
