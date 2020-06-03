import pickle
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
import runway

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
n_styles = 16

@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
    global Gs
    tflib.init_tf()
    if opts['checkpoint'] is None:
        opts['checkpoint'] = 'checkpoints\\network-snapshot-000600.pkl'
    with open(opts['checkpoint'], 'rb') as file:
        _G, _D, Gs = pickle.load(file, encoding='latin1')
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    rnd = np.random.RandomState()
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
    return Gs


generate_inputs = {
    'z': runway.vector(256, sampling_std=0.5),
    'truncation': runway.number(min=0, max=1, default=0.8, step=0.01)
}

for i in range(4):
    for j in range(4):
        generate_inputs.update({'style_%i/%i' %(i, j): runway.number(min=0, max=1, default=0.5, step=0.01)})

@runway.command('generate', inputs=generate_inputs, outputs={'image': runway.image})
def convert(model, inputs):
    z = inputs['z']
    truncation = inputs['truncation']
    latent = z.reshape((1, 256))
    style_mix = np.zeros([4, 4])
    for i in range(4):
        for j in range(4):
            style_mix[i, j] = inputs['style_%i/%i' %(i, j)]

    style_mix = style_mix.reshape(-1)
    dlatent = model.components.mapping.run(latent, None)
    mixed_dlatent = dlatent.copy()
    for i in range(n_styles):
        mixed_dlatent[0][i] *= style_mix[i]

    images = model.components.synthesis.run(mixed_dlatent, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
    #images = model.run(latents, None, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
    output = np.clip(images[0], 0, 255).astype(np.uint8)
    return {'image': output}


if __name__ == '__main__':
    runway.run()
