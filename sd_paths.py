from src_core import plugins
from src_core.classes import paths

# default_ckpt = 'sd-v1-4.ckpt'
# default_ckpt = 'model.ckpt'

default_ckpt = paths.plug_res / 'sd1111' / 'sd-v1-5.ckpt'
vae_path = paths.plug_res / 'sd1111' / 'vae.vae.pt'
config = paths.plug_repos / 'sd1111' / 'stable_diffusion' / 'configs/stable-diffusion/v1-inference.yaml'


def res(join=""):
    return plugins.get_plug('sd1111').res(join)
