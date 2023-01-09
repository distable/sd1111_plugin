from src_core import plugins
from src_core.classes import paths

config = paths.plug_repos / 'sd1111' / 'stable_diffusion' / 'configs/stable-diffusion/v1-inference.yaml'
# config = paths.plug_repos / 'sd1111' / 'stablediffusion' / 'configs/stable-diffusion/v2-inference.yaml'
# config = paths.plug_repos / 'sd1111' / 'stablediffusion' / 'configs/stable-diffusion/v2-midas-inference.yaml'


def res(join=""):
    return plugins.get_plug('sd1111').res(join)
