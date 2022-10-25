import argparse
import os

from PIL.Image import Image

import devices as devices
from src_core import plugins
from src_core.plugins import Plugin

# Options
from src_plugins.sd1111_plugin import safe, sd_paths
from src_plugins.sd1111_plugin.SDAttention import SDAttention

attention = SDAttention.SPLIT_DOGGETT
lowvram = False
medvram = True
lowram = False
precision = "full"
no_half = True
opt_channelslast = False
always_batch_cond_uncond = False  # disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram
xformers = False
force_enable_xformers = False
use_cpu = False
batch_cond_uncond = always_batch_cond_uncond or not (lowvram or medvram)
use_scale_latent_for_hires_fix = False
# Arguments
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device-id", type=str, help="Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before)", default=None)
cmd_opts = parser.parse_args()

# Hardware and optimizations
# ----------------------------------------
weight_load_location = None if lowram else "cpu"

parallel_processing_allowed = not lowvram and not medvram
safe.run(devices.enable_tf32, "Enabling TF32")
devices.set(devices.get_optimal_device(), 'half')

# State
# ----------------------------------------
instance = None
sdmodel = None
hnmodel = None
clipmodel = None

from src_plugins.sd1111_plugin import sd_hypernetwork, sd_models, SDJob
from src_plugins.sd1111_plugin.SDJob import SDJob_img, SDJob_txt
from src_plugins.sd1111_plugin import sd_hijack


class SDPlugin(Plugin):
    # TODO os env variables for no reason, enjoy
    taming_transformers_commit_hash = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "24268930bf1dce879235a7fddd0b2355b84d7ea6")
    stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc")
    k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "f4e99857772fc3a126ba886aadf795a332774878")

    def title(self):
        return "Stable Diffusion AUTO1111"

    def describe(self):
        return "Stable Diffusion plugin adapted from AUTOMATIC1111's code."

    def init(self):
        global instance
        instance = self

    def install(self):
        self.gitclone("https://github.com/CompVis/taming-transformers.git", "taming-transformers", SDPlugin.taming_transformers_commit_hash)
        self.gitclone("https://github.com/CompVis/stable-diffusion.git", 'stable_diffusion', SDPlugin.stable_diffusion_commit_hash)
        self.gitclone("https://github.com/crowsonkb/k-diffusion.git", 'k-diffusion', SDPlugin.k_diffusion_commit_hash)

        # TODO install xformers if enabled
        if attention == SDAttention.XFORMERS:
            pass

    def load(self):
        sd_hijack.init()
        # Interrogate
        # import interrogate
        # interrogator = interrogate.InterrogateModels("interrogate")

        sd_hypernetwork.discover_hypernetworks(res("hypernetworks"))
        sd_models.discover_sdmodels()
        sd_models.load_sdmodel()

        # codeformer.setup_model(cmd_opts.codeformer_models_path)
        # gfpgan.setup_model(cmd_opts.gfpgan_models_path)
        # SDPlugin.face_restorers.append(modules.face_restoration.FaceRestoration())
        # modelloader.load_upscalers()

    def select_hn(self, name: str):
        sd_hypernetwork.load_hypernetwork(name)

    def txt2img(self,
                job: SDJob_txt = None,
                prompt: str | list = "",
                promptneg: str | list = "",
                sampler: str = "euler-a",
                steps: int = 22,
                cfg=7,
                width=512,
                height=512,
                seed=-1):
        SDJob.process_images(job or SDJob_txt(prompt=prompt, promptneg=promptneg, sampler=sampler, steps=steps, cfg=cfg, width=width, height=height, seed=seed))

    def img2img(self,
                job: SDJob_img = None,
                pil: Image | list = None,
                mask_pil: Image = None,
                mask_latent = None,
                prompt: str = "",
                promptneg: str = "",
                sampler: str = "euler-a",
                steps: int = 22,
                cfg=7,
                width=512,
                height=512,
                seed=-1):
        SDJob.process_images(job or SDJob_img(pil=pil, mask_latent=mask_latent, mask_pil=mask_pil, prompt=prompt, promptneg=promptneg, sampler=sampler, steps=steps, cfg=cfg, width=width, height=height, seed=seed))

def res(join=""):
    return plugins.get('sd1111_plugin').res(join)