import argparse

import src_plugins.sd1111_plugin.SDState
import user_conf
from src_core.plugins import plugjob
from src_core.classes.Plugin import Plugin
# Options
from src_plugins.sd1111_plugin import safe, sd_job, sd_paths
from src_core.lib import devices
from src_plugins.sd1111_plugin.SDAttention import SDAttention
from src_plugins.sd1111_plugin.SDOptions import attention

# Arguments
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device-id", type=str, help="Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before)", default=None)
cmd_opts = parser.parse_args()

# Hardware and optimizations
# ----------------------------------------

safe.run(devices.enable_tf32, "Enabling TF32")
devices.set(devices.get_optimal_device(), user_conf.precision)

# State
# ----------------------------------------

from src_plugins.sd1111_plugin import sd_hypernetwork, sd_models
from src_plugins.sd1111_plugin import sd_hijack
from src_plugins.sd1111_plugin.sd_job import sd_img, sd_txt



class SDPlugin(Plugin):
    # TODO os env variables for no reason, enjoy

    def title(self):
        return "Stable Diffusion AUTO1111"

    def describe(self):
        return "Stable Diffusion plugin adapted from AUTOMATIC1111's code."

    def init(self):
        src_plugins.sd1111_plugin.SDState.instance = self

    def load(self):
        devices.set(devices.get_optimal_device(), 'full')
        sd_hijack.init()
        # Interrogate
        # import interrogate
        # interrogator = interrogate.InterrogateModels("interrogate")

        sd_hypernetwork.discover_hypernetworks(sd_paths.res("hypernetworks"))
        sd_models.discover_sdmodels()
        sd_models.load_sdmodel()

        # codeformer.setup_model(cmd_opts.codeformer_models_path)
        # gfpgan.setup_model(cmd_opts.gfpgan_models_path)
        # SDPlugin.face_restorers.append(modules.face_restoration.FaceRestoration())
        # modelloader.load_upscalers()

    def install(self):
        # TODO install xformers if enabled
        if attention == SDAttention.XFORMERS:
            pass

    def select_hn(self, name: str):
        sd_hypernetwork.load_hypernetwork(name)

    @plugjob
    def txt2img(self, job: sd_txt = None, **kwargs):
        return sd_job.process_images(job or sd_txt(**kwargs))

    @plugjob
    def img2img(self, job: sd_img = None, **kwargs):
        return sd_job.process_images(job or sd_img(**kwargs))
