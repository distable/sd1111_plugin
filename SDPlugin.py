import argparse
import os
import re
from pathlib import Path

import src_plugins.sd1111_plugin.SDState
import user_conf
from src_core.lib.corelib import to_dict
from src_core.plugins import plugjob
from src_core.classes.Plugin import Plugin
# Options
from src_plugins.sd1111_plugin import __conf__, safe, sd_job, sd_paths
from src_core.lib import devices
from src_plugins.sd1111_plugin.SDAttention import SDAttention

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


class sd_auto(sd_img, sd_txt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SDPlugin(Plugin):
    # TODO os env variables for no reason, enjoy

    def title(self):
        return "Stable Diffusion AUTO1111"

    def describe(self):
        return "Stable Diffusion plugin adapted from AUTOMATIC1111's code."

    def init(self):
        src_plugins.sd1111_plugin.SDState.instance = self

    def enable_midas_autodownload(self):
        """
        Gives the ldm.modules.midas.api.load_model function automatic downloading.
        When the 512-depth-ema model, and other future models like it, is loaded,
        it calls midas.api.load_model to load the associated midas depth model.
        This function applies a wrapper to download the model to the correct
        location automatically.
        """
        if not 'v2' in str(sd_paths.config):
            return

        from os import mkdir
        from urllib import request
        import os
        import ldm.modules.midas as midas

        midas_path = self.res() / 'midas'

        # stable-diffusion-stability-ai hard-codes the midas model path to
        # a location that differs from where other scripts using this model look.
        # HACK: Overriding the path here.
        for k, v in midas.api.ISL_PATHS.items():
            file_name = os.path.basename(v)
            midas.api.ISL_PATHS[k] = os.path.join(midas_path, file_name)

        midas_urls = {
            "dpt_large"      : "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
            "dpt_hybrid"     : "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
            "midas_v21"      : "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
            "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
        }

        midas.api.load_model_inner = midas.api.load_model

        def load_model_wrapper(model_type):
            path = midas.api.ISL_PATHS[model_type]
            if not os.path.exists(path):
                if not os.path.exists(midas_path):
                    mkdir(midas_path)

                print(f"Downloading midas model weights for {model_type} to {path}")
                request.urlretrieve(midas_urls[model_type], path)
                print(f"{model_type} downloaded")

            return midas.api.load_model_inner(model_type)

        midas.api.load_model = load_model_wrapper

    def load(self):
        devices.set(devices.get_optimal_device(), 'full')
        sd_hijack.init()
        # Interrogate
        # import interrogate
        # interrogator = interrogate.InterrogateModels("interrogate")

        self.enable_midas_autodownload()
        sd_hypernetwork.discover_hypernetworks(sd_paths.res("hypernetworks"))
        sd_models.discover_sdmodels()
        sd_models.load_sdmodel()

        # codeformer.setup_model(cmd_opts.codeformer_models_path)
        # gfpgan.setup_model(cmd_opts.gfpgan_models_path)
        # SDPlugin.face_restorers.append(modules.face_restoration.FaceRestoration())
        # modelloader.load_upscalers()

    def install(self):
        import platform
        from src_core.installing import pipargs

        def fnr(directory: Path, match, replace, fargs=[]):
            """
            Find and replace
            """
            # Escape dots in match
            # match = match.replace(".", "\.")

            # Iterate all files recursively in directory
            # If it's a .py file, open into a string
            for root, dirs, files in os.walk(directory):
                root = Path(root)
                for txt1 in files:
                    if txt1.endswith(".py"):
                        file_path = root / txt1
                        with open(file_path, "r") as f:
                            txt1 = f.read()

                        # Replace the target string with regex search
                        txt2 = re.sub(match + "\(", replace + "(", txt1)
                        if txt1 != txt2 and fargs:
                            # Regex find all replaced indices
                            indices = [m.start() for m in re.finditer(match, txt2)]
                            for i in reversed(indices):
                                # and search for its matching closing parenthesis
                                # Then, add the fargs to the end of the function call
                                depth = 1
                                start_index = txt2.find("(", i) + 1
                                end_index = -1
                                for i in range(start_index, len(txt2)):
                                    slice = txt2[start_index:i]
                                    if txt2[i] == "(":
                                        depth += 1
                                    elif txt2[i] == ")":
                                        depth -= 1
                                        if depth == 0:
                                            end_index = i
                                            break


                                if end_index != -1:
                                    # Args like ['arg=1', 'arg2=2'] are converted to ', arg=1, arg2=2'
                                    if start_index == end_index:
                                        # Empty args
                                        txt2 = f"{txt2[:end_index]}{', '.join(fargs)}{txt2[end_index:]}"
                                    else:
                                        txt2 = f"{txt2[:end_index]}, {', '.join(fargs)}{txt2[end_index:]}"

                                # txt2 = txt2.replace("def " + match, "def " + match + "(" + ",".join(fargs) + ")")

                        # Write the file out again
                        file_path.unlink()
                        with open(file_path, "w") as f:
                            f.write(txt2)

        # TODO install xformers if enabled
        if __conf__.bit8:
            pipargs("install bitsandbytes")
            dir = self.repo('stable_diffusion')

            fnr(dir, 'optim.AdamW', 'bnb.optim.AdamW8bit')
            fnr(dir, 'optim.Adam', 'bnb.optim.Adam8bit')
            fnr(dir, 'optim.RMSprop', 'bnb.optim.RMSprop8bit')
            fnr(dir, 'nn.Embedding', 'bnb.nn.Embedding')
            fnr(dir, 'nn.Linear', 'bnb.nn.Linear8BitLt', fargs=['threshold=6.0', 'has_fp16_weights=False'])

        if __conf__.attention == SDAttention.XFORMERS:
            try:
                import xformers
                print(xformers)
            except ImportError:
                if platform.python_version().startswith("3.10"):
                    if platform.system() == "Windows":
                        pipargs("install https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/c/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl", "xformers")
                    elif platform.system() == "Linux":
                        pass
                        # pipargs("install -U ninja", "xformers")
                        # pipargs("install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers", "xformers")

            try:
                import xformers
                print(xformers)
                print("Xformers failed to install")
            except ImportError:
                pass


    def select_hn(self, name: str):
        sd_hypernetwork.load_hypernetwork(name)

    @plugjob(key='sd_job')
    def sd1111(self, args: sd_auto):
        return sd_job.process_images(args)

    @plugjob(key='sd_job')
    def txt2img(self, args: sd_txt):
        return sd_job.process_images(args)

    @plugjob(key='sd_job')
    def img2img(self, j: sd_img):
        if j.ctx.image is None:
            txt = sd_txt(**to_dict(j))
            txt.ctx = j.ctx
            txt.job = j.job
            txt.session = j.session
            return self.txt2img(self, txt)

        if j.init_images is None:
            j.init_images = [j.ctx.image]

        ret = sd_job.process_images(j)

        return ret
