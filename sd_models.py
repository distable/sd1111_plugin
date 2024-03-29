import collections
import os.path
import sys
from collections import namedtuple
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src_core import plugins
from . import __conf__
from src_plugins.sd1111.sd_hijack_inpainting import do_inpainting_hijack, should_hijack_inpainting
from src_core.lib import devices, modellib
from src_core.classes.printlib import printerr
from src_plugins.sd1111 import options, sd_paths, SDState

g_infos = {}
g_loaded = collections.OrderedDict()

ckpt_dict_replacements = {
    'cond_stage_model.transformer.embeddings.'      : 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.'         : 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}
vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}

CheckpointInfo = namedtuple("CheckpointInfo", ['filename', 'title', 'hash', 'model_name', 'config'])


def checkpoint_titles():
    return sorted([x.title for x in g_infos.values()])


def discover_sdmodels():
    g_infos.clear()
    all_paths = modellib.discover_models(model_dir=plugins.get('sd1111').res(),
                                         command_path=plugins.get('sd1111').res(),
                                         ext_filter=[".ckpt"])

    def modeltitle(path, shorthash):
        return f'{Path(path).name} [{shorthash}]', Path(path).with_suffix("").name


    cmd_ckpt = plugins.get('sd1111').res(__conf__.res_ckpt)
    if os.path.exists(cmd_ckpt):
        h = get_model_hash(cmd_ckpt)
        title, short_model_name = modeltitle(cmd_ckpt, h)
        g_infos[title] = CheckpointInfo(cmd_ckpt, title, h, short_model_name, sd_paths.config)
        # options.opts.context['sd_model_checkpoint'] = title

    for filename in all_paths:
        h = get_model_hash(filename)
        title, short_model_name = modeltitle(filename, h)

        basename, _ = os.path.splitext(filename)
        config = basename + ".yaml"
        if not os.path.exists(config):
            config = sd_paths.config

        g_infos[title] = CheckpointInfo(filename, title, h, short_model_name, config)


def get_closest_by_name(search_name):
    applicable = sorted([info for info in g_infos.values() if search_name in info.title], key=lambda x: len(x.title))
    if len(applicable) > 0:
        return applicable[0]
    return None


def get_model_hash(filename):
    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


def get_checkpoint(path=None):
    """
    Set the current active checkpoint
    """
    info = g_infos.get(path, None)
    if info is not None:
        return info

    if len(g_infos) == 0:
        printerr(f"No checkpoints found. When searching for checkpoints, looked at:")
        ckpt = plugins.get('sd1111').res(__conf__.res_ckpt)
        if ckpt is not None:
            printerr(f" - file {os.path.abspath(ckpt)}")
        printerr(f" - directory {sd_paths.res()}")

        printerr(f"Can't run without a checkpoint. Find and place a .ckpt file into any of those locations. The program will exit.")
        exit(1)

    info = next(iter(g_infos.values()))
    if path is not None:
        print(f"Checkpoint {path} not found; loading fallback {info.title}", file=sys.stderr)

    return info


def transform_ckpt_dict_key(k):
    for text, replacement in ckpt_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


def get_state_dict_from_ckpt(pl_sd):
    if "state_dict" in pl_sd:
        pl_sd = pl_sd["state_dict"]

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_ckpt_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def load_model_weights(model, info):
    path = info.filename
    hash = info.hash

    if info not in g_loaded:
        print(f"{path}")

        # TODO what does 'pl' stand for ?
        pl_sd = torch.load(path, map_location=__conf__.weight_load_device)
        # if "global_step" in pl_sd:
        #     print(f"Global Step: {pl_sd['global_step']}")

        sd = get_state_dict_from_ckpt(pl_sd)
        missing, extra = model.load_state_dict(sd, strict=False)

        if __conf__.opt_channelslast:
            model.to(memory_format=torch.channels_last)

        if not __conf__.no_half:
            model.half()

        devices.dtype = torch.float32 if __conf__.no_half else torch.float16
        devices.dtype_vae = torch.float32 if __conf__.no_half or __conf__.no_half_vae else torch.float16

        vae_file = os.path.splitext(path)[0] + ".vae.pt"
        if not os.path.exists(vae_file):
            vae_file = plugins.get('sd1111').res(__conf__.res_vae)

        if os.path.exists(vae_file):
            print(f"{vae_file}")
            vae_ckpt = torch.load(vae_file, map_location=__conf__.weight_load_device)
            vae_dict = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss" and k not in vae_ignore_keys}
            model.first_stage_model.load_state_dict(vae_dict)

        model.first_stage_model.to(devices.dtype_vae)

        g_loaded[info] = model.state_dict().copy()
        while len(g_loaded) > options.opts.sd_checkpoint_cache:
            g_loaded.popitem(last=False)  # LRU
    else:
        print(f"[{hash}] from cache")
        g_loaded.move_to_end(info)
        model.load_state_dict(g_loaded[info])

    model.hash = hash
    model.ckptpath = path
    model.info = info


def load_sdmodel(info=None):
    from src_plugins.sd1111 import sd_hijack, modelsplit
    info = info or get_checkpoint()

    if info.config != sd_paths.config:
        print(f"Loading config from: {info.config}")

    config = OmegaConf.load(info.config)

    if should_hijack_inpainting(info):
        # Hardcoded config for now...
        config.model.target = "ldm.models.diffusion.ddpm.LatentInpaintDiffusion"
        config.model.params.use_ema = False
        config.model.params.conditioning_key = "hybrid"
        config.model.params.unet_config.params.in_channels = 9

        # Create a "fake" config with a different name so that we know to unload it when switching models.
        info = info._replace(config=info.config.replace(".yaml", "-inpainting.yaml"))

    do_inpainting_hijack()
    from ldm.util import instantiate_from_config
    sdmodel = instantiate_from_config(config.model)
    load_model_weights(sdmodel, info)

    # import tomesd
    # tomesd.apply_patch(sdmodel, ratio=0.5)

    if __conf__.lowvram or __conf__.medvram:
        modelsplit.setup_for_low_vram(sdmodel, __conf__.medvram)
    else:
        sdmodel.to(devices.device)

    sd_hijack.model_hijack.hijack(sdmodel)

    sdmodel.eval()
    SDState.sdmodel = sdmodel

    # script_callbacks.model_loaded_callback(sdmodel)

    print(f"Model loaded.")
    return sdmodel


def reload_model_weights(sdmodel, info=None):
    import modelsplit, sd_hijack
    info = info or get_checkpoint()

    if sdmodel.ckptpath == info.filename:
        return

    if sd_paths.config != info.config or should_hijack_inpainting(info) != should_hijack_inpainting(sdmodel.info):
        g_loaded.clear()
        load_sdmodel(info)
        return SDState.sdmodel

    if __conf__.lowvram or __conf__.medvram:
        modelsplit.send_everything_to_cpu()
    else:
        sdmodel.to(devices.cpu)

    sd_hijack.model_hijack.undo_hijack(sdmodel)

    load_model_weights(sdmodel, info)

    sd_hijack.model_hijack.hijack(sdmodel)
    # script_callbacks.model_loaded_callback(sd_model)

    if not __conf__.lowvram and not __conf__.medvram:
        sdmodel.to(devices.device)

    print(f"Weights loaded.")
    return sdmodel
