import user_conf
from src_plugins.sd1111_plugin.SDAttention import SDAttention

attention = SDAttention.SPLIT_DOGGETT
lowvram = False
medvram = True
lowram = False
precision = "full"
no_half = True
no_half_vae = True
opt_channelslast = False
always_batch_cond_uncond = False  # disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram
xformers = False
force_enable_xformers = False
batch_cond_uncond = always_batch_cond_uncond or not (lowvram or medvram)
use_cpu = False
use_scale_latent_for_hires_fix = False
weight_load_location = None if lowram else "cpu"
parallel_processing_allowed = not lowvram and not medvram
