from src_plugins.sd1111_plugin.SDAttention import SDAttention

# Paths
res_ckpt = 'sd-v1-5.ckpt'
res_vae = "vae.vae.pt"

attention = SDAttention.SPLIT_DOGGETT
# attention = SDAttention.SUBQUAD
lowvram = True
medvram = True
lowram = False
precision = 'full'
no_half = True
no_half_vae = True
opt_channelslast = True
always_batch_cond_uncond = False  # disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram
xformers = False
bit8 = False
force_enable_xformers = False
batch_cond_uncond = always_batch_cond_uncond or not (lowvram or medvram)
use_cpu = False
use_scale_latent_for_hires_fix = False
weight_load_device = None if lowram else "cpu"
parallel_processing_allowed = not lowvram and not medvram

comma_padding_backtrack = 0
enable_emphasis = True
use_old_emphasis_implementation = False

CLIP_stop_at_last_layers = 0

# TODO deployed:
# batch_cond_uncond
sub_quad_q_chunk_size = 2048
sub_quad_kv_chunk_size = None
sub_quad_chunk_threshold = None

# parser.add_argument("--opt-sub-quad-attention", action='store_true', help="enable memory efficient sub-quadratic cross-attention layer optimization. By default, it's on when cuda is unavailable.")
# parser.add_argument("--sub-quad-q-chunk-size", type=int, help="query chunk size for the sub-quadratic cross-attention layer optimization to use", default=1024)
# parser.add_argument("--sub-quad-kv-chunk-size", type=int, help="kv chunk size for the sub-quadratic cross-attention layer optimization to use", default=None)
# parser.add_argument("--sub-quad-chunk-threshold", type=int, help="the size threshold in bytes for the sub-quadratic cross-attention layer optimization to use chunking", default=None)