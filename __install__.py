import os

from src_core.installing import gitclone

taming_transformers_commit_hash = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "24268930bf1dce879235a7fddd0b2355b84d7ea6")
k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "60e5042ca0da89c14d1dd59d73883280f8fce991")
stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc")
# stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "c12d960d1ee4f9134c2516862ef991ec52d3f59e")

gitclone("https://github.com/CompVis/taming-transformers.git", taming_transformers_commit_hash)
gitclone("https://github.com/crowsonkb/k-diffusion.git", k_diffusion_commit_hash)
gitclone("https://github.com/CompVis/stable-diffusion.git", stable_diffusion_commit_hash)
# gitclone("https://github.com/Stability-AI/stablediffusion.git", stable_diffusion_commit_hash)

# sys.path.insert(0, (paths.plug_repos / "stable_diffusion").as_posix())
# sys.path.insert(0, (paths.plug_repos / "stable_diffusion" / "ldm").as_posix())

# TODO install xformers if enabled



