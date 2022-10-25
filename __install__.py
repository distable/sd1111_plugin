import os

from src_core.installing import gitclone

taming_transformers_commit_hash = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "24268930bf1dce879235a7fddd0b2355b84d7ea6")
stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc")
k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "f4e99857772fc3a126ba886aadf795a332774878")

gitclone("https://github.com/CompVis/taming-transformers.git", taming_transformers_commit_hash)
gitclone("https://github.com/CompVis/stable-diffusion.git", stable_diffusion_commit_hash)
gitclone("https://github.com/crowsonkb/k-diffusion.git", k_diffusion_commit_hash)

# sys.path.insert(0, (paths.plug_repos / "stable_diffusion").as_posix())
# sys.path.insert(0, (paths.plug_repos / "stable_diffusion" / "ldm").as_posix())

# TODO install xformers if enabled
