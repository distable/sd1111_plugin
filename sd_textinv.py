import os
import sys
import traceback

import torch
import tqdm
import datetime
import csv

from PIL import Image, PngImagePlugin

import src_plugins.sd1111_plugin.options
import src_plugins.sd1111_plugin.sd_paths
import src_plugins.sd1111_plugin.SDState
from src_plugins.sd1111_plugin import sd_hijack, sd_models
from src_core.lib import devices
import src_plugins.sd1111_plugin.sd_textinv_dataset
from src_plugins.sd1111_plugin.sd_job import process_images, sd_txt
from src_plugins.sd1111_plugin.sd_textinv_learn_schedule import LearnRateScheduler
from src_plugins.sd1111_plugin.image_embedding import (embedding_to_b64, embedding_from_b64,
                                                       insert_image_data_embed, extract_image_data_embed,
                                                       caption_image_overlay)

import src_core.classes.paths


class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum


class EmbeddingDatabase:
    def __init__(self, embeddings_dir):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.dir_mtime = None
        self.embeddings_dir = embeddings_dir

    def register_embedding(self, embedding, model):

        self.word_embeddings[embedding.name] = embedding

        ids = model.cond_stage_model.tokenizer([embedding.name], add_special_tokens=False)['input_ids'][0]

        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []

        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)

        return embedding

    def load_textual_inversion_embeddings(self):
        mt = os.path.getmtime(self.embeddings_dir)
        if self.dir_mtime is not None and mt <= self.dir_mtime:
            return

        self.dir_mtime = mt
        self.ids_lookup.clear()
        self.word_embeddings.clear()

        def process_file(path, filename):
            name = os.path.splitext(filename)[0]

            data = []

            if os.path.splitext(filename.upper())[-1] in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
                embed_image = Image.open(path)
                if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                    data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
                    name = data.get_plug('name', name)
                else:
                    data = extract_image_data_embed(embed_image)
                    name = data.get_plug('name', name)
            else:
                data = torch.load(path, map_location="cpu")

            # textual inversion embeddings
            if 'string_to_param' in data:
                param_dict = data['string_to_param']
                if hasattr(param_dict, '_parameters'):
                    param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
                assert len(param_dict) == 1, 'embedding file has multiple terms in it'
                emb = next(iter(param_dict.items()))[1]
            # diffuser concepts
            elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
                assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

                emb = next(iter(data.values()))
                if len(emb.shape) == 1:
                    emb = emb.unsqueeze(0)
            else:
                raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

            vec = emb.detach().to(devices.device, dtype=torch.float32)
            embedding = Embedding(vec, name)
            embedding.step = data.get_plug('step', None)
            embedding.sd_checkpoint = data.get_plug('hash', None)
            embedding.sd_checkpoint_name = data.get_plug('sd_checkpoint_name', None)
            self.register_embedding(embedding, src_plugins.sd1111_plugin.SDState.sdmodel)

        for fn in os.listdir(self.embeddings_dir):
            try:
                fullfn = os.path.join(self.embeddings_dir, fn)

                if os.stat(fullfn).st_size == 0:
                    continue

                process_file(fullfn, fn)
            except Exception:
                print(f"Error loading emedding {fn}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

        print(f"Loaded a total of {len(self.word_embeddings)} textual inversion embeddings.")
        print("Embeddings:", ', '.join(self.word_embeddings.keys()))

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None


def create_embedding(name, num_vectors_per_token, overwrite_old, init_text='*'):
    cond_model = src_plugins.sd1111_plugin.SDState.sdmodel.cond_stage_model
    embedding_layer = cond_model.wrapped.transformer.text_model.embeddings

    ids = cond_model.tokenizer(init_text, max_length=num_vectors_per_token, return_tensors="pt", add_special_tokens=False)["input_ids"]
    embedded = embedding_layer.token_embedding.wrapped(ids.to(devices.device)).squeeze(0)
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=devices.device)

    for i in range(num_vectors_per_token):
        vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]

    fn = os.path.join(src_core.classes.paths.plug_embeddings, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    embedding = Embedding(vec, name)
    embedding.step = 0
    embedding.save(fn)

    return fn


def write_loss(log_directory, filename, step, epoch_len, values):
    if src_plugins.sd1111_plugin.options.opts.training_write_csv_every == 0:
        return

    if step % src_plugins.sd1111_plugin.options.opts.training_write_csv_every != 0:
        return

    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True

    with open(os.path.join(log_directory, filename), "a+", newline='') as fout:
        csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])

        if write_csv_header:
            csv_writer.writeheader()

        epoch = step // epoch_len
        epoch_step = step - epoch * epoch_len

        csv_writer.writerow({
            "step": step + 1,
            "epoch": epoch + 1,
            "epoch_step": epoch_step + 1,
            **values,
        })


def train_embedding(embedding_name, learn_rate, batch_size, data_root, log_directory, training_width, training_height, steps, create_image_every, save_embedding_every, template_file, save_image_with_stored_embedding, preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height):
    assert embedding_name, 'embedding not selected'

    # SDPlugin.state.textinfo = "Initializing textual inversion training..."
    # SDPlugin.state.job_count = steps

    filename = os.path.join(src_core.classes.paths.plug_embeddings, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None

    cond_model = src_plugins.sd1111_plugin.SDState.sdmodel.cond_stage_model

    # SDPlugin.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    with torch.autocast("cuda"):
        ds = src_plugins.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=src_plugins.sd1111_plugin.options.opts.training_image_repeats_per_epoch, placeholder_token=embedding_name, model=src_plugins.sd1111_plugin.SDState.sdmodel, device=devices.device, template_file=template_file, batch_size=batch_size)

    hijack = sd_hijack.model_hijack

    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    embedding.vec.requires_grad = True

    losses = torch.zeros((32,))

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    embedding_yet_to_be_embedded = False

    ititial_step = embedding.progress_i or 0
    if ititial_step > steps:
        return embedding, filename

    scheduler = LearnRateScheduler(learn_rate, steps, ititial_step)
    optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate)

    pbar = tqdm.tqdm(enumerate(ds), total=steps-ititial_step)
    for i, entries in pbar:
        embedding.progress_i = i + ititial_step

        scheduler.apply(optimizer, embedding.progress_i)
        if scheduler.finished:
            break

        # if SDPlugin.state.interrupted:
            break

        with torch.autocast("cuda"):
            c = cond_model([entry.cond_text for entry in entries])
            x = torch.stack([entry.latent for entry in entries]).to(devices.device)
            loss = src_plugins.sd1111_plugin.SDState.sdmodel(x, c)[0]
            del x

            losses[embedding.progress_i % losses.shape[0]] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        epoch_num = embedding.progress_i // len(ds)
        epoch_step = embedding.progress_i - (epoch_num * len(ds)) + 1

        pbar.set_description(f"[Epoch {epoch_num}: {epoch_step}/{len(ds)}]loss: {losses.mean():.7f}")

        if embedding.progress_i > 0 and embedding_dir is not None and embedding.progress_i % save_embedding_every == 0:
            last_saved_file = os.path.join(embedding_dir, f'{embedding_name}-{embedding.progress_i}.pt')
            embedding.save(last_saved_file)
            embedding_yet_to_be_embedded = True

        write_loss(log_directory, "textual_inversion_loss.csv", embedding.progress_i, len(ds), {
            "loss": f"{losses.mean():.7f}",
            "learn_rate": scheduler.learn_rate
        })

        if embedding.progress_i > 0 and images_dir is not None and embedding.progress_i % create_image_every == 0:
            last_saved_image = os.path.join(images_dir, f'{embedding_name}-{embedding.progress_i}.png')

            from src_plugins.sd1111_plugin import sd_job
            p = sd_txt(
                sd_model=src_plugins.sd1111_plugin.SDState.sdmodel,
                do_not_save_grid=True,
                do_not_save_samples=True,
                do_not_reload_embeddings=True,
            )

            if preview_from_txt2img:
                p.prompt = preview_prompt
                p.promptneg = preview_negative_prompt
                p.steps = preview_steps
                p.sampler_index = preview_sampler_index
                p.cfg = preview_cfg_scale
                p.seed = preview_seed
                p.width = preview_width
                p.height = preview_height
            else:
                p.prompt = entries[0].cond_text
                p.steps = 20
                p.width = training_width
                p.height = training_height

            preview_text = p.prompt

            processed = process_images(p)
            image = processed.images[0]

            # SDPlugin.state.current_image = image

            if save_image_with_stored_embedding and os.path.exists(last_saved_file) and embedding_yet_to_be_embedded:

                last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{embedding.progress_i}.png')

                info = PngImagePlugin.PngInfo()
                data = torch.load(last_saved_file)
                info.add_text("sd-ti-embedding", embedding_to_b64(data))

                title = "<{}>".format(data.get_plug('name', '???'))

                try:
                    vectorSize = list(data['string_to_param'].values())[0].shape[0]
                except Exception as e:
                    vectorSize = '?'

                checkpoint = sd_models.select_checkpoint()
                footer_left = checkpoint.model_name
                footer_mid = '[{}]'.format(checkpoint.hash)
                footer_right = '{}v {}s'.format(vectorSize, embedding.progress_i)

                captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                captioned_image = insert_image_data_embed(captioned_image, data)

                captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                embedding_yet_to_be_embedded = False

            image.save(last_saved_image)

            last_saved_image += f", prompt: {preview_text}"

        # SDPlugin.state.job_no = embedding.progress_i

#         SDPlugin.state.textinfo = f"""
# <p>
# Loss: {losses.mean():.7f}<br/>
# Step: {embedding.progress_i}<br/>
# Last prompt: {html.escape(entries[0].cond_text)}<br/>
# Last saved embedding: {html.escape(last_saved_file)}<br/>
# Last saved image: {html.escape(last_saved_image)}<br/>
# </p>
# """

    checkpoint = sd_models.select_checkpoint()

    embedding.sd_checkpoint = checkpoint.hash
    embedding.sd_checkpoint_name = checkpoint.model_name
    embedding.cached_checksum = None
    embedding.save(filename)

    return embedding, filename
