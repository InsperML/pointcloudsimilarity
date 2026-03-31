#!/usr/bin/env python3
"""
laion_extract_embeddings.py

Streams laion/laion400m from Hugging Face, computes CLIP image & text embeddings,
and saves them into sharded .pth files.

Notes:
- The script streams the dataset (no full download).
- It writes shard files to avoid accumulating huge tensors in memory.
- It is robust to missing images/urls and will skip corrupted items.
- Default CLIP model: openai/clip-vit-base-patch32 (changeable).
"""

import os
import io
import math
import time
import json
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from PIL import Image, UnidentifiedImageError
import requests
from tqdm.auto import tqdm

# ---------------------------
# User-configurable settings
# ---------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"   # change if you prefer another CLIP
DATASET = "laion/laion400m"
SPLIT = "train"
BATCH_SIZE = 64            # batch size for text and image separately (tune for GPU/CPU memory)
SHARD_SIZE = 50_000        # number of examples per .pth shard file
MAX_EXAMPLES = None        # set to None for all examples; otherwise set an int for testing
OUTPUT_DIR = "laion_embeddings_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_AUTH_TOKEN = os.getenv("HF_TOKEN", None)   # optional
TIMEOUT = 10               # requests timeout for image downloads in seconds
CACHE_DIR = "/mnt/data3/img_datasets"
# ---------------------------
# Helpers
# ---------------------------
def mkdir(path):
    os.makedirs(path, exist_ok=True)

def pil_from_url(url):
    """Download image via requests and return PIL.Image or raise."""
    resp = requests.get(url, timeout=TIMEOUT, stream=True)
    resp.raise_for_status()
    img_bytes = io.BytesIO(resp.content)
    return Image.open(img_bytes).convert("RGB")

def prepare_image(example, image_field_candidates=("image", "img", "image_url", "url")):
    """
    Try to extract a PIL image from the dataset example.
    Returns PIL.Image or raises Exception.
    """
    # Case 1: dataset already contains an image object under 'image' (PIL/Image or bytes)
    for f in image_field_candidates:
        if f in example and example[f] is not None:
            val = example[f]
            # If HF dataset returns PIL.Image already (sometimes it does)
            if isinstance(val, Image.Image):
                return val
            # If bytes
            if isinstance(val, (bytes, bytearray)):
                return Image.open(io.BytesIO(val)).convert("RGB")
            # If it's a URL string
            if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                return pil_from_url(val)
    # Fallback: attempt to find any string field that looks like a URL
    for k, v in example.items():
        if isinstance(v, str) and v.startswith("http"):
            # try to download
            return pil_from_url(v)
    raise ValueError("No image found in example")

def get_text_from_example(example, text_field_candidates=("caption", "text", "caption_text", "title")):
    for f in text_field_candidates:
        if f in example and example[f] is not None:
            return str(example[f])
    # fallback: try to find first string field that is not a URL
    for k, v in example.items():
        if isinstance(v, str) and not v.startswith("http"):
            return v
    return ""


# ---------------------------
# Main extraction logic
# ---------------------------
def run():
    mkdir(OUTPUT_DIR)
    device = torch.device(DEVICE)
    print(f"Device: {device}, model: {MODEL_NAME}")
    print("Loading model and processor...")
    model = CLIPModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model.to(device)
    model.eval()

    print(f"Streaming dataset {DATASET} split={SPLIT} ... (this may take a while)")
    ds_kwargs = {}
    if HF_AUTH_TOKEN:
        ds_kwargs["use_auth_token"] = HF_AUTH_TOKEN

    dataset = load_dataset(DATASET, split=SPLIT, streaming=True, cache_dir=CACHE_DIR, **ds_kwargs)

    image_embeddings_shards = []
    text_embeddings_shards = []
    shard_count = 0
    global_index = 0

    # buffers to accumulate embeddings before writing shard to disk
    image_buf = []
    text_buf = []
    id_buf = []

    def flush_shard(ifinal=False):
        nonlocal shard_count, image_buf, text_buf, id_buf
        if len(id_buf) == 0:
            return
        shard_fname = os.path.join(OUTPUT_DIR, f"embeddings_shard_{shard_count:05d}.pth")
        # stack tensors
        image_tensor = torch.stack(image_buf) if len(image_buf) > 0 else torch.empty((0, model.visual_projection.out_features))
        text_tensor = torch.stack(text_buf) if len(text_buf) > 0 else torch.empty((0, model.text_projection.out_features))
        ids = id_buf
        torch.save({
            "image_embeddings": image_tensor.cpu(),
            "text_embeddings": text_tensor.cpu(),
            "ids": ids
        }, shard_fname)
        print(f"Saved shard {shard_fname} (n={len(ids)})")
        shard_count += 1
        # reset buffers
        image_buf = []
        text_buf = []
        id_buf = []

    batch_images = []
    batch_texts = []
    batch_ids = []

    pbar = tqdm(total=MAX_EXAMPLES or math.inf, desc="Processed", unit="img")

    try:
        for example in dataset:
            if MAX_EXAMPLES and global_index >= MAX_EXAMPLES:
                break

            # Attempt to extract text and image
            try:
                text = get_text_from_example(example)
            except Exception:
                text = ""

            # Try extract image (skip if fail)
            pil_image = None
            try:
                pil_image = prepare_image(example)
            except Exception:
                # skip image if any error
                pil_image = None

            # Prepare batch entries; keep an ID (index) to track correspondence
            batch_ids.append(global_index)
            batch_texts.append(text)
            batch_images.append(pil_image)  # can be None; we will handle later

            global_index += 1
            pbar.update(1)

            # If we reached a processing batch size, compute embeddings
            if len(batch_ids) >= BATCH_SIZE:
                # Text embeddings
                with torch.no_grad():
                    # Text: tokenizer in processor
                    try:
                        text_inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                        text_outputs = model.get_text_features(**text_inputs)  # (B, D)
                    except Exception as e:
                        # fallback: create zero tensor for text
                        print(f"Warning: text encoding failed for batch starting id {batch_ids[0]}: {e}")
                        text_outputs = torch.zeros((len(batch_ids), model.config.projection_dim), device=device)

                    # Image embeddings: build list of valid images and masks to put zeros for missing
                    valid_images = []
                    valid_indices = []
                    for i, im in enumerate(batch_images):
                        if im is None:
                            continue
                        valid_images.append(im)
                        valid_indices.append(i)

                    if len(valid_images) > 0:
                        try:
                            image_inputs = processor(images=valid_images, return_tensors="pt")
                            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                            image_outputs_valid = model.get_image_features(**image_inputs)  # (N_valid, D)
                        except Exception as e:
                            print(f"Warning: image encoding failed for batch ids {batch_ids}: {e}")
                            image_outputs_valid = torch.zeros((len(valid_images), model.config.projection_dim), device=device)
                    else:
                        image_outputs_valid = None

                    # Now create full image_outputs aligned with batch size, filling zeros for missing images.
                    image_outputs = []
                    vi = 0
                    for i in range(len(batch_ids)):
                        if image_outputs_valid is not None and i in valid_indices:
                            image_outputs.append(image_outputs_valid[vi])
                            vi += 1
                        else:
                            image_outputs.append(torch.zeros((model.config.projection_dim,), device=device))

                    # normalize optional: uncomment if you want unit vectors
                    # text_outputs = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)
                    # image_outputs = torch.stack(image_outputs)
                    image_outputs = torch.stack(image_outputs)

                # Append to shard buffers (move to cpu to reduce GPU memory pressure)
                for t in image_outputs:
                    image_buf.append(t.detach().cpu())
                for t in text_outputs:
                    text_buf.append(t.detach().cpu())
                for iid in batch_ids:
                    id_buf.append(iid)

                # reset batch
                batch_images = []
                batch_texts = []
                batch_ids = []

                # When shard is large enough, flush to disk
                if len(id_buf) >= SHARD_SIZE:
                    flush_shard()

        # End for-loop. Process any remainder in batch
        if len(batch_ids) > 0:
            with torch.no_grad():
                # Text
                try:
                    text_inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                    text_outputs = model.get_text_features(**text_inputs)
                except Exception as e:
                    print(f"Warning: final text encoding failed: {e}")
                    text_outputs = torch.zeros((len(batch_ids), model.config.projection_dim), device=device)

                # Images
                valid_images = []
                valid_indices = []
                for i, im in enumerate(batch_images):
                    if im is None:
                        continue
                    valid_images.append(im)
                    valid_indices.append(i)

                if len(valid_images) > 0:
                    try:
                        image_inputs = processor(images=valid_images, return_tensors="pt")
                        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                        image_outputs_valid = model.get_image_features(**image_inputs)
                    except Exception as e:
                        print(f"Warning: final image encoding failed: {e}")
                        image_outputs_valid = torch.zeros((len(valid_images), model.config.projection_dim), device=device)
                else:
                    image_outputs_valid = None

                image_outputs = []
                vi = 0
                for i in range(len(batch_ids)):
                    if image_outputs_valid is not None and i in valid_indices:
                        image_outputs.append(image_outputs_valid[vi])
                        vi += 1
                    else:
                        image_outputs.append(torch.zeros((model.config.projection_dim,), device=device))
                image_outputs = torch.stack(image_outputs)

            for t in image_outputs:
                image_buf.append(t.detach().cpu())
            for t in text_outputs:
                text_buf.append(t.detach().cpu())
            for iid in batch_ids:
                id_buf.append(iid)

        # Final flush
        flush_shard(ifinal=True)
        pbar.close()
        print("Done. All shards saved to", OUTPUT_DIR)

    except KeyboardInterrupt:
        print("Interrupted by user. Flushing current buffers...")
        flush_shard()
        pbar.close()
    except Exception as e:
        print("Fatal error:", e)
        flush_shard()
        pbar.close()


if __name__ == "__main__":
    run()
