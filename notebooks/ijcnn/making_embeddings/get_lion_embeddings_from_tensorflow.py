#!/usr/bin/env python3
"""
export_laion_embeddings_from_tfds.py

Load laion400m/embeddings from TensorFlow Datasets and write precomputed
image and text embeddings into sharded .pth files.

Each shard produces two files (per SHARD):
 - <OUT_DIR>/image_embeddings_shard_{shard:05d}.pth
 - <OUT_DIR>/text_embeddings_shard_{shard:05d}.pth

Each .pth file contains a dict:
{
  "embeddings": torch.Tensor (N, D),
  "ids": [global_index_0, global_index_1, ...],
  "source_fields": { optional metadata fields saved (like 'caption' or 'url') }
}

Notes:
- This script reads TFDS (may require substantial local disk/network I/O
  depending on TFDS download configuration).
- It is resilient to a variety of possible field names for embeddings.
"""

import os
import argparse
import math
from typing import Any, Dict, Optional, List

import torch
import numpy as np
from tqdm import tqdm

try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
except Exception as e:
    raise RuntimeError("This script requires tensorflow_datasets and tensorflow. "
                       "Install with: pip install 'tensorflow-datasets' tensorflow") from e

# ----------------------------
# Helpers
# ----------------------------
def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def to_torch(x: Any, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array or list to torch tensor with dtype float32."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(np.asarray(x), dtype=dtype)

def pick_field(example: Dict, candidates: List[str]) -> Optional[Any]:
    """Return the first matching field value in example for candidates, or None."""
    for c in candidates:
        if c in example and example[c] is not None:
            return example[c]
    return None

# ----------------------------
# Main logic
# ----------------------------
def export_embeddings(
    dataset_name: str = "laion400m/embeddings",
    split: str = "train",
    out_dir: str = "laion_embeddings_pth",
    shard_size: int = 50_000,
    max_examples: Optional[int] = None,
    resume: bool = False,
    cache_dir: Optional[str] = None,
):
    mkdir(out_dir)
    print(f"Loading TFDS dataset {dataset_name}, split={split} ...")
    # tfds loads as tf.data.Dataset
    ds = tfds.load(dataset_name, split=split, shuffle_files=False, data_dir=cache_dir)

    # Convert to numpy iterator
    ds_numpy = tfds.as_numpy(ds)

    # Candidate field names (defensive)
    image_emb_fields = ["image_embedding", "img_emb", "image_emb", "image_vector", "img_vector"]
    text_emb_fields = ["text_embedding", "text_emb", "caption_embedding", "text_vector"]
    # Helpful metadata fields to optionally save in shard (caption, url)
    metadata_fields = ["caption", "text", "url", "image_url", "id"]

    # Determine resume start index by scanning existing shards if resume=True
    start_index = 0
    shard_index = 0
    if resume:
        # find largest shard index saved
        existing = [fn for fn in os.listdir(out_dir) if fn.startswith("image_embeddings_shard_") and fn.endswith(".pth")]
        if existing:
            existing.sort()
            last = existing[-1]
            # filename pattern: image_embeddings_shard_00000.pth
            try:
                k = int(last.split("_")[-1].split(".")[0])
                shard_index = k + 1
                # compute start_index by summing ids in existing shards could be expensive;
                # instead set start_index = shard_index * shard_size, which is a reasonable resume hint.
                start_index = shard_index * shard_size
                print(f"Resume requested: starting from shard index {shard_index}, estimated start_index {start_index}")
            except Exception:
                print("Could not parse existing shard indices; starting from zero.")

    # Buffers
    image_buf: List[torch.Tensor] = []
    text_buf: List[torch.Tensor] = []
    id_buf: List[int] = []
    meta_buf: List[Dict[str, Any]] = []

    saved_count = 0
    global_index = 0
    pbar_total = max_examples if max_examples is not None else None
    pbar = tqdm(total=pbar_total, desc="Processed examples")

    def flush_shard(shard_i: int):
        nonlocal image_buf, text_buf, id_buf, meta_buf, saved_count
        if len(id_buf) == 0:
            return
        image_fname = os.path.join(out_dir, f"image_embeddings_shard_{shard_i:05d}.pth")
        text_fname = os.path.join(out_dir, f"text_embeddings_shard_{shard_i:05d}.pth")
        # Stack tensors
        image_tensor = torch.stack(image_buf) if len(image_buf) > 0 else torch.empty((0,))
        text_tensor = torch.stack(text_buf) if len(text_buf) > 0 else torch.empty((0,))
        # Save image shard
        torch.save({
            "embeddings": image_tensor,
            "ids": id_buf,
            "metadata": meta_buf
        }, image_fname)
        # Save text shard
        torch.save({
            "embeddings": text_tensor,
            "ids": id_buf,
            "metadata": meta_buf
        }, text_fname)
        print(f"Saved shard {shard_i:05d} -> {len(id_buf)} examples (files: {os.path.basename(image_fname)}, {os.path.basename(text_fname)})")
        saved_count += len(id_buf)
        # reset buffers
        image_buf = []
        text_buf = []
        id_buf = []
        meta_buf = []

    try:
        for example in ds_numpy:
            # Skip until start_index if resuming with a rough index
            if global_index < start_index:
                global_index += 1
                if pbar_total is not None:
                    pbar.update(1)
                continue

            # Optionally stop early
            if max_examples is not None and global_index >= max_examples:
                break

            # Attempt to pick embeddings
            img_emb = pick_field(example, image_emb_fields)
            txt_emb = pick_field(example, text_emb_fields)

            # If fields are nested TF Tensors, convert to numpy first (tfds.as_numpy usually does that)
            # If either embedding missing, skip example (but still count id)
            if img_emb is None and txt_emb is None:
                # nothing to save; skip
                global_index += 1
                if pbar_total is not None:
                    pbar.update(1)
                continue

            # Convert to torch tensors (float32)
            try:
                if img_emb is None:
                    # create zeros or skip; here we create zeros of same dim as text if available
                    if txt_emb is not None:
                        img_tensor = torch.zeros_like(to_torch(txt_emb))
                    else:
                        img_tensor = torch.zeros((0,), dtype=torch.float32)
                else:
                    img_tensor = to_torch(img_emb, dtype=torch.float32)

                if txt_emb is None:
                    if img_emb is not None:
                        txt_tensor = torch.zeros_like(to_torch(img_emb))
                    else:
                        txt_tensor = torch.zeros((0,), dtype=torch.float32)
                else:
                    txt_tensor = to_torch(txt_emb, dtype=torch.float32)
            except Exception as e:
                print(f"Warning: failed to convert embeddings at index {global_index}: {e}. Skipping.")
                global_index += 1
                if pbar_total is not None:
                    pbar.update(1)
                continue

            # Collect a small metadata dictionary
            meta = {}
            for mf in metadata_fields:
                if mf in example and example[mf] is not None:
                    # convert bytes to str if necessary
                    v = example[mf]
                    if isinstance(v, (bytes, bytearray)):
                        try:
                            v = v.decode("utf-8", errors="replace")
                        except Exception:
                            v = str(v)
                    meta[mf] = v

            image_buf.append(img_tensor)
            text_buf.append(txt_tensor)
            id_buf.append(global_index)
            meta_buf.append(meta)

            global_index += 1
            if pbar_total is not None:
                pbar.update(1)
            else:
                pbar.update(1)

            # Flush shard if buffer is full
            if len(id_buf) >= shard_size:
                flush_shard(shard_index)
                shard_index += 1

        # end for

        # final flush
        flush_shard(shard_index)
        pbar.close()
        print(f"Finished. Total examples saved (approx): {saved_count}. Output directory: {out_dir}")

    except KeyboardInterrupt:
        print("Interrupted by user. Flushing current buffers...")
        flush_shard(shard_index)
        pbar.close()

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Export LAION TFDS embeddings to sharded .pth files")
    parser.add_argument("--dataset", type=str, default="laion400m/embeddings", help="TFDS dataset name")
    parser.add_argument("--split", type=str, default="train", help="TFDS split")
    parser.add_argument("--out_dir", type=str, default="laion_embeddings_pth", help="Output directory")
    parser.add_argument("--cache_dir", type=str, default=None, help="TFDS cache directory (optional)")
    parser.add_argument("--shard_size", type=int, default=50_000, help="Number of examples per shard")
    parser.add_argument("--max_examples", type=int, default=None, help="Max examples (for testing)")
    parser.add_argument("--resume", action="store_true", help="Attempt to resume from existing shards")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    export_embeddings(
        dataset_name=args.dataset,
        split=args.split,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        max_examples=args.max_examples,
        resume=args.resume,
        cache_dir=args.cache_dir,
    )
