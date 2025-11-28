import os
import json
from pathlib import Path
from PIL import Image
import numpy as np

import argparse


def blend_frames(frame_paths, out_size=None, weights=None):
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    if out_size is not None:
        imgs = [img.resize(out_size, Image.LANCZOS) for img in imgs]
    arrs = [np.array(img).astype(np.float32) for img in imgs]
    n = len(arrs)
    if weights is None:
        # emphasize later frames: linear ramp
        weights = np.linspace(0.05, 0.6, n)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()
    blended = sum(w * a for w, a in zip(weights, arrs))
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def normalise_path(base_dir, p):
    # paths in index.json may be relative with backslashes like '..\\processed_samples\\...'
    p = p.replace('\\', os.sep).replace('/', os.sep)
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def process_index(index_json_path, out_images_dir, out_meta_file, resolution=None, max_per_index=None):
    base_dir = os.path.dirname(index_json_path)
    with open(index_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(out_meta_file, 'a', encoding='utf-8') as mf:
        for idx, item in enumerate(data):
            if max_per_index is not None and idx >= max_per_index:
                break
            vid = item.get('video_id') or item.get('video') or item.get('id')
            label = item.get('label') or item.get('full_text') or item.get('text') or ''
            input_frames = item.get('input_frames', [])
            target_frame = item.get('target_frame')
            if not input_frames or target_frame is None:
                continue
            frame_paths = [normalise_path(base_dir, p) for p in input_frames]
            target_path = normalise_path(base_dir, target_frame)
            if any(not os.path.exists(fp) for fp in frame_paths) or not os.path.exists(target_path):
                # skip missing files
                continue

            out_id = f"{Path(index_json_path).parent.name}_{vid}"
            in_out_path = os.path.join(out_images_dir, f"{out_id}_input.png")
            targ_out_path = os.path.join(out_images_dir, f"{out_id}_target.png")

            try:
                blended = blend_frames(frame_paths, out_size=resolution)
                blended.save(in_out_path)
                # copy target to targ_out_path
                Image.open(target_path).convert("RGB").resize(blended.size, Image.LANCZOS).save(targ_out_path)

                record = {
                    "input_image": os.path.abspath(in_out_path),
                    "edited_image": os.path.abspath(targ_out_path),
                    "edit_prompt": label,
                }
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Failed to process {vid}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, default='processed_samples', help='path to processed_samples folder')
    parser.add_argument('--out_dir', type=str, default='data_prepared', help='output dataset dir')
    parser.add_argument('--resolution', type=int, default=None, help='resize resolution for blended images (square)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max_per_index', type=int, default=None, help='Only process the first N entries per index.json (for quick tests).')
    args = parser.parse_args()

    processed_dir = args.processed_dir
    out_dir = args.out_dir
    images_dir = os.path.join(out_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    meta_file = os.path.join(out_dir, 'metadata.jsonl')
    if args.overwrite and os.path.exists(meta_file):
        os.remove(meta_file)

    # iterate over action folders
    for action in os.listdir(processed_dir):
        idx_path = os.path.join(processed_dir, action, 'index.json')
        if os.path.exists(idx_path):
            print(f"Processing {idx_path}")
            process_index(
                idx_path,
                images_dir,
                meta_file,
                resolution=(args.resolution, args.resolution) if args.resolution else None,
                max_per_index=args.max_per_index,
            )

    print(f"Preprocessing finished. Metadata: {meta_file}")
