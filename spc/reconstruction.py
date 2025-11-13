# spc/reconstruction.py
from PIL import Image
import json
import os

# Increase PIL image size limit to handle large images
Image.MAX_IMAGE_PIXELS = None

def reconstruct_from_folder(extracted_folder: str, output_path: str):
    """
    Reads meta.json and base + patches from extracted_folder and reconstructs image.
    """
    meta_path = os.path.join(extracted_folder, "meta.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    width = meta['width']
    height = meta['height']
    base_filename = meta['base']['filename']
    base_path = os.path.join(extracted_folder, base_filename)
    base_img = Image.open(base_path).convert("RGB")
    canvas = Image.new("RGB", (width, height))
    canvas.paste(base_img, (0,0))

    # paste patches: low-priority already in base; but overlay high-priority patches to refine
    patches = meta.get('patches', [])
    for p in patches:
        patch_path = os.path.join(extracted_folder, p['filename'])
        if not os.path.exists(patch_path):
            continue
        patch_img = Image.open(patch_path).convert("RGB")
        x, y = int(p['x']), int(p['y'])
        canvas.paste(patch_img, (x, y))
    canvas.save(output_path)
    return output_path
