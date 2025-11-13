# spc/utils.py
"""
Utility functions: ROI merging, quantization helpers, AVIF/WebP save fallback, zip helpers, IoU.
"""
import json
import os
import zipfile
from typing import List, Tuple
import numpy as np
from PIL import Image
import imageio
import io

# Increase PIL image size limit to handle large images
Image.MAX_IMAGE_PIXELS = None

def iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) of two boxes.
    Boxes are in format (x, y, w, h).
    Returns IoU value between 0 and 1.
    """
    # boxes as (x,y,w,h)
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    
    # Calculate intersection
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Calculate union
    areaA = aw * ah
    areaB = bw * bh
    union = areaA + areaB - inter_area
    
    if union == 0:
        return 0.0
    return inter_area / union

def merge_rois(*box_lists, iou_threshold=0.2, distance_threshold=None):
    """
    Merge overlapping boxes from multiple lists using an improved greedy agglomerative merge.
    
    Args:
        *box_lists: Variable number of lists of boxes, each box as (x, y, w, h)
        iou_threshold: IoU threshold for merging (default 0.2, lower = more aggressive merging)
        distance_threshold: Optional distance threshold for nearby boxes (as fraction of average box size)
    
    Returns:
        Merged list of boxes (x, y, w, h)
    """
    boxes = []
    for bl in box_lists:
        boxes.extend(bl or [])
    if not boxes:
        return []
    
    # Convert to integers and filter invalid boxes
    valid_boxes = []
    for b in boxes:
        try:
            x, y, w, h = map(int, b)
            if w > 0 and h > 0:  # Only keep boxes with positive dimensions
                valid_boxes.append((x, y, w, h))
        except (ValueError, TypeError):
            continue
    
    if not valid_boxes:
        return []
    
    boxes = valid_boxes
    
    # Calculate average box size for distance threshold if needed
    if distance_threshold is None:
        avg_size = sum(w * h for _, _, w, h in boxes) / len(boxes)
        avg_dim = (avg_size ** 0.5) if avg_size > 0 else 100
        distance_threshold = avg_dim * 0.3  # 30% of average dimension
    
    merged = []
    used = [False] * len(boxes)
    
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        
        x, y, w, h = b
        cur = [x, y, w, h]
        used[i] = True
        changed = True
        
        # Iteratively merge with overlapping boxes
        while changed:
            changed = False
            for j, bj in enumerate(boxes):
                if used[j]:
                    continue
                
                bj_x, bj_y, bj_w, bj_h = bj
                
                # Check IoU overlap
                iou_val = iou(tuple(cur), bj)
                if iou_val >= iou_threshold:
                    # Merge: create bounding box that encloses both
                    x1 = min(cur[0], bj_x)
                    y1 = min(cur[1], bj_y)
                    x2 = max(cur[0] + cur[2], bj_x + bj_w)
                    y2 = max(cur[1] + cur[3], bj_y + bj_h)
                    cur = [x1, y1, x2 - x1, y2 - y1]
                    used[j] = True
                    changed = True
                else:
                    # Check if boxes are very close (for nearby but non-overlapping boxes)
                    cur_center_x = cur[0] + cur[2] / 2
                    cur_center_y = cur[1] + cur[3] / 2
                    bj_center_x = bj_x + bj_w / 2
                    bj_center_y = bj_y + bj_h / 2
                    
                    distance = ((cur_center_x - bj_center_x) ** 2 + (cur_center_y - bj_center_y) ** 2) ** 0.5
                    
                    if distance < distance_threshold and iou_val > 0:
                        # Merge nearby boxes with some overlap
                        x1 = min(cur[0], bj_x)
                        y1 = min(cur[1], bj_y)
                        x2 = max(cur[0] + cur[2], bj_x + bj_w)
                        y2 = max(cur[1] + cur[3], bj_y + bj_h)
                        cur = [x1, y1, x2 - x1, y2 - y1]
                        used[j] = True
                        changed = True
        
        merged.append(tuple(cur))
    
    return merged

def quantize_to_nbits(arr: np.ndarray, nbits=4):
    """
    Simple uniform quantization of 8-bit values to nbits.
    nbits=4 => 16 levels.
    """
    if nbits >= 8:
        return arr
    levels = 2**nbits
    step = 256 // levels
    quant = (arr // step) * step
    return quant.astype(np.uint8)

def try_save_avif(pil_img: Image.Image, path: str, quality=20):
    """
    Try to save PIL image as AVIF using imageio; if not possible fallback to WEBP.
    pil_img: PIL Image in RGB
    """
    arr = np.array(pil_img)
    try:
        # imageio will select avif plugin if available
        imageio.imwrite(path, arr, format='AVIF', quality_mode='quality', quality=quality)
    except Exception:
        # fallback to webp
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.webp', '.avif'):
            path = os.path.splitext(path)[0] + '.webp'
        pil_img.save(path, format='WEBP', quality=max(10, min(95, quality)))
    return path

def zip_files(output_spc: str, files_and_names: dict):
    """
    Create a zip (.spc) from a dict {filepath: arcname}
    """
    with zipfile.ZipFile(output_spc, 'w', zipfile.ZIP_DEFLATED) as z:
        for fpath, arcname in files_and_names.items():
            z.write(fpath, arcname)

def unzip_to_folder(spc_path: str, target_folder: str):
    import zipfile
    os.makedirs(target_folder, exist_ok=True)
    with zipfile.ZipFile(spc_path, 'r') as z:
        z.extractall(target_folder)
