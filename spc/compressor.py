# spc/compressor.py
"""
Main SPCompressor class.
Implements compress() and decompress() along with helper routine to compute PSNR/SSIM comparisons.
"""
import os
import json
import tempfile
from PIL import Image, ImageFilter, ImageDraw, ImageFont

# Increase PIL image size limit to handle large images (default is ~178M pixels)
# Setting to None disables the limit, or set to a specific value like 500000000
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from .detection import detect_text, detect_faces, detect_saliency
from .utils import merge_rois, quantize_to_nbits, try_save_avif, zip_files, unzip_to_folder
from .reconstruction import reconstruct_from_folder
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import time

class SPCompressor:
    def __init__(self, patch_quality_high=80, patch_quality_low=20, patch_nbits=4, base_quality=20):
        """
        patch_quality_high: quality for high-priority patches (1-100)
        patch_quality_low: quality for base layer (1-100)
        patch_nbits: quantization bits for patches (e.g. 4)
        base_quality: quality for base AVIF/WebP
        """
        self.patch_quality_high = patch_quality_high
        self.patch_quality_low = patch_quality_low
        self.patch_nbits = patch_nbits
        self.base_quality = base_quality

    def compress(self, image_path: str, output_path: str, jpeg_baseline_path: str = None, progress_callback=None):
        """
        Compress image into .spc archive:
        - Detect ROIs (text/faces/saliency)
        - Merge ROIs, extract patches
        - Create base by blurring ROI regions
        - Save base and patches (AVIF/WebP)
        - Write meta.json and zip into output_path (SPC)
        Returns stats dict.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        img = Image.open(image_path).convert("RGB")
        
        # Store image dimensions before any processing
        img_width = img.width
        img_height = img.height
        
        # For very large images, downscale for detection to save memory
        # Detection doesn't need full resolution
        max_detection_size = 4000  # Max dimension for detection
        scale_factor = 1.0
        detection_img = img
        if max(img.width, img.height) > max_detection_size:
            scale_factor = max_detection_size / max(img.width, img.height)
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            detection_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Use uint8 arrays to save memory (instead of default which might be larger)
        arr = np.array(detection_img, dtype=np.uint8)
        
        # 1. detect ROIs on downscaled image
        if progress_callback:
            progress_callback(5, "Detecting regions...", "Detecting text regions")
        text_boxes = detect_text(arr)
        
        if progress_callback:
            progress_callback(10, "Detecting regions...", "Detecting faces")
        face_boxes = detect_faces(arr)
        
        if progress_callback:
            progress_callback(15, "Detecting regions...", "Detecting salient regions")
        sal_boxes = detect_saliency(arr)
        
        # Scale ROI boxes back to original image size
        if scale_factor < 1.0:
            text_boxes = [(int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor)) 
                          for x,y,w,h in text_boxes]
            face_boxes = [(int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor)) 
                          for x,y,w,h in face_boxes]
            sal_boxes = [(int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor)) 
                         for x,y,w,h in sal_boxes]
        
        if progress_callback:
            progress_callback(20, "Processing detections...", "Merging detected regions")
        merged = merge_rois(text_boxes, face_boxes, sal_boxes, iou_threshold=0.2)
        
        # Free detection array memory
        del arr, detection_img

        # Create visualization image with detected regions marked
        if progress_callback:
            progress_callback(25, "Creating visualization...", "Drawing detection boxes")
        visualization_path = None
        if output_path:
            output_dir = os.path.dirname(output_path) or "."
            os.makedirs(output_dir, exist_ok=True)
            vis_filename = os.path.splitext(os.path.basename(output_path))[0] + "_detections.png"
            visualization_path = os.path.join(output_dir, vis_filename)
            
            # Create a copy of the image for visualization
            vis_img = img.copy()
            draw = ImageDraw.Draw(vis_img)
            
            # Draw bounding boxes with different colors for each detection type
            # Faces - Red
            for (x, y, w, h) in face_boxes:
                draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
                # Try to add label
                try:
                    draw.text((x, y - 15), "Face", fill="red")
                except:
                    pass
            
            # Text - Blue
            for (x, y, w, h) in text_boxes:
                draw.rectangle([x, y, x + w, y + h], outline="blue", width=3)
                try:
                    draw.text((x, y - 15), "Text", fill="blue")
                except:
                    pass
            
            # Saliency - Green
            for (x, y, w, h) in sal_boxes:
                draw.rectangle([x, y, x + w, y + h], outline="green", width=3)
                try:
                    draw.text((x, y - 15), "Salient", fill="green")
                except:
                    pass
            
            # Merged patches - Yellow (thicker outline to show final patches)
            for (x, y, w, h) in merged:
                draw.rectangle([x, y, x + w, y + h], outline="yellow", width=2)
            
            # Save visualization
            vis_img.save(visualization_path, format="PNG")
            # Free visualization image memory
            del vis_img, draw

        # 2. create base by blurring ROI areas (we paste blurred ROI regions back to remove details)
        if progress_callback:
            progress_callback(30, "Creating base layer...", "Blurring background regions")
        # Use thumbnail approach for very large images to save memory during blur
        base = img.copy()
        # For very large images, blur on a smaller copy then resize back
        if max(img.width, img.height) > 6000:
            # Blur on downscaled version, then resize back
            blur_scale = 6000 / max(img.width, img.height)
            blur_width = int(img.width * blur_scale)
            blur_height = int(img.height * blur_scale)
            small_base = base.resize((blur_width, blur_height), Image.Resampling.LANCZOS)
            blurred_small = small_base.filter(ImageFilter.GaussianBlur(radius=12))
            blurred = blurred_small.resize((img.width, img.height), Image.Resampling.LANCZOS)
            del small_base, blurred_small
        else:
            blurred = base.filter(ImageFilter.GaussianBlur(radius=12))
        
        for (x,y,w,h) in merged:
            x2 = min(base.width, x+w)
            y2 = min(base.height, y+h)
            # paste blurred patch onto base (covering ROI)
            base.paste(blurred.crop((x,y,x2,y2)), (x,y))
        
        # Free blurred image memory
        del blurred

        # 3. Save base as AVIF/WebP
        if progress_callback:
            progress_callback(40, "Saving base image...", "Compressing base layer")
        # Free base_arr if it was created (it wasn't in the optimized version, but just in case)
        tmpdir = tempfile.mkdtemp(prefix="spc_")
        base_filename = "base.avif"
        base_path = os.path.join(tmpdir, base_filename)
        try_save_avif(base, base_path, quality=self.base_quality)

        # 4. extract patches and quantize
        if progress_callback:
            progress_callback(45, "Extracting patches...", f"Processing {len(merged)} patches")
        patches_meta = []
        patch_files = {}
        for idx, (x,y,w,h) in enumerate(merged):
            if progress_callback and idx % max(1, len(merged) // 10) == 0:
                progress = 45 + int((idx / len(merged)) * 10) if len(merged) > 0 else 45
                progress_callback(progress, "Extracting patches...", f"Processing patch {idx+1}/{len(merged)}")
            x2 = min(img.width, x+w)
            y2 = min(img.height, y+h)
            patch = img.crop((x,y,x2,y2))
            # quantize patch - use uint8 to save memory
            patch_arr = np.array(patch, dtype=np.uint8)
            qarr = quantize_to_nbits(patch_arr, nbits=self.patch_nbits)
            qpatch = Image.fromarray(qarr)
            # save high-quality patch
            patch_filename = f"patch_{idx:03d}.avif"
            patch_path = os.path.join(tmpdir, patch_filename)
            try_save_avif(qpatch, patch_path, quality=self.patch_quality_high)
            patches_meta.append({
                "id": idx,
                "filename": patch_filename,
                "x": int(x), "y": int(y), "w": int(x2-x), "h": int(y2-y),
                "quant": self.patch_nbits
            })
            patch_files[patch_path] = patch_filename
            # Free patch memory immediately
            del patch, patch_arr, qarr, qpatch

        # 5. metadata
        if progress_callback:
            progress_callback(55, "Creating metadata...", "Generating package metadata")
        meta = {
            "width": img.width,
            "height": img.height,
            "patch_count": len(patches_meta),
            "patches": patches_meta,
            "base": {"filename": base_filename, "quality": self.base_quality}
        }
        meta_path = os.path.join(tmpdir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        
        # Also save meta.json to output directory for direct download
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        json_output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(output_path))[0] + "_meta.json")
        with open(json_output_path, "w") as f:
            json.dump(meta, f, indent=2)  # Pretty print for readability

        # 6. prepare file list and zip into SPC and ZIP
        if progress_callback:
            progress_callback(58, "Packaging files...", "Creating SPC and ZIP archives")
        # Ensure all required files are included: base image, patches, and meta.json
        files_and_names = {
            base_path: base_filename,  # Base image (AVIF/WebP)
            meta_path: "meta.json"      # Metadata file
        }
        # Add all patch files
        for patch_path, patch_filename in patch_files.items():
            files_and_names[patch_path] = patch_filename
        
        # Verify all files exist before zipping
        for file_path in files_and_names.keys():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # write zip (both .spc and .zip formats)
        zip_files(output_path, files_and_names)
        # Also create .zip file with same contents (base image, patches, and meta.json)
        zip_output_path = os.path.splitext(output_path)[0] + ".zip"
        zip_files(zip_output_path, files_and_names)

        # 7. optional baseline jpeg for comparison
        # Use the already-loaded img instead of reloading
        if jpeg_baseline_path:
            # produce baseline JPEG with medium quality
            # For very large images, save directly without keeping in memory
            img.save(jpeg_baseline_path, format="JPEG", quality=50, optimize=True)

        # Free large image objects before computing stats (after all operations)
        del img, base
        
        # compute sizes and stats
        orig_size = os.path.getsize(image_path)
        spc_size = os.path.getsize(output_path)
        zip_size = os.path.getsize(zip_output_path) if os.path.exists(zip_output_path) else None
        jpeg_size = os.path.getsize(jpeg_baseline_path) if jpeg_baseline_path and os.path.exists(jpeg_baseline_path) else None
        stats = {
            "original_size_kb": orig_size / 1024,
            "spc_size_kb": spc_size / 1024,
            "zip_size_kb": zip_size / 1024 if zip_size else None,
            "jpeg_size_kb": jpeg_size / 1024 if jpeg_size else None,
            "spc_reduction_pct": 100.0 * (orig_size - spc_size) / orig_size,
            "zip_reduction_pct": 100.0 * (orig_size - zip_size) / orig_size if zip_size else None,
            "jpeg_reduction_pct": 100.0 * (orig_size - jpeg_size) / orig_size if jpeg_size else None,
            "input_url": "/" + image_path.replace("\\","/"),
            "spc_url": "/" + output_path.replace("\\","/"),
            "zip_url": "/" + zip_output_path.replace("\\","/") if 'zip_output_path' in locals() else None,
            "json_url": "/" + json_output_path.replace("\\","/") if 'json_output_path' in locals() else None,
            "visualization_url": "/" + visualization_path.replace("\\","/") if visualization_path and os.path.exists(visualization_path) else None,
            # Detection statistics
            "detection_stats": {
                "faces_detected": len(face_boxes),
                "text_regions_detected": len(text_boxes),
                "saliency_regions_detected": len(sal_boxes),
                "total_patches": len(merged),
                "image_width": img_width,
                "image_height": img_height,
            }
        }
        return stats

    def decompress(self, spc_path: str, output_path: str):
        """
        Decompress .spc or .zip into reconstructed image (PNG).
        Both formats are supported as they use the same ZIP structure.
        """
        tmpdir = tempfile.mkdtemp(prefix="spc_ex_")
        unzip_dir = os.path.join(tmpdir, "extracted")
        unzip_to_folder(spc_path, unzip_dir)
        # reconstruct
        return reconstruct_from_folder(unzip_dir, output_path)

    def compute_final_metrics(self, original_path: str, reconstructed_path: str, jpeg_path: str = None):
        """
        Compute PSNR and SSIM between original and reconstructed/jpegs.
        """
        import numpy as np
        from PIL import Image
        orig = np.array(Image.open(original_path).convert("RGB")).astype(np.float32) / 255.0
        recon = np.array(Image.open(reconstructed_path).convert("RGB")).astype(np.float32) / 255.0
        # align shapes
        h,w,_ = orig.shape
        recon = recon[:h, :w, :]
        psnr_recon = psnr(orig, recon, data_range=1.0)
        ssim_recon = ssim((orig*255).astype('uint8'), (recon*255).astype('uint8'), channel_axis=-1, data_range=255)

        psnr_jpeg = None
        ssim_jpeg = None
        if jpeg_path and os.path.exists(jpeg_path):
            jpeg = np.array(Image.open(jpeg_path).convert("RGB")).astype(np.float32) / 255.0
            jpeg = jpeg[:h, :w, :]
            psnr_jpeg = psnr(orig, jpeg, data_range=1.0)
            ssim_jpeg = ssim((orig*255).astype('uint8'), (jpeg*255).astype('uint8'), channel_axis=-1, data_range=255)

        return {
            "psnr_recon": psnr_recon,
            "ssim_recon": ssim_recon,
            "psnr_jpeg": psnr_jpeg,
            "ssim_jpeg": ssim_jpeg
        }
