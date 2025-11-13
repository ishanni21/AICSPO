# app/routes.py
import os
import tempfile
import uuid
import zipfile
import threading
from flask import Blueprint, request, render_template, send_file, current_app, url_for, session, jsonify
from spc.compressor import SPCompressor

spc_routes = Blueprint("spc_routes", __name__, template_folder="templates", static_folder="static")
compressor = SPCompressor()

# Store progress for each task
progress_store = {}

def get_outputs_dir():
    """Get the outputs directory path, creating it if needed."""
    static_folder = current_app.static_folder
    # Flask's static_folder might be relative or absolute
    if not os.path.isabs(static_folder):
        # Resolve relative to the app root (where run.py is located)
        # routes.py is in app/, so go up one level to get app root
        app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        static_folder = os.path.join(app_root, static_folder)
    outputs_dir = os.path.join(static_folder, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    return outputs_dir

@spc_routes.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@spc_routes.route("/progress/<task_id>")
def get_progress(task_id):
    """Get progress for a task."""
    progress = progress_store.get(task_id, {"percent": 0, "status": "Starting...", "step": ""})
    return jsonify(progress)

@spc_routes.route("/result/<task_id>")
def get_result(task_id):
    """Get result page for a completed task."""
    progress = progress_store.get(task_id, {})
    
    if not progress:
        return "Task not found. The compression may have completed. Please try uploading again.", 404
    
    if progress.get("step") == "error":
        return f"Error: {progress.get('status')}", 500
    
    if progress.get("percent", 0) < 100:
        # If still processing, redirect back to home with message
        return f"Task still processing ({progress.get('percent', 0)}%). Please wait...", 202
    
    stats = progress.get("stats")
    metrics = progress.get("metrics")
    uid = progress.get("uid")
    
    if not stats or not metrics or not uid:
        # Results not ready yet, wait a moment
        import time
        time.sleep(0.5)  # Wait half a second
        progress = progress_store.get(task_id, {})
        stats = progress.get("stats")
        metrics = progress.get("metrics")
        uid = progress.get("uid")
        
        if not stats or not metrics or not uid:
            return f"Results not available yet. Progress: {progress.get('percent', 0)}%. Please wait a moment and refresh.", 202
    
    outputs_dir = get_outputs_dir()
    spc_out = os.path.join(outputs_dir, f"output_{uid}.spc")
    reconstructed = os.path.join(outputs_dir, f"reconstructed_{uid}.png")
    jpeg_baseline = os.path.join(outputs_dir, f"baseline_{uid}.jpg")
    in_path = os.path.join(outputs_dir, f"input_{uid}.png")
    
    # Build download links
    download_spc = "/static/outputs/" + os.path.basename(spc_out)
    download_recon = "/static/outputs/" + os.path.basename(reconstructed)
    download_jpeg = "/static/outputs/" + os.path.basename(jpeg_baseline)
    download_original = "/static/outputs/" + os.path.basename(in_path)
    download_zip = "/static/outputs/" + f"output_{uid}.zip"
    
    download_json = stats.get("json_url", None)
    if download_json:
        json_filename = os.path.basename(download_json.lstrip("/"))
        download_json = "/static/outputs/" + json_filename
    
    visualization_url = stats.get("visualization_url", None)
    if visualization_url:
        vis_filename = os.path.basename(visualization_url.lstrip("/"))
        visualization_url = "/static/outputs/" + vis_filename

    return render_template("result.html",
                           stats=stats,
                           metrics=metrics,
                           spc_url=download_spc,
                           recon_url=download_recon,
                           jpeg_url=download_jpeg,
                           original_url=download_original,
                           zip_url=download_zip,
                           json_url=download_json,
                           visualization_url=visualization_url)

@spc_routes.route("/compress", methods=["POST"])
def compress_route():
    uploaded = request.files.get("image")
    if not uploaded:
        return "No image uploaded", 400

    # Get outputs directory
    outputs_dir = get_outputs_dir()

    # unique names to avoid clashing
    uid = uuid.uuid4().hex[:8]
    task_id = uid
    in_path = os.path.join(outputs_dir, f"input_{uid}.png")
    uploaded.save(in_path)
    
    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.content_type and 'application/json' in request.content_type

    spc_out = os.path.join(outputs_dir, f"output_{uid}.spc")
    reconstructed = os.path.join(outputs_dir, f"reconstructed_{uid}.png")
    jpeg_baseline = os.path.join(outputs_dir, f"baseline_{uid}.jpg")

    # Initialize progress immediately (before compression starts)
    progress_store[task_id] = {"percent": 0, "status": "Initializing...", "step": "Preparing image for compression..."}

    def update_progress(percent, status, step=""):
        # Merge with existing progress to preserve stats/metrics/uid
        if task_id in progress_store:
            progress_store[task_id].update({"percent": percent, "status": status, "step": step})
        else:
            progress_store[task_id] = {"percent": percent, "status": status, "step": step}

    def run_compression():
        try:
            stats = compressor.compress(in_path, spc_out, jpeg_baseline_path=jpeg_baseline, progress_callback=update_progress)
            update_progress(60, "Decompressing...", "Reconstructing image from SPC package")
            # Run decompression to produce reconstructed image
            compressor.decompress(spc_out, reconstructed)
            update_progress(80, "Computing metrics...", "Calculating quality metrics")
            # After decompress, compute metrics comparing reconstructed with original
            metrics = compressor.compute_final_metrics(original_path=in_path,
                                                       reconstructed_path=reconstructed,
                                                       jpeg_path=jpeg_baseline)
            update_progress(90, "Finalizing...", "Preparing results")

            # Create a zip file containing all output files
            update_progress(95, "Creating package...", "Packaging output files")
            zip_out = os.path.join(outputs_dir, f"output_{uid}.zip")
            json_file = os.path.join(outputs_dir, f"output_{uid}_meta.json")
            with zipfile.ZipFile(zip_out, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all output files to the zip
                zipf.write(in_path, os.path.basename(in_path))
                zipf.write(spc_out, os.path.basename(spc_out))
                zipf.write(reconstructed, os.path.basename(reconstructed))
                zipf.write(jpeg_baseline, os.path.basename(jpeg_baseline))
                # Add JSON metadata file if it exists
                if os.path.exists(json_file):
                    zipf.write(json_file, os.path.basename(json_file))

            # Store results for later retrieval (update without overwriting)
            current_progress = progress_store.get(task_id, {})
            current_progress["stats"] = stats
            current_progress["metrics"] = metrics
            current_progress["uid"] = uid
            current_progress["percent"] = 100
            current_progress["status"] = "Complete!"
            current_progress["step"] = "Compression finished successfully"
            progress_store[task_id] = current_progress
        except Exception as e:
            progress_store[task_id] = {"percent": 0, "status": f"Error: {str(e)}", "step": "error"}

    # If AJAX request, start compression in background and return task_id immediately
    if is_ajax:
        # Update progress to show we're starting
        progress_store[task_id].update({
            "percent": 1,
            "status": "Starting compression...",
            "step": "Loading image and initializing..."
        })
        thread = threading.Thread(target=run_compression, daemon=True)
        thread.start()
        return jsonify({"task_id": task_id})
    
    # For non-AJAX requests, run synchronously and return results directly
    run_compression()
    
    if progress_store[task_id].get("step") == "error":
        return f"Compression failed: {progress_store[task_id].get('status')}", 500
    
    stats = progress_store[task_id].get("stats")
    metrics = progress_store[task_id].get("metrics")
    
    if not stats or not metrics:
        return "Compression completed but results not available. Please try again.", 500
    
    # Build download links (served from static)
    # Flask serves static files from /static/ URL prefix
    download_spc = "/static/outputs/" + os.path.basename(spc_out)
    download_recon = "/static/outputs/" + os.path.basename(reconstructed)
    download_jpeg = "/static/outputs/" + os.path.basename(jpeg_baseline)
    download_original = "/static/outputs/" + os.path.basename(in_path)
    download_zip = "/static/outputs/" + f"output_{uid}.zip"
    # Get JSON URL from stats (created during compression)
    download_json = stats.get("json_url", None) if stats else None
    if download_json:
        # Ensure it uses the static path format
        json_filename = os.path.basename(download_json.lstrip("/"))
        download_json = "/static/outputs/" + json_filename
    
    # Get visualization URL from stats
    visualization_url = stats.get("visualization_url", None) if stats else None
    if visualization_url:
        # Ensure it uses the static path format
        vis_filename = os.path.basename(visualization_url.lstrip("/"))
        visualization_url = "/static/outputs/" + vis_filename

    return render_template("result.html",
                           stats=stats,
                           metrics=metrics,
                           spc_url=download_spc,
                           recon_url=download_recon,
                           jpeg_url=download_jpeg,
                           original_url=download_original,
                           zip_url=download_zip,
                           json_url=download_json,
                           visualization_url=visualization_url)

@spc_routes.route("/decompress", methods=["POST"])
def decompress_route():
    uploaded = request.files.get("spc")
    if not uploaded:
        return "No spc uploaded", 400

    # Get outputs directory
    outputs_dir = get_outputs_dir()

    uid = uuid.uuid4().hex[:8]
    in_spc = os.path.join(outputs_dir, f"uploaded_{uid}.spc")
    out_image = os.path.join(outputs_dir, f"decompressed_{uid}.png")
    uploaded.save(in_spc)
    compressor.decompress(in_spc, out_image)
    return send_file(out_image, as_attachment=True, download_name=f"decompressed_{uid}.png")

@spc_routes.route("/view", methods=["POST"])
def view_spc_route():
    """View contents of an SPC package file."""
    uploaded = request.files.get("spc")
    if not uploaded:
        return "No SPC file uploaded", 400

    # Get outputs directory
    outputs_dir = get_outputs_dir()
    uid = uuid.uuid4().hex[:8]
    in_spc = os.path.join(outputs_dir, f"view_{uid}.spc")
    uploaded.save(in_spc)

    # Extract and list contents
    contents = []
    meta_data = None
    meta_json_str = None
    try:
        with zipfile.ZipFile(in_spc, 'r') as z:
            for info in z.namelist():
                file_size = None
                compressed_size = None
                # Get file size from ZipInfo
                try:
                    file_info = z.getinfo(info)
                    file_size = file_info.file_size
                    compressed_size = file_info.compress_size
                except:
                    pass
                
                contents.append({
                    "name": info,
                    "size": file_size,
                    "compressed_size": compressed_size
                })
                
                # Try to read and parse meta.json
                if info == "meta.json":
                    try:
                        import json
                        meta_data = json.loads(z.read(info).decode('utf-8'))
                        meta_json_str = json.dumps(meta_data, indent=2)
                    except:
                        pass
        
        # Clean up uploaded file
        os.remove(in_spc)
        
        return render_template("view_spc.html", 
                             contents=contents, 
                             meta_data=meta_data,
                             meta_json_str=meta_json_str,
                             filename=uploaded.filename)
    except zipfile.BadZipFile:
        os.remove(in_spc)
        return "Invalid SPC/ZIP file", 400
    except Exception as e:
        if os.path.exists(in_spc):
            os.remove(in_spc)
        return f"Error reading SPC file: {str(e)}", 500

@spc_routes.route("/view/<path:filename>")
def view_spc_by_path(filename):
    """View contents of an SPC package from the outputs directory."""
    outputs_dir = get_outputs_dir()
    spc_path = os.path.join(outputs_dir, filename)
    
    if not os.path.exists(spc_path):
        return "SPC file not found", 404
    
    if not filename.endswith(('.spc', '.zip')):
        return "Invalid file type", 400
    
    # Extract and list contents
    contents = []
    meta_data = None
    meta_json_str = None
    try:
        with zipfile.ZipFile(spc_path, 'r') as z:
            for info in z.namelist():
                file_size = None
                compressed_size = None
                # Get file size from ZipInfo
                try:
                    file_info = z.getinfo(info)
                    file_size = file_info.file_size
                    compressed_size = file_info.compress_size
                except:
                    pass
                
                contents.append({
                    "name": info,
                    "size": file_size,
                    "compressed_size": compressed_size
                })
                
                # Try to read and parse meta.json
                if info == "meta.json":
                    try:
                        import json
                        meta_data = json.loads(z.read(info).decode('utf-8'))
                        meta_json_str = json.dumps(meta_data, indent=2)
                    except:
                        pass
        
        return render_template("view_spc.html", 
                             contents=contents, 
                             meta_data=meta_data,
                             meta_json_str=meta_json_str,
                             filename=filename)
    except zipfile.BadZipFile:
        return "Invalid SPC/ZIP file", 400
    except Exception as e:
        return f"Error reading SPC file: {str(e)}", 500
