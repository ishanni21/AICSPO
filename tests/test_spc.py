# tests/test_spc.py
"""
Basic pipeline test: compress + decompress a sample image in tests/sample.jpg
Make sure tests/sample.jpg exists (you can put a small image here).
"""
import os
from spc.compressor import SPCompressor

def test_spc_pipeline():
    inp = "tests/sample.jpg"
    os.makedirs("tests", exist_ok=True)
    if not os.path.exists(inp):
        # create a sample image if missing
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (512, 512), color=(230,230,240))
        d = ImageDraw.Draw(img)
        d.text((30, 200), "Sample Text", fill=(10,10,10))
        img.save(inp, "JPEG", quality=90)

    comp = SPCompressor()
    out_spc = "tests/sample.spc"
    out_recon = "tests/sample_recon.png"
    comp.compress(inp, out_spc, jpeg_baseline_path="tests/sample_baseline.jpg")
    assert os.path.exists(out_spc)
    comp.decompress(out_spc, out_recon)
    assert os.path.exists(out_recon)
    print("SPC pipeline test OK.")

if __name__ == "__main__":
    test_spc_pipeline()
