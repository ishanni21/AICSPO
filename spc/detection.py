# spc/detection.py
"""
Detection module.
Tries to use easyocr and MTCNN; if not present or for quick runs, falls back to saliency-only detection.
Returns bounding boxes in (x, y, w, h) format.
"""
from typing import List, Tuple
import numpy as np
import cv2

def detect_text(image) -> List[Tuple[int,int,int,int]]:
    """
    Attempt to detect text boxes using easyocr if available.
    Returns list of (x, y, w, h)
    """
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        # image is numpy BGR or RGB? we assume RGB
        results = reader.readtext(image)
        boxes = []
        for (bbox, text, prob) in results:
            # bbox is list of 4 points
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x, y = min(xs), min(ys)
            w, h = max(xs)-x, max(ys)-y
            boxes.append((x,y,w,h))
        return boxes
    except Exception:
        # fallback: no easyocr or failed -> return empty
        return []

def detect_faces(image) -> List[Tuple[int,int,int,int]]:
    """
    Attempt to detect faces using multiple methods:
    1. Try MTCNN (if available)
    2. Fall back to OpenCV Haar Cascade (more reliable)
    3. Fall back to OpenCV DNN face detector (if available)
    Return list of boxes (x,y,w,h).
    """
    boxes = []
    
    # Method 1: Try MTCNN first (more accurate but requires installation)
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
        # Ensure image is in correct format (RGB uint8)
        if image.dtype != np.uint8:
            rgb = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            rgb = image.copy()
        # MTCNN expects RGB format
        detections = detector.detect_faces(rgb)
        for det in detections:
            # Check confidence if available
            confidence = det.get('confidence', 0.5)
            if confidence >= 0.5:  # Only use detections with reasonable confidence
                x, y, w, h = det['box']
                boxes.append((max(0,int(x)), max(0,int(y)), int(w), int(h)))
        if boxes:
            return boxes
    except Exception as e:
        # Silently fall through to next method
        pass
    
    # Method 2: OpenCV Haar Cascade (always available, reliable fallback)
    try:
        # Convert to grayscale for Haar Cascade
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Try to load the cascade file
        # OpenCV includes face cascade, but we need to find it
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_default.xml',
        ]
        
        face_cascade = None
        for path in cascade_paths:
            try:
                face_cascade = cv2.CascadeClassifier(path)
                if not face_cascade.empty():
                    break
            except:
                continue
        
        if face_cascade is not None and not face_cascade.empty():
            # Detect faces with parameters tuned for better detection
            # scaleFactor: how much the image size is reduced at each scale
            # minNeighbors: how many neighbors each candidate rectangle should have
            # minSize: minimum possible object size
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,  # Slightly larger steps for faster detection
                minNeighbors=4,    # Lower threshold to catch more faces
                minSize=(20, 20),  # Smaller minimum size to catch smaller faces
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in faces:
                boxes.append((int(x), int(y), int(w), int(h)))
            if boxes:
                return boxes
    except Exception:
        pass
    
    # Method 3: OpenCV DNN face detector (more modern, if available)
    try:
        # DNN models path (these would need to be downloaded separately)
        # For now, we'll skip this as it requires model files
        pass
    except Exception:
        pass
    
    return boxes

def detect_saliency(image, top_n=5):
    """
    Use OpenCV spectral residual saliency map to return bounding boxes of most salient contours.
    """
    try:
        # image expected as numpy RGB or BGR
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, salMap) = sal.computeSaliency(image)
        salMap = (salMap * 255).astype("uint8")
        # threshold and find contours
        _, thresh = cv2.threshold(salMap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # sort by area, keep top_n
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]
        boxes = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            boxes.append((int(x), int(y), int(w), int(h)))
        return boxes
    except Exception:
        return []
