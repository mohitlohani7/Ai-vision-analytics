from ultralytics import YOLO
import cv2
from collections import defaultdict
import os
import numpy as np

# Load model once to speed up live processing
model = YOLO("yolov8n.pt")

def analyze_video(input_path, output_path, progress_bar=None):
    """
    Processes video files for object detection and tracking.
    """
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return None, 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Codec Strategy
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    analytics = {
        "total_frames": 0,
        "unique_ids": defaultdict(set),
        "class_count": defaultdict(int),
        "confidence_sum": defaultdict(float),
        "confidence_count": defaultdict(int)
    }

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Tracking Mode
        results = model.track(frame, persist=True, conf=0.3, verbose=False)
        
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                if box.id is None: continue
                
                cls_name = model.names[int(box.cls[0])]
                track_id = int(box.id[0])
                conf = float(box.conf[0])
                
                analytics["unique_ids"][cls_name].add(track_id)
                analytics["confidence_sum"][cls_name] += conf
                analytics["confidence_count"][cls_name] += 1

        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        frame_count += 1
        analytics["total_frames"] += 1
        
        if progress_bar and total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()
    
    for cls_name, ids in analytics["unique_ids"].items():
        analytics["class_count"][cls_name] = len(ids)

    duration_sec = analytics["total_frames"] / fps
    return analytics, duration_sec

def analyze_image(input_path, output_path):
    """
    Processes a single image for object detection.
    """
    frame = cv2.imread(input_path)
    if frame is None:
        return None, 0

    # Prediction Mode
    results = model.predict(frame, conf=0.3)
    
    analytics = {
        "unique_ids": defaultdict(set),
        "class_count": defaultdict(int),
        "confidence_sum": defaultdict(float),
        "confidence_count": defaultdict(int)
    }

    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        
        analytics["class_count"][cls_name] += 1
        analytics["confidence_sum"][cls_name] += conf
        analytics["confidence_count"][cls_name] += 1

    annotated_frame = results[0].plot()
    cv2.imwrite(output_path, annotated_frame)

    return analytics, 0

def process_live_frame(frame):
    """
    Processes a single frame for Real-Time Live Feed.
    Returns the annotated frame (numpy array).
    """
    # Track mode maintains consistency in live video
    results = model.track(frame, persist=True, conf=0.3, verbose=False)
    annotated_frame = results[0].plot()
    return annotated_frame