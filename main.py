import os
import cv2
import numpy as np
import io
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = Flask(__name__)

# Load the YOLO model
model = YOLO("model/yolo11m.pt")

@app.route("/")
def index():
    return render_template("index.html")

def annotate_image(result, font_size=15, line_thickness=5, txt_color=(255, 255, 255), draw_label=True):
    """
    Annotates the image using YOLO detections. If draw_label is False,
    it draws only bounding boxes without text.
    """
    im_annotated = result.orig_img.copy()
    annotator = Annotator(im_annotated, line_width=line_thickness, font_size=font_size)
    
    boxes = result.boxes
    names = result.names  # e.g., {0: "building"}
    
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0]
            cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
            conf_val = float(box.conf[0]) if hasattr(box, 'conf') else 0
            label = f"{names[cls]} {conf_val:.2f}" if draw_label else ""
            annotator.box_label(xyxy, label, color=colors(cls, True), txt_color=txt_color)
            
    annotated_image = annotator.result()
    return annotated_image


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives an image file along with optional detection parameters, processes the image using YOLO,
    and returns the number of detections and both the original and annotated images in Base64.
    Optional parameters (via form data):
      - conf: Confidence threshold (default 0.4)
      - iou: IoU threshold (default 0.35)
      - imgsz: Image size (default 640)
      - draw_label: "true" or "false" (default "true")
    """
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded."}), 400

    # Get parameters from form data, with defaults
    try:
        conf = float(request.form.get("conf", 0.4))
        iou = float(request.form.get("iou", 0.35))
        imgsz = int(request.form.get("imgsz", 640))
        draw_label_str = request.form.get("draw_label", "true")
        draw_label = draw_label_str.lower() == "true"
    except Exception as e:
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400

    # Read and decode the uploaded image.
    file_bytes = file.read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image file."}), 400

    try:
        results = model.predict(img, conf=conf, iou=iou, imgsz=imgsz)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    result = results[0]
    predictions = len(result.boxes) if hasattr(result, "boxes") else 0

    # Annotate the image.
    annotated_image = annotate_image(result, font_size=15, line_thickness=2, txt_color=(0, 0, 0), draw_label=draw_label)

    # Convert annotated image to Base64.
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_image_rgb)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    buf.seek(0)
    annotated_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # Convert original image to Base64.
    original_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_pil = Image.fromarray(original_image_rgb)
    buf_orig = io.BytesIO()
    original_pil.save(buf_orig, format="PNG")
    buf_orig.seek(0)
    original_b64 = base64.b64encode(buf_orig.read()).decode("utf-8")

    return jsonify({
        "predictions": predictions,
        "annotated_image": f"data:image/png;base64,{annotated_b64}",
        "original_image": f"data:image/png;base64,{original_b64}"
    })

@app.route("/predict_multiple", methods=["POST"])
def predict_multiple():
    """
    Accepts multiple image files via the "files" field, along with optional parameters,
    and returns a list of detection results (with each image's annotated and original images in Base64).
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    try:
        conf = float(request.form.get("conf", 0.4))
        iou = float(request.form.get("iou", 0.35))
        imgsz = int(request.form.get("imgsz", 640))
        draw_label_str = request.form.get("draw_label", "true")
        draw_label = draw_label_str.lower() == "true"
    except Exception as e:
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400

    results_list = []
    for file in files:
        file_bytes = file.read()
        np_img = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            results_list.append({"filename": file.filename, "error": "Invalid image file."})
            continue
        try:
            results = model.predict(img, conf=conf, iou=iou, imgsz=imgsz)
        except Exception as e:
            results_list.append({"filename": file.filename, "error": f"Model prediction failed: {str(e)}"})
            continue
        result = results[0]
        predictions = len(result.boxes) if hasattr(result, "boxes") else 0
        annotated_image = annotate_image(result, font_size=15, line_thickness=2, txt_color=(0, 0, 0), draw_label=draw_label)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_image_rgb)
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        buf.seek(0)
        annotated_b64 = base64.b64encode(buf.read()).decode("utf-8")
        original_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_image_rgb)
        buf_orig = io.BytesIO()
        original_pil.save(buf_orig, format="PNG")
        buf_orig.seek(0)
        original_b64 = base64.b64encode(buf_orig.read()).decode("utf-8")
        results_list.append({
            "filename": file.filename,
            "predictions": predictions,
            "annotated_image": f"data:image/png;base64,{annotated_b64}",
            "original_image": f"data:image/png;base64,{original_b64}"
        })

    return jsonify({"results": results_list})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
