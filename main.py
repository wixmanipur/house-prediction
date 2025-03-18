from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
import cv2
import io
import base64
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = FastAPI()

# Enable CORS for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your pre-trained YOLO model
model = YOLO('model/yolo11m.pt')

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="templates"), name="static")

@app.get("/")
async def homepage():
    return FileResponse("templates/index.html")

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

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.4),
    iou: float = Form(0.35),
    imgsz: int = Form(640),
    draw_label: bool = Form(True)
):
    """
    Receives an image file along with detection parameters, processes the image using YOLO,
    and returns the number of detections and both the original and annotated images in Base64.
    """
    # Read and decode the uploaded image
    image_bytes = await file.read()
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Validate image decoding
    if img is None:
        return JSONResponse(content={"error": "Invalid image file."}, status_code=400)
    
    try:
        results = model.predict(
            img,
            conf=conf,
            iou=iou,
            imgsz=imgsz
        )
    except Exception as e:
        return JSONResponse(content={"error": f"Model prediction failed: {str(e)}"}, status_code=500)
    
    result = results[0]
    predictions = len(result.boxes) if hasattr(result, 'boxes') else 0
    
    # Annotate image
    annotated_image = annotate_image(
        result,
        font_size=15,
        line_thickness=2,
        txt_color=(0, 0, 0),
        draw_label=draw_label
    )
    
    # Convert annotated image to Base64
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_image_rgb)
    buffer = io.BytesIO()
    annotated_pil.save(buffer, format="PNG")
    buffer.seek(0)
    annotated_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    
    # Convert original image to Base64
    original_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_pil = Image.fromarray(original_image_rgb)
    buffer_original = io.BytesIO()
    original_pil.save(buffer_original, format="PNG")
    buffer_original.seek(0)
    original_b64 = base64.b64encode(buffer_original.read()).decode("utf-8")
    
    return JSONResponse(content={
        "predictions": predictions,
        "annotated_image": f"data:image/png;base64,{annotated_b64}",
        "original_image": f"data:image/png;base64,{original_b64}"
    })
     
  
@app.post("/predict_multiple")
async def predict_multiple(
    files: List[UploadFile] = File(...),
    conf: float = Form(0.4),
    iou: float = Form(0.35),
    imgsz: int = Form(640),
    draw_label: bool = Form(True)
):
    results_list = []
    for file in files:
        image_bytes = await file.read()
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
        # Validate image decoding
        if img is None:
            results_list.append({"filename": file.filename, "error": "Invalid image file."})
            continue
        
        try:
            results = model.predict(
                img,
                conf=conf,
                iou=iou,
                imgsz=imgsz
            )
        except Exception as e:
            results_list.append({"filename": file.filename, "error": f"Model prediction failed: {str(e)}"})
            continue
        
        result = results[0]
        predictions = len(result.boxes) if hasattr(result, 'boxes') else 0
        
        # Annotate image
        annotated_image = annotate_image(
            result,
            font_size=15,
            line_thickness=2,
            txt_color=(0, 0, 0),
            draw_label=draw_label
        )
        
        # Convert annotated image to Base64
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_image_rgb)
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="PNG")
        buffer.seek(0)
        annotated_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        # Convert original image to Base64
        original_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_image_rgb)
        buffer_original = io.BytesIO()
        original_pil.save(buffer_original, format="PNG")
        buffer_original.seek(0)
        original_b64 = base64.b64encode(buffer_original.read()).decode("utf-8")
        
        results_list.append({
            "filename": file.filename,
            "predictions": predictions,
            "annotated_image": f"data:image/png;base64,{annotated_b64}",
            "original_image": f"data:image/png;base64,{original_b64}"
        })
        
    return JSONResponse(content={"results": results_list})

# (Optional) Health-check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})
