"""
FastAPI server for Car Damage Detection using locally trained YOLO11 model
This uses your custom-trained model instead of Roboflow API
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# Windows event loop fix
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize FastAPI app
app = FastAPI(
    title="Car Damage Detection API",
    description="AI-powered car damage detection and cost estimation using YOLO11",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Configuration
# ============================================================================

# Path to your trained model weights
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")  # Change this to your model path
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE", "0.25"))

# Cost table (in USD) - adjust based on your region
COST_TABLE = {
    "damaged-head-light": 200,
    "damaged-hood": 600,
    "damaged-trunk": 400,
    "damaged-window": 300,
    "damaged-windscreen": 350,
    "damaged_bumper": 450,
    "damaged_door": 500,
    "damaged_fender": 350,
    "damaged_mirror_glass": 150,
    "dent-or-scratch": 200,
    "missing_grille": 250,
    # Add more classes as needed
}

# Color map for visualization (BGR format)
COLOR_MAP = {
    "damaged-head-light": (255, 0, 0),
    "damaged-hood": (0, 255, 0),
    "damaged-trunk": (0, 0, 255),
    "damaged-window": (255, 255, 0),
    "damaged-windscreen": (255, 0, 255),
    "damaged_bumper": (0, 255, 255),
    "damaged_door": (128, 0, 128),
    "damaged_fender": (0, 128, 255),
    "damaged_mirror_glass": (255, 128, 0),
    "dent-or-scratch": (128, 255, 0),
    "missing_grille": (0, 128, 128),
}

# ============================================================================
# Global model instance
# ============================================================================

model: Optional[YOLO] = None


def load_model():
    """Load YOLO model on startup"""
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at: {MODEL_PATH}\n"
            f"Please train a model first using train.py or update MODEL_PATH"
        )
    print(f"🔄 Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully! Device: {model.device}")
    print(f"📊 Classes: {model.names}")


@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    load_model()


# ============================================================================
# Helper Functions
# ============================================================================

def bgr_to_hex(bgr: tuple) -> str:
    """Convert BGR color to hex"""
    b, g, r = bgr
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def calculate_severity(mask_area: float, box_area: float) -> str:
    """Calculate damage severity based on area ratio"""
    ratio = mask_area / max(box_area, 1)
    
    if ratio < 0.1:
        return "Minor"
    elif ratio < 0.4:
        return "Moderate"
    else:
        return "Severe"


def calculate_cost(class_name: str, severity: str) -> float:
    """Calculate repair cost based on part and severity"""
    base_cost = COST_TABLE.get(class_name, 100)
    
    multipliers = {
        "Minor": 0.5,
        "Moderate": 1.0,
        "Severe": 1.5
    }
    
    return base_cost * multipliers.get(severity, 1.0)


# ============================================================================
# Response Models
# ============================================================================

class DamageInfo(BaseModel):
    part: str
    severity: str
    cost: float
    confidence: float
    color: str


class InferenceResponse(BaseModel):
    total_cost: float
    damages: list[DamageInfo]
    image: str  # Base64 encoded annotated image


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": MODEL_PATH,
        "classes": list(COST_TABLE.keys()) if model else [],
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check with model status"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(...),
    confidence: float = CONFIDENCE_THRESHOLD,
    return_image: bool = True
):
    """
    Perform inference on uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        confidence: Confidence threshold (0.0 - 1.0)
        return_image: Whether to return annotated image
    
    Returns:
        JSON with detected damages, costs, and optional annotated image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and decode image
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        img_h, img_w = img.shape[:2]
        
        # Run inference
        results = model.predict(
            img,
            conf=confidence,
            imgsz=640,
            verbose=False
        )[0]
        
        # Process results
        total_cost = 0.0
        damages = []
        overlay = img.copy()
        
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            
            for mask, box, cls_id, conf in zip(masks, boxes, classes, confs):
                cls_id = int(cls_id)
                class_name = model.names[cls_id]
                color = COLOR_MAP.get(class_name, (255, 255, 255))
                
                # Resize mask to image dimensions
                mask_resized = cv2.resize(mask, (img_w, img_h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Draw segmentation mask
                overlay[mask_binary == 1] = color
                
                # Calculate severity
                mask_area = np.sum(mask_binary)
                x1, y1, x2, y2 = map(int, box)
                box_area = (x2 - x1) * (y2 - y1)
                severity = calculate_severity(mask_area, box_area)
                
                # Calculate cost
                cost = calculate_cost(class_name, severity)
                total_cost += cost
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name} ({severity})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Append damage info
                damages.append({
                    "part": class_name,
                    "severity": severity,
                    "cost": round(cost, 2),
                    "confidence": round(float(conf), 2),
                    "color": bgr_to_hex(color)
                })
        
        # Blend overlay with original image
        img_annotated = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        
        # Add total cost text
        if damages:
            cost_text = f"Total Est. Cost: ${total_cost:.2f}"
            text_size, _ = cv2.getTextSize(cost_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            text_x = (img_w - text_size[0]) // 2
            text_y = img_h - 30
            
            # Background rectangle for text
            cv2.rectangle(img_annotated, 
                        (text_x - 10, text_y - text_size[1] - 10),
                        (text_x + text_size[0] + 10, text_y + 10),
                        (0, 0, 0), -1)
            cv2.putText(img_annotated, cost_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Encode image to base64
        img_data_uri = ""
        if return_image:
            _, buffer = cv2.imencode(".jpg", img_annotated)
            img_base64 = base64.b64encode(buffer).decode("utf-8")
            img_data_uri = f"data:image/jpeg;base64,{img_base64}"
        
        return {
            "total_cost": round(total_cost, 2),
            "damages": damages,
            "image": img_data_uri
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
async def reload_model(model_path: Optional[str] = None):
    """
    Reload model (useful for updating to a newly trained model)
    
    Args:
        model_path: Optional new model path
    """
    global MODEL_PATH
    
    if model_path:
        MODEL_PATH = model_path
    
    try:
        load_model()
        return {"status": "success", "model_path": MODEL_PATH}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))  # Different port from original app.py
    
    print("=" * 70)
    print("🚀 Car Damage Detection API - Local YOLO Model")
    print("=" * 70)
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📚 Docs: http://{host}:{port}/docs")
    print(f"🔧 Model: {MODEL_PATH}")
    print("=" * 70)
    
    uvicorn.run(
        "inference_api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

