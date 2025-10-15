import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware 
import requests
import cv2
import numpy as np
import base64
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "UonXDOdVT64dROJdbVjs"  
MODEL_URL = "https://outline.roboflow.com/car-damage-detection-tuzuq-ucy9a/2"

cost_table = {
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
    "missing_grille": 250
}

color_map = {
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
    "missing_grille": (0, 128, 128)
}

def bgr_to_hex(bgr):
    b, g, r = bgr
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def run_inference(image_bytes, file, confidence: float):
    """Helper function to call Roboflow with given confidence"""
    files = {"file": (file.filename, image_bytes, file.content_type)}
    params = {
        "api_key": API_KEY,
        "confidence": confidence,
        "overlap": 0.4
    }
    return requests.post(MODEL_URL, params=params, files=files)


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)

        img_h, img_w = img.shape[:2]
        overlay = img.copy()

        response = run_inference(image_bytes, file, confidence=0.7)
        result = response.json()

        # If no predictions → retry with confidence 0.5
        if not result.get("predictions"):
            response = run_inference(image_bytes, file, confidence=0.5)
            result = response.json()

        total_cost = 0
        damages = []

        for pred in result.get("predictions", []):
            cls = pred["class"]
            conf = pred["confidence"]
            points = pred.get("points", [])
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]

            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            color = color_map.get(cls, (255, 255, 255))

            if points:
                pts = np.array([[p["x"], p["y"]] for p in points], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color)

            # Severity calculation
            if points:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 1)
                mask_area = int(np.sum(mask))
                box_area = max((x2 - x1) * (y2 - y1), 1)
                ratio = mask_area / box_area
            else:
                ratio = 0.3

            if ratio < 0.1:
                severity = "Minor"
            elif ratio < 0.4:
                severity = "Moderate"
            else:
                severity = "Severe"

            base_cost = cost_table.get(cls, 100)
            if severity == "Minor":
                cost = base_cost * 0.5
            elif severity == "Moderate":
                cost = base_cost * 1.0
            else:
                cost = base_cost * 1.5

            total_cost += cost

            damages.append({
                "part": cls,
                "severity": severity,
                "cost": round(cost, 2),
                "confidence": round(conf, 2),
                "color": bgr_to_hex(color)
            })

            # Draw label
            label = f"{cls}"
            cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)

        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # Encode image
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        img_data_uri = f"data:image/jpeg;base64,{img_base64}"

        return JSONResponse({
            "total_cost": round(total_cost, 2),
            "damages": damages,
            "image": img_data_uri
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    print(f"Starting app on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, log_level="info")
