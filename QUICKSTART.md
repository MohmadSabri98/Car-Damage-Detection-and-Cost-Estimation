# ⚡ Quick Start Guide

Get up and running with Car Damage Detection in 5 minutes!

## 🎯 Prerequisites

- Python 3.8+
- pip installed
- (Optional) GPU for faster inference

## 📦 Installation (3 steps)

### Step 1: Clone & Setup

```bash
git clone https://github.com/yourusername/car-damage-detection.git
cd car-damage-detection

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get Model Weights

**Option A: Use Pre-trained Model**
- Download `best.pt` from releases
- Place it in the project root

**Option B: Train Your Own**
```bash
python train.py --epochs 50 --batch 16
```

### Step 3: Start Server

```bash
python inference_api.py
```

Server starts at: **http://localhost:8001** 🚀

## 🧪 Test It!

### Option 1: Web Interface

Open browser: **http://localhost:8001/docs**

Click **"Try it out"** → Upload image → **Execute**

### Option 2: Command Line

```bash
# Test health
curl http://localhost:8001/health

# Test inference
curl -X POST "http://localhost:8001/infer" \
  -F "file=@your_car_image.jpg"
```

### Option 3: Python Script

```python
import requests

url = "http://localhost:8001/infer"
files = {"file": open("car_image.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Total Cost: ${result['total_cost']}")
for damage in result['damages']:
    print(f"- {damage['part']}: ${damage['cost']}")
```

## 🎉 You're Done!

Your API is now running and ready to detect car damage!

### Next Steps

- 🎨 **Try the React Frontend**: `cd car-damage-react && npm install && npm run dev`
- 📖 **Read Full Docs**: [README.md](README.md)
- 🔌 **API Documentation**: http://localhost:8001/docs
- 💻 **Frontend Docs**: [car-damage-react/README.md](car-damage-react/README.md)

## 🐛 Issues?

**Model not found?**
```bash
export MODEL_PATH=/path/to/your/best.pt
```

**Port already in use?**
```bash
export PORT=8002
python inference_api.py
```

**Need help?** Open an issue on GitHub!

