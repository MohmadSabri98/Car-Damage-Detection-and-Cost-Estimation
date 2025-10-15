# Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

## Configuration Options

### Model Configuration
- `MODEL_PATH`: Path to trained YOLO weights file (default: `best.pt`)
- `CONFIDENCE`: Confidence threshold for detections (default: `0.25`)

### Server Configuration
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8001`)

### GPU Configuration
- `CUDA_VISIBLE_DEVICES`: GPU device ID (e.g., `0`, `1`, or `0,1` for multiple GPUs)

## Example Configurations

### Local Development
```env
MODEL_PATH=runs/train/car_damage_seg/weights/best.pt
CONFIDENCE=0.25
HOST=127.0.0.1
PORT=8001
```

### Production
```env
MODEL_PATH=/app/models/best.pt
CONFIDENCE=0.3
HOST=0.0.0.0
PORT=8001
CUDA_VISIBLE_DEVICES=0
```

