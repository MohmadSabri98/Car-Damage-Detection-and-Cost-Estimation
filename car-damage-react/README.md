# 🎨 Car Damage Detection - React Frontend

Modern React web application for the Car Damage Detection API.

## 🚀 Features

- 📸 **Image Upload** - Drag & drop or click to upload car images
- 🔍 **Real-time Detection** - Instant damage detection results
- 💰 **Cost Breakdown** - Detailed repair cost estimation
- 🎨 **Visual Results** - Annotated images with damage segmentation
- 📊 **Damage Summary** - List of detected damages with severity
- 🎭 **Modern UI** - Clean, responsive interface

## 📋 Prerequisites

- Node.js 16+ and npm
- Backend API running (see main [README](../README.md))

## 🛠️ Installation

```bash
# Navigate to frontend directory
cd car-damage-react

# Install dependencies
npm install
```

## 🏃 Running the App

### Development Mode

```bash
npm run dev
```

The app will start at **http://localhost:5173**

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## ⚙️ Configuration

### API Endpoint

Update the API endpoint in your components if needed:

```javascript
// Default: http://localhost:8000/infer (for Roboflow API)
// Or: http://localhost:8001/infer (for local YOLO model)

const API_URL = "http://localhost:8001/infer";
```

### CORS Configuration

Make sure your backend allows requests from the frontend:

In `app.py` or `inference_api.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📁 Project Structure

```
car-damage-react/
├── public/
│   └── vite.svg
├── src/
│   ├── component/
│   │   ├── Header/
│   │   │   ├── Header.jsx
│   │   │   └── Header.css
│   │   └── ImageUpload/
│   │       ├── ImageUpload.jsx       # Main upload component
│   │       ├── DamageCard.jsx        # Individual damage display
│   │       ├── DamageCard.css
│   │       ├── SummarySection.jsx    # Cost summary
│   │       └── SummarySection.css
│   ├── App.jsx                       # Main app component
│   ├── App.css
│   ├── main.jsx                      # Entry point
│   └── index.css                     # Global styles
├── index.html
├── package.json
├── vite.config.js
└── README.md                         # This file
```

## 🎨 Components

### Header
Navigation bar with branding

### ImageUpload
Main component for uploading images and displaying results
- File input with drag & drop
- Image preview
- API integration
- Results display

### DamageCard
Individual damage information card
- Part name
- Severity level
- Cost
- Confidence score
- Color indicator

### SummarySection
Overall cost summary
- Total estimated cost
- Number of damages detected

## 🔧 Tech Stack

- **React 18** - UI library
- **Vite** - Build tool & dev server
- **CSS3** - Styling
- **Fetch API** - HTTP requests

## 🌐 Deployment

### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

### Deploy to Netlify

```bash
npm run build
# Upload dist/ folder to Netlify
```

### Deploy with Docker

```dockerfile
# Dockerfile for React app
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
RUN npm install -g serve
CMD ["serve", "-s", "dist", "-l", "3000"]
```

## 🐛 Troubleshooting

**API Connection Error**
```
Error: Failed to fetch
Solution: Check if backend is running on correct port
```

**CORS Error**
```
Solution: Update CORS settings in backend
```

**Build Fails**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## 📱 Mobile Support

The app is fully responsive and works on:
- 📱 Mobile phones
- 📱 Tablets
- 💻 Desktop browsers

## 🔄 Integration with Backend

The frontend connects to either:

1. **Roboflow API** (app.py):
   - Port: 8000
   - Endpoint: `/infer`

2. **Local YOLO Model** (inference_api.py):
   - Port: 8001
   - Endpoint: `/infer`

Update the API URL in your component to match your backend.

## 🚀 Future Enhancements

- [ ] Camera integration for direct photo capture
- [ ] Image history/gallery
- [ ] Export report to PDF
- [ ] Multi-language support
- [ ] Dark mode
- [ ] Comparison with insurance quotes
- [ ] User authentication

## 📄 License

Part of the Car Damage Detection project - MIT License

---

**For complete project documentation, see the main [README](../README.md)**
