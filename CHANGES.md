# Changelog

All notable changes to the Safe Media Social Feed project are documented in this file.

---

## [2.0.0] - 2026-01-07

### Branch: `main`

### Summary
Complete overhaul of the NSFW detection system with multi-model ensemble architecture for improved accuracy and detailed analysis reporting.

### Features Added

| Feature | Description |
|---------|-------------|
| Multi-Model Ensemble | 3 AI models working together (OpenNSFW2, NudeNet, Transformers) |
| Weighted Scoring | Configurable weights for each model (30%, 40%, 30%) |
| Confidence Levels | Detection confidence: low, medium, high, very_high |
| Content Categories | safe, suggestive, partial_nudity, explicit_nudity, sexual_content |
| Detailed Analysis API | New `/api/analyze` endpoint for testing without saving |
| Video Frame Analysis | New `/api/analyze/video` with frame-by-frame breakdown |
| Model Status API | New `/api/models/status` to check model health |
| Body Part Detection | Bounding boxes for detected explicit content |

### Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `backend/nsfw_detector.py` | **NEW** | Multi-model ensemble detection module |
| `backend/main.py` | Modified | Integrated new detection system, added endpoints |
| `backend/requirements.txt` | Modified | Added new dependencies |

### New Dependencies

```txt
transformers==4.36.0
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
```

### Requirements to Run

1. **Install new dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **First run will download models (~500MB):**
   - Falconsai/nsfw_image_detection (HuggingFace)
   - Models are cached after first download

3. **Minimum System Requirements:**
   - RAM: 8GB recommended (models load ~3GB)
   - Storage: 2GB free space for model cache
   - GPU: Optional (CPU works but slower)

### API Changes

| Endpoint | Status | Description |
|----------|--------|-------------|
| `POST /api/analyze` | **NEW** | Analyze media without saving |
| `POST /api/analyze/video` | **NEW** | Detailed video frame analysis |
| `GET /api/models/status` | **NEW** | Check detection model status |
| `POST /api/upload` | Modified | Now returns detailed `analysis` object |
| `GET /api/health` | Modified | Now includes model status |

### Response Schema Changes

**Upload/Analyze Response now includes:**
```json
{
  "is_safe": false,
  "final_score": 0.87,
  "confidence": "high",
  "category": "explicit_nudity",
  "threshold_used": 0.2,
  "models": {
    "opennsfw2": { "score": 0.82, "label": "nsfw" },
    "nudenet": { "score": 0.91, "label": "explicit", "detections": [...] },
    "transformers": { "score": 0.85, "label": "nsfw" }
  },
  "detections_summary": [...],
  "summary": "EXPLICIT NUDITY DETECTED...",
  "recommendation": "BLOCK: High confidence unsafe content detected"
}
```

### Database Changes

| Column | Change | Description |
|--------|--------|-------------|
| `analysis_json` | **NEW** | Stores full analysis JSON for each post |

**Note:** Existing database will auto-migrate (new column is nullable).

---

## [1.0.0] - 2026-01-06

### Branch: `main`

### Summary
Initial release of Safe Media Social Feed with basic NSFW detection.

### Features

- Image upload with NSFW detection
- Video upload with frame sampling
- Instagram-style feed UI
- OpenNSFW2 detection model
- NudeNet body part detection
- SQLite database for posts
- Basic threshold-based blocking (20%)

### Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Python |
| Frontend | Next.js 14 + React 18 |
| Database | SQLite + SQLAlchemy |
| Detection | OpenNSFW2 + NudeNet |
| Styling | Tailwind CSS |

### Requirements

```txt
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
pillow==10.1.0
numpy==1.26.2
tensorflow==2.15.0
sqlalchemy==2.0.23
opennsfw2==0.10.2
opencv-python-headless==4.8.1.78
nudenet==3.4.2
```

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 2.0.0 | 2026-01-07 | Multi-model ensemble detection system |
| 1.0.0 | 2026-01-06 | Initial release |

---

## Upgrade Guide

### From 1.0.0 to 2.0.0

1. **Backup your database:**
   ```bash
   cp backend/posts.db backend/posts.db.backup
   ```

2. **Update dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Restart the server:**
   ```bash
   python main.py
   ```

4. **First request will be slow** (model download/loading)

5. **Verify models loaded:**
   ```bash
   curl http://localhost:8000/api/models/status
   ```

---

## Notes

- Detection threshold remains at 20% (very strict)
- All existing posts remain compatible
- Frontend requires no changes for basic functionality
- New analysis data available in API responses
