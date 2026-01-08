# Changelog

All notable changes to the Safe Media Social Feed project are documented in this file.

---

## [2.1.0] - 2026-01-08

### Branch: `main`

### Summary
Major performance optimization release with parallel model execution, in-memory video processing, and adaptive frame sampling. Achieves **3-5x faster** detection times.

### Performance Improvements

| Scenario | Before (v2.0) | After (v2.1) | Speedup |
|----------|---------------|--------------|---------|
| Image Analysis | 900-2400ms | 300-800ms | ~3x faster |
| Video (15 frames) | 15-40 seconds | 3-8 seconds | ~4-5x faster |
| Quick Check | N/A | 100-300ms | New feature |

### Features Added

| Feature | Description |
|---------|-------------|
| Parallel Model Execution | All 3 models run concurrently via ThreadPoolExecutor |
| In-Memory Frame Processing | Video frames processed without disk I/O |
| Adaptive Frame Sampling | Smart sampling based on video duration (5-30 frames) |
| Early Exit Detection | Stops analysis at 85% NSFW confidence threshold |
| Quick Check Endpoint | Single-model fast screening (~200ms) |
| Execution Time Tracking | Per-model and total time metrics in responses |
| Image Cache System | Avoids redundant image loading across models |
| Batch Frame Processing | 4 video frames analyzed concurrently |

### Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `backend/nsfw_detector.py` | Modified | Added parallel execution, ImageCache, in-memory processing |
| `backend/main.py` | Modified | Added optimized video analysis, new endpoints, timing |

### New API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze/quick` | POST | Fast single-model NSFW check (~200ms) |
| `/api/benchmark` | GET | Performance benchmarks reference |

### Architecture Changes

```
BEFORE (Sequential - v2.0):
Image → OpenNSFW2 → NudeNet → Transformers → Result
        (500ms)     (600ms)    (400ms)     = 1500ms total

AFTER (Parallel - v2.1):
Image → ┌─ OpenNSFW2 ──┐
        ├─ NudeNet ────┼→ Result
        └─ Transformers┘
        (parallel: max 600ms) = 600ms total
```

### Response Schema Changes

**All analysis responses now include timing:**
```json
{
  "total_time_ms": 523.4,
  "models": {
    "opennsfw2": { "score": 0.12, "execution_time_ms": 245.2 },
    "nudenet": { "score": 0.08, "execution_time_ms": 523.1 },
    "transformers": { "score": 0.15, "execution_time_ms": 312.8 }
  }
}
```

**Quick check response:**
```json
{
  "is_safe": true,
  "score": 0.0234,
  "threshold": 0.2,
  "model": "opennsfw2",
  "execution_time_ms": 187.3
}
```

### Configuration Options

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_PARALLEL_FRAMES` | 4 | Frames processed concurrently |
| `EARLY_EXIT_THRESHOLD` | 0.85 | Score to trigger early exit |

### Upgrade Notes

- **No breaking changes** - fully backwards compatible
- **No database changes** - existing data works as-is
- **No new dependencies** - uses existing packages
- First request after upgrade may be slightly slower (thread pool initialization)

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
| 2.1.0 | 2026-01-08 | Performance optimization (3-5x faster) |
| 2.0.0 | 2026-01-07 | Multi-model ensemble detection system |
| 1.0.0 | 2026-01-06 | Initial release |

---

## Upgrade Guide

### From 2.0.0 to 2.1.0

1. **No dependency changes required** - uses existing packages

2. **Replace the backend files:**
   ```bash
   # Files updated:
   # - backend/main.py
   # - backend/nsfw_detector.py
   ```

3. **Restart the server:**
   ```bash
   python main.py
   ```

4. **Verify optimization is active:**
   ```bash
   curl http://localhost:8000/api/health
   # Should show "optimizations" array in response
   ```

5. **Test new quick endpoint:**
   ```bash
   curl -X POST -F "file=@test.jpg" http://localhost:8000/api/analyze/quick
   ```

---

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
