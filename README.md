# Safe Media Social Feed

A social media feed application with advanced AI-powered NSFW content detection. Upload images and videos - safe content gets displayed in the public feed, while inappropriate content is automatically blocked.

## Features

- **Image & Video Upload**: Support for common image formats (JPG, PNG, GIF, WebP) and video formats (MP4, MOV, AVI, WebM)
- **Multi-Model AI Detection**: Ensemble of 3 AI models for accurate content moderation
- **Parallel Processing**: Optimized for speed with 3-5x faster detection (v2.1)
- **Detailed Analysis**: Per-model scores, confidence levels, and body part detection
- **Instagram-Style Feed**: Clean, modern UI with a familiar social media layout
- **Real-time Feedback**: Immediate feedback with detailed analysis reports
- **Safety Scores**: Visual safety indicators on each post

## Detection System (v2.1)

### Multi-Model Ensemble

| Model | Weight | Purpose |
|-------|--------|---------|
| OpenNSFW2 | 30% | General NSFW classification |
| NudeNet | 40% | Body part detection with bounding boxes |
| Transformers | 30% | HuggingFace NSFW image classification |

### Performance (v2.1 Optimizations)

| Scenario | Time | Description |
|----------|------|-------------|
| Quick Check | 100-300ms | Single model fast screening |
| Full Image Analysis | 300-800ms | All 3 models in parallel |
| Video (15 frames) | 3-8 seconds | Parallel frame processing |

### Content Categories

- `safe` - No concerning content detected
- `suggestive` - Borderline content
- `partial_nudity` - Non-explicit nudity
- `explicit_nudity` - Explicit content
- `sexual_content` - Sexual material

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Multi-Model Ensemble** - OpenNSFW2 + NudeNet + Transformers
- **SQLite** - Lightweight database for post storage
- **OpenCV** - Video frame extraction
- **ThreadPoolExecutor** - Parallel model execution

### Frontend
- **Next.js 14** - React framework with App Router
- **Tailwind CSS** - Utility-first CSS framework
- **React Dropzone** - Drag-and-drop file uploads
- **Lucide React** - Icon library

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn
- 8GB RAM recommended (models use ~3GB)
- 2GB free disk space (model cache)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   python main.py
   ```

   Or with uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. First startup will download models (~500MB) - this is normal.

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create environment file:
   ```bash
   cp .env.local.example .env.local
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

The application will be available at `http://localhost:3000`

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload media with NSFW detection |
| GET | `/api/posts` | Get all safe posts |
| GET | `/api/posts/{id}` | Get a specific post |
| DELETE | `/api/posts/{id}` | Delete a post |
| GET | `/api/health` | Health check with model status |

### Analysis Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Analyze media without saving |
| POST | `/api/analyze/quick` | Fast single-model check (~200ms) |
| POST | `/api/analyze/video` | Detailed frame-by-frame analysis |
| GET | `/api/models/status` | Check detection model status |
| GET | `/api/benchmark` | Performance benchmarks |

### Example Response

```json
{
  "is_safe": false,
  "final_score": 0.87,
  "confidence": "high",
  "category": "explicit_nudity",
  "threshold_used": 0.2,
  "total_time_ms": 523.4,
  "models": {
    "opennsfw2": { "score": 0.82, "label": "nsfw", "execution_time_ms": 245.2 },
    "nudenet": { "score": 0.91, "label": "explicit", "execution_time_ms": 523.1 },
    "transformers": { "score": 0.85, "label": "nsfw", "execution_time_ms": 312.8 }
  },
  "detections_summary": [
    { "part": "FEMALE_BREAST_EXPOSED", "confidence": 0.94, "location": {...} }
  ],
  "summary": "EXPLICIT NUDITY DETECTED: FEMALE_BREAST_EXPOSED (94%)",
  "recommendation": "BLOCK: High confidence unsafe content detected"
}
```

## How It Works

1. **Upload**: User selects or drags an image/video to upload
2. **Analysis**: Backend analyzes content using 3 AI models in parallel
   - For images: All models run concurrently (~500ms)
   - For videos: Adaptive frame sampling with parallel processing
3. **Ensemble Score**: Weighted combination of all model scores
4. **Decision**: If NSFW score < 20%, content is approved
5. **Result**:
   - Safe content is stored and appears in the feed
   - Unsafe content is rejected with detailed analysis

### Video Processing

```
Video → Extract Frames (adaptive 5-30) → Parallel Analysis → Worst Frame Score
                                              ↓
                              [4 frames processed concurrently]
                                              ↓
                              Early exit if score > 85%
```

## Configuration

### NSFW Threshold

The default NSFW threshold is set to 0.2 (20% - very strict). Adjust in `backend/main.py`:

```python
NSFW_THRESHOLD = 0.2  # Lower = stricter, Higher = more lenient
```

### Model Weights

Customize model influence in `backend/nsfw_detector.py`:

```python
MODEL_WEIGHTS = {
    "opennsfw2": 0.30,    # General classifier
    "nudenet": 0.40,      # Body part detection (highest weight)
    "transformers": 0.30, # HuggingFace model
}
```

### Video Processing

```python
MAX_PARALLEL_FRAMES = 4      # Frames processed concurrently
EARLY_EXIT_THRESHOLD = 0.85  # Stop early if NSFW score exceeds this
```

### Adaptive Frame Sampling

| Video Duration | Frames Sampled |
|---------------|----------------|
| ≤ 5 seconds | Every 0.5s (max 15) |
| 5-30 seconds | Every 2s (max 20) |
| 30-120 seconds | Every 5s (max 25) |
| > 120 seconds | Every 10s (max 30) |

## Project Structure

```
safe-media-social-feed/
├── backend/
│   ├── main.py              # FastAPI application (optimized v2.1)
│   ├── nsfw_detector.py     # Multi-model ensemble detector
│   ├── requirements.txt     # Python dependencies
│   ├── uploads/             # Uploaded media storage
│   └── posts.db             # SQLite database
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── globals.css
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx
│   │   ├── components/
│   │   │   ├── Feed.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── PostCard.tsx
│   │   │   └── UploadModal.tsx
│   │   └── lib/
│   │       └── api.ts
│   ├── package.json
│   └── tailwind.config.ts
├── CHANGES.md               # Version changelog
└── README.md
```

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 2.1.0 | 2026-01-08 | Performance optimization (3-5x faster) |
| 2.0.0 | 2026-01-07 | Multi-model ensemble detection |
| 1.0.0 | 2026-01-06 | Initial release |

See [CHANGES.md](CHANGES.md) for detailed changelog.

## License

MIT
