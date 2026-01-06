# Safe Media Social Feed

A social media feed application with automatic NSFW content detection. Upload images and videos - safe content gets displayed in the public feed, while inappropriate content is automatically blocked.

## Features

- **Image & Video Upload**: Support for common image formats (JPG, PNG, GIF, WebP) and video formats (MP4, MOV, AVI, WebM)
- **AI-Powered NSFW Detection**: Automatic content moderation using OpenNSFW2 deep learning model
- **Instagram-Style Feed**: Clean, modern UI with a familiar social media layout
- **Real-time Feedback**: Immediate feedback on upload success or content blocking
- **Safety Scores**: Visual safety indicators on each post

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **OpenNSFW2** - Pre-trained NSFW detection model
- **SQLite** - Lightweight database for post storage
- **OpenCV** - Video frame extraction for analysis

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

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload media with NSFW detection |
| GET | `/api/posts` | Get all safe posts |
| GET | `/api/posts/{id}` | Get a specific post |
| DELETE | `/api/posts/{id}` | Delete a post |
| GET | `/api/health` | Health check |

## How It Works

1. **Upload**: User selects or drags an image/video to upload
2. **Analysis**: Backend analyzes the content using the OpenNSFW2 model
   - For images: Direct analysis
   - For videos: Samples multiple frames and checks each
3. **Decision**: If NSFW score < 60%, content is approved
4. **Result**:
   - Safe content is stored and appears in the feed
   - Unsafe content is rejected with a warning message

## Configuration

### NSFW Threshold

The default NSFW threshold is set to 0.6 (60%). You can adjust this in `backend/main.py`:

```python
NSFW_THRESHOLD = 0.6  # Adjust as needed
```

### Video Frame Sampling

For videos, the system samples 10 frames by default. Adjust in the `analyze_video_nsfw` function:

```python
def analyze_video_nsfw(video_path: str, sample_frames: int = 10):
```

## Project Structure

```
safe-media-social-feed/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── uploads/             # Uploaded media storage
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
└── README.md
```

## License

MIT
