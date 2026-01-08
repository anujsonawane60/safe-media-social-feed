"""
Safe Media Social Feed API (Optimized)

A robust content moderation system using multi-model ensemble
NSFW detection for accurate nudity and sexual content filtering.

OPTIMIZATIONS:
- Parallel model execution for images
- In-memory video frame processing (no temp file I/O)
- Adaptive frame sampling based on video length
- Parallel frame batch processing
- Aggressive early exit on high-confidence detections
- Execution time tracking for performance monitoring
"""

import os
import uuid
import shutil
import logging
import time
import asyncio
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import cv2

from nsfw_detector import get_detector, EnsembleResult, ContentCategory, ConfidenceLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Safe Media Social Feed API",
    description="Robust NSFW detection system with multi-model ensemble (Optimized)",
    version="2.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./posts.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_DIR = UPLOAD_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# NSFW threshold - configurable
NSFW_THRESHOLD = 0.2

# Video processing configuration
MAX_PARALLEL_FRAMES = 4  # Number of frames to process in parallel
EARLY_EXIT_THRESHOLD = 0.85  # Score threshold for early exit


# ============== Pydantic Response Models ==============

class ModelDetailResponse(BaseModel):
    """Individual model analysis result."""
    score: float
    label: str
    is_available: bool
    detections: list = []
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


class DetectionDetail(BaseModel):
    """Detected body part details."""
    part: str
    confidence: float
    location: Optional[dict] = None


class AnalysisResponse(BaseModel):
    """Detailed NSFW analysis response."""
    is_safe: bool
    final_score: float
    confidence: str
    category: str
    threshold_used: float
    models: dict  # model_name -> ModelDetailResponse
    detections_summary: List[DetectionDetail]
    summary: str
    recommendation: str
    total_time_ms: Optional[float] = None


class PostResponse(BaseModel):
    """Post data response."""
    id: str
    filename: str
    original_filename: str
    media_type: str
    caption: Optional[str]
    nsfw_score: float
    is_safe: bool
    created_at: datetime
    file_url: str
    analysis_details: Optional[dict] = None

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Upload endpoint response."""
    success: bool
    message: str
    post: Optional[PostResponse] = None
    analysis: Optional[AnalysisResponse] = None


class VideoAnalysisResponse(BaseModel):
    """Video analysis with frame-by-frame details."""
    is_safe: bool
    final_score: float
    confidence: str
    category: str
    frames_analyzed: int
    worst_frame: Optional[int] = None
    frame_scores: List[dict]
    summary: str
    recommendation: str
    total_time_ms: Optional[float] = None


class PerformanceMetrics(BaseModel):
    """Performance metrics for analysis."""
    total_time_ms: float
    model_times: dict
    frames_processed: Optional[int] = None
    early_exit: bool = False


# ============== Database Model ==============

class Post(Base):
    __tablename__ = "posts"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False)
    caption = Column(String, nullable=True)
    nsfw_score = Column(Float, nullable=False)
    is_safe = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String, nullable=False)
    analysis_json = Column(Text, nullable=True)  # Store full analysis as JSON


Base.metadata.create_all(bind=engine)


# ============== Helper Functions ==============

def get_media_type(filename: str) -> str:
    """Determine if the file is an image or video based on extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}

    ext = Path(filename).suffix.lower()
    if ext in image_extensions:
        return 'image'
    elif ext in video_extensions:
        return 'video'
    else:
        return 'unknown'


def ensemble_to_response(result: EnsembleResult) -> AnalysisResponse:
    """Convert EnsembleResult to API response."""
    return AnalysisResponse(
        is_safe=result.is_safe,
        final_score=result.final_score,
        confidence=result.confidence.value,
        category=result.category.value,
        threshold_used=result.threshold_used,
        models=result.models,
        detections_summary=[
            DetectionDetail(
                part=d["part"],
                confidence=d["confidence"],
                location=d.get("location")
            ) for d in result.detections_summary
        ],
        summary=result.summary,
        recommendation=result.recommendation,
        total_time_ms=result.total_time_ms
    )


def get_adaptive_sample_count(total_frames: int, fps: float, duration_seconds: float) -> int:
    """
    Calculate optimal number of frames to sample based on video characteristics.

    Short videos: More frames per second
    Long videos: Fewer frames per second but spread evenly
    """
    if duration_seconds <= 5:
        # Short video: sample every 0.5 seconds
        return min(int(duration_seconds * 2) + 1, total_frames, 15)
    elif duration_seconds <= 30:
        # Medium video: sample every 2 seconds
        return min(int(duration_seconds / 2) + 1, total_frames, 20)
    elif duration_seconds <= 120:
        # Long video: sample every 5 seconds
        return min(int(duration_seconds / 5) + 1, total_frames, 25)
    else:
        # Very long video: sample every 10 seconds, max 30 frames
        return min(int(duration_seconds / 10) + 1, total_frames, 30)


def analyze_frame_batch(frames: List[tuple], detector) -> List[tuple]:
    """
    Analyze a batch of frames in parallel.

    Args:
        frames: List of (frame_index, frame_array, timestamp) tuples
        detector: NSFW detector instance

    Returns:
        List of (frame_index, result, timestamp) tuples
    """
    results = []

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FRAMES) as executor:
        future_to_frame = {
            executor.submit(detector.analyze_image_from_array, frame[1]): frame
            for frame in frames
        }

        for future in as_completed(future_to_frame):
            frame_info = future_to_frame[future]
            frame_idx, frame_array, timestamp = frame_info

            try:
                result = future.result()
                results.append((frame_idx, result, timestamp))
            except Exception as e:
                logger.error(f"Frame {frame_idx} analysis failed: {e}")
                # Create a safe result for failed frames
                results.append((frame_idx, None, timestamp))

    return results


def analyze_video_frames_optimized(video_path: str, max_frames: int = None) -> tuple[EnsembleResult, VideoAnalysisResponse]:
    """
    Analyze video with optimized in-memory frame processing.

    OPTIMIZATIONS:
    - No temp file writes for frames (in-memory processing)
    - Parallel frame batch processing
    - Adaptive frame sampling
    - Aggressive early exit on high-confidence NSFW detection

    Returns:
        tuple: (worst_frame_result, video_analysis_response)
    """
    total_start = time.time()
    logger.info(f"Starting optimized video analysis: {video_path}")

    detector = get_detector(NSFW_THRESHOLD)

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_seconds = total_frames / fps if fps > 0 else 0

        if total_frames == 0:
            raise ValueError("Empty video file")

        # Adaptive sampling based on video length
        if max_frames is None:
            sample_count = get_adaptive_sample_count(total_frames, fps, duration_seconds)
        else:
            sample_count = min(max_frames, total_frames)

        logger.info(f"Video: {duration_seconds:.1f}s, {total_frames} frames, sampling {sample_count} frames")

        # Calculate frame indices to sample
        frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        frame_scores = []
        worst_result = None
        worst_score = 0.0
        worst_frame_idx = None
        early_exit = False

        # Process frames in batches for parallel execution
        batch_size = MAX_PARALLEL_FRAMES
        current_batch = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            timestamp = idx / fps if fps > 0 else 0
            current_batch.append((int(idx), frame, timestamp))

            # Process batch when full or last frame
            if len(current_batch) >= batch_size or idx == frame_indices[-1]:
                batch_results = analyze_frame_batch(current_batch, detector)

                for frame_idx, result, ts in batch_results:
                    if result is None:
                        # Skip failed frames
                        continue

                    frame_scores.append({
                        "frame": frame_idx,
                        "timestamp": round(ts, 2),
                        "score": result.final_score,
                        "category": result.category.value,
                        "is_safe": result.is_safe,
                        "analysis_time_ms": result.total_time_ms
                    })

                    if result.final_score > worst_score:
                        worst_score = result.final_score
                        worst_result = result
                        worst_frame_idx = frame_idx

                    # Early exit if clearly NSFW
                    if result.final_score > EARLY_EXIT_THRESHOLD:
                        logger.warning(f"High NSFW score ({result.final_score:.2f}) at frame {frame_idx}, stopping early")
                        early_exit = True
                        break

                current_batch = []

                if early_exit:
                    break

        cap.release()

        # If no valid analysis, return safe result
        if worst_result is None:
            worst_result = EnsembleResult(
                is_safe=True,
                final_score=0.0,
                confidence=ConfidenceLevel.LOW,
                category=ContentCategory.SAFE,
                threshold_used=NSFW_THRESHOLD,
                models={},
                detections_summary=[],
                summary="Could not analyze video frames",
                recommendation="ALLOW WITH CAUTION: Analysis incomplete"
            )

        total_time = (time.time() - total_start) * 1000

        # Build video response
        video_response = VideoAnalysisResponse(
            is_safe=worst_result.is_safe,
            final_score=worst_result.final_score,
            confidence=worst_result.confidence.value,
            category=worst_result.category.value,
            frames_analyzed=len(frame_scores),
            worst_frame=worst_frame_idx,
            frame_scores=frame_scores,
            summary=worst_result.summary + (f" (Early exit: high confidence)" if early_exit else ""),
            recommendation=worst_result.recommendation,
            total_time_ms=round(total_time, 1)
        )

        logger.info(f"Video analysis complete: {len(frame_scores)} frames in {total_time:.0f}ms "
                   f"(avg {total_time/max(len(frame_scores), 1):.0f}ms/frame)")

        return worst_result, video_response

    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        raise


# Legacy function for backwards compatibility
def analyze_video_frames(video_path: str, sample_frames: int = 15) -> tuple[EnsembleResult, VideoAnalysisResponse]:
    """
    Legacy wrapper for video analysis.
    Calls the optimized version.
    """
    return analyze_video_frames_optimized(video_path, max_frames=sample_frames)


# ============== API Endpoints ==============

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_media(file: UploadFile = File(...)):
    """
    Analyze media for NSFW content without saving.

    This endpoint is for testing the detection system.
    Returns detailed analysis with per-model breakdown and timing info.
    """
    media_type = get_media_type(file.filename)
    if media_type == 'unknown':
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload an image or video."
        )

    # Save temporarily for analysis
    temp_path = TEMP_DIR / f"analyze_{uuid.uuid4()}{Path(file.filename).suffix}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        detector = get_detector(NSFW_THRESHOLD)

        if media_type == 'image':
            result = detector.analyze_image(str(temp_path))
            return ensemble_to_response(result)
        else:
            result, video_response = analyze_video_frames_optimized(str(temp_path))
            # Return the worst frame analysis for videos
            response = ensemble_to_response(result)
            response.total_time_ms = video_response.total_time_ms
            return response

    finally:
        # Clean up
        temp_path.unlink(missing_ok=True)


@app.post("/api/analyze/video", response_model=VideoAnalysisResponse)
async def analyze_video_detailed(file: UploadFile = File(...)):
    """
    Analyze video with detailed frame-by-frame breakdown.

    Returns analysis for each sampled frame including timestamps and timing.
    """
    media_type = get_media_type(file.filename)
    if media_type != 'video':
        raise HTTPException(
            status_code=400,
            detail="This endpoint only accepts video files."
        )

    temp_path = TEMP_DIR / f"video_{uuid.uuid4()}{Path(file.filename).suffix}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result, video_response = analyze_video_frames_optimized(str(temp_path))
        return video_response

    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/analyze/quick")
async def quick_analyze(file: UploadFile = File(...)):
    """
    Quick NSFW check using only the fastest model.

    Use this for preliminary screening - returns in ~200-300ms.
    """
    media_type = get_media_type(file.filename)
    if media_type != 'image':
        raise HTTPException(
            status_code=400,
            detail="Quick analysis only supports images."
        )

    temp_path = TEMP_DIR / f"quick_{uuid.uuid4()}{Path(file.filename).suffix}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        start_time = time.time()
        detector = get_detector(NSFW_THRESHOLD)
        is_safe, score = detector.quick_check(str(temp_path))
        elapsed = (time.time() - start_time) * 1000

        return {
            "is_safe": is_safe,
            "score": round(score, 4),
            "threshold": NSFW_THRESHOLD,
            "model": "opennsfw2",
            "execution_time_ms": round(elapsed, 1)
        }

    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/upload", response_model=UploadResponse)
async def upload_media(
    file: UploadFile = File(...),
    caption: Optional[str] = Form(None)
):
    """
    Upload media file with NSFW detection.

    If content passes moderation, it's saved and posted.
    Returns detailed analysis regardless of outcome.
    """
    upload_start = time.time()
    logger.info(f"Upload request received: {file.filename}")

    media_type = get_media_type(file.filename)
    if media_type == 'unknown':
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload an image or video."
        )

    # Generate unique filename
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename

    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Analyze for NSFW content
    try:
        detector = get_detector(NSFW_THRESHOLD)

        if media_type == 'image':
            result = detector.analyze_image(str(file_path))
            analysis_response = ensemble_to_response(result)
        else:
            result, video_response = analyze_video_frames_optimized(str(file_path))
            analysis_response = AnalysisResponse(
                is_safe=video_response.is_safe,
                final_score=video_response.final_score,
                confidence=video_response.confidence,
                category=video_response.category,
                threshold_used=NSFW_THRESHOLD,
                models=result.models,
                detections_summary=[
                    DetectionDetail(part=d["part"], confidence=d["confidence"], location=d.get("location"))
                    for d in result.detections_summary
                ],
                summary=f"{video_response.summary} (Analyzed {video_response.frames_analyzed} frames)",
                recommendation=video_response.recommendation,
                total_time_ms=video_response.total_time_ms
            )

        logger.info(f"Analysis complete - Score: {result.final_score}, Safe: {result.is_safe}")

    except Exception as e:
        file_path.unlink(missing_ok=True)
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze content: {str(e)}")

    # If content is unsafe, delete and return
    if not result.is_safe:
        file_path.unlink(missing_ok=True)
        logger.warning(f"BLOCKED: {result.category.value} - Score: {result.final_score}")

        return UploadResponse(
            success=False,
            message=f"Content blocked: {result.summary}",
            analysis=analysis_response
        )

    # Save to database
    import json
    db = SessionLocal()
    try:
        post = Post(
            id=str(uuid.uuid4()),
            filename=unique_filename,
            original_filename=file.filename,
            media_type=media_type,
            caption=caption,
            nsfw_score=result.final_score,
            is_safe=True,
            file_path=str(file_path),
            analysis_json=json.dumps({
                "final_score": result.final_score,
                "category": result.category.value,
                "confidence": result.confidence.value,
                "models": result.models,
                "total_time_ms": getattr(result, 'total_time_ms', None)
            })
        )
        db.add(post)
        db.commit()
        db.refresh(post)

        post_response = PostResponse(
            id=post.id,
            filename=post.filename,
            original_filename=post.original_filename,
            media_type=post.media_type,
            caption=post.caption,
            nsfw_score=post.nsfw_score,
            is_safe=post.is_safe,
            created_at=post.created_at,
            file_url=f"/uploads/{post.filename}"
        )

        total_time = (time.time() - upload_start) * 1000
        logger.info(f"Post created: {post.id} in {total_time:.0f}ms total")

        return UploadResponse(
            success=True,
            message="Content uploaded successfully!",
            post=post_response,
            analysis=analysis_response
        )

    except Exception as e:
        file_path.unlink(missing_ok=True)
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()


@app.get("/api/posts", response_model=List[PostResponse])
async def get_posts(skip: int = 0, limit: int = 20):
    """Get all safe posts for the feed."""
    db = SessionLocal()
    try:
        posts = db.query(Post).filter(Post.is_safe == True).order_by(Post.created_at.desc()).offset(skip).limit(limit).all()

        return [
            PostResponse(
                id=post.id,
                filename=post.filename,
                original_filename=post.original_filename,
                media_type=post.media_type,
                caption=post.caption,
                nsfw_score=post.nsfw_score,
                is_safe=post.is_safe,
                created_at=post.created_at,
                file_url=f"/uploads/{post.filename}"
            )
            for post in posts
        ]
    finally:
        db.close()


@app.get("/api/posts/{post_id}", response_model=PostResponse)
async def get_post(post_id: str):
    """Get a specific post by ID."""
    db = SessionLocal()
    try:
        post = db.query(Post).filter(Post.id == post_id, Post.is_safe == True).first()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        return PostResponse(
            id=post.id,
            filename=post.filename,
            original_filename=post.original_filename,
            media_type=post.media_type,
            caption=post.caption,
            nsfw_score=post.nsfw_score,
            is_safe=post.is_safe,
            created_at=post.created_at,
            file_url=f"/uploads/{post.filename}"
        )
    finally:
        db.close()


@app.delete("/api/posts/{post_id}")
async def delete_post(post_id: str):
    """Delete a post."""
    db = SessionLocal()
    try:
        post = db.query(Post).filter(Post.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        # Delete file
        file_path = Path(post.file_path)
        file_path.unlink(missing_ok=True)

        # Delete from database
        db.delete(post)
        db.commit()

        return {"message": "Post deleted successfully"}
    finally:
        db.close()


@app.get("/api/health")
async def health_check():
    """Health check endpoint with model status."""
    detector = get_detector(NSFW_THRESHOLD)
    model_status = detector.get_model_status()

    return {
        "status": "healthy",
        "message": "Safe Media Social Feed API v2.1 - Optimized Detection",
        "models": model_status,
        "threshold": NSFW_THRESHOLD,
        "optimizations": [
            "parallel_model_execution",
            "in_memory_frame_processing",
            "adaptive_frame_sampling",
            "early_exit_detection"
        ]
    }


@app.get("/api/models/status")
async def get_model_status():
    """Get detailed status of all detection models."""
    detector = get_detector(NSFW_THRESHOLD)
    return detector.get_model_status()


@app.get("/api/benchmark")
async def get_benchmark_info():
    """
    Get expected performance benchmarks.

    These are approximate times based on typical hardware.
    Actual times depend on CPU/GPU availability.
    """
    return {
        "image_analysis": {
            "quick_check": "100-300ms (single model)",
            "full_analysis": "300-800ms (3 models parallel)",
            "previous_sequential": "900-2400ms (3 models sequential)"
        },
        "video_analysis": {
            "per_frame": "300-800ms (parallel models)",
            "batch_processing": f"{MAX_PARALLEL_FRAMES} frames processed concurrently",
            "early_exit_threshold": EARLY_EXIT_THRESHOLD,
            "adaptive_sampling": "5-30 frames based on video length"
        },
        "speedup_factors": {
            "image": "~3x faster (parallel vs sequential)",
            "video": "~4-5x faster (parallel frames + in-memory processing)"
        }
    }


# Preload models on startup
@app.on_event("startup")
async def startup_event():
    """Preload ML models on startup for faster first request."""
    logger.info("Starting up - loading detection models in parallel...")
    start_time = time.time()
    detector = get_detector(NSFW_THRESHOLD)
    detector.load_models()
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Startup complete - all models loaded in {elapsed:.0f}ms")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
