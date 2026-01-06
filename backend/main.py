import os
import uuid
import shutil
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Safe Media Social Feed API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = "sqlite:///./posts.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# NSFW threshold - VERY STRICT: anything above 0.2 (20%) is blocked
NSFW_THRESHOLD = 0.2

# NudeNet classes that indicate unsafe content
UNSAFE_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    "FEMALE_BREAST_COVERED",  # Also flag covered nudity for stricter filtering
    "BELLY_EXPOSED",
    "MALE_BREAST_EXPOSED",
}

# Highly explicit classes - immediate block
EXPLICIT_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

# Initialize NudeNet detector globally
nudenet_classifier = None

def get_nudenet_classifier():
    """Lazy load NudeNet classifier."""
    global nudenet_classifier
    if nudenet_classifier is None:
        try:
            from nudenet import NudeDetector
            nudenet_classifier = NudeDetector()
            logger.info("NudeNet detector loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NudeNet: {e}")
            nudenet_classifier = None
    return nudenet_classifier


def analyze_with_nudenet(image_path: str) -> tuple[float, list]:
    """
    Analyze image with NudeNet detector.
    Returns (nsfw_score, detected_labels)
    """
    try:
        detector = get_nudenet_classifier()
        if detector is None:
            logger.warning("NudeNet not available, skipping")
            return 0.0, []

        # Detect nudity
        detections = detector.detect(image_path)
        logger.info(f"NudeNet detections: {detections}")

        if not detections:
            return 0.0, []

        detected_labels = []
        max_score = 0.0
        has_explicit = False

        for detection in detections:
            label = detection.get('class', '')
            score = detection.get('score', 0.0)
            detected_labels.append(f"{label}:{score:.2f}")

            # Check for explicit content
            if label in EXPLICIT_CLASSES:
                has_explicit = True
                max_score = max(max_score, score)
                logger.warning(f"EXPLICIT CONTENT DETECTED: {label} with score {score}")
            elif label in UNSAFE_CLASSES:
                max_score = max(max_score, score * 0.8)  # Weight less than explicit

        # If explicit content found, return high score
        if has_explicit:
            return max(max_score, 0.9), detected_labels

        return max_score, detected_labels

    except Exception as e:
        logger.error(f"NudeNet analysis error: {e}")
        return 0.0, []


def analyze_with_opennsfw2(image_path: str) -> float:
    """Analyze image with OpenNSFW2."""
    try:
        import opennsfw2 as n2

        # Predict NSFW probability
        nsfw_probability = n2.predict_image(image_path)
        logger.info(f"OpenNSFW2 score: {nsfw_probability}")
        return float(nsfw_probability)

    except Exception as e:
        logger.error(f"OpenNSFW2 analysis error: {e}")
        return 0.0


def analyze_image_nsfw(image_path: str) -> tuple[float, str]:
    """
    Analyze an image for NSFW content using multiple detectors.
    Returns (nsfw_score, reason)
    """
    logger.info(f"Analyzing image: {image_path}")

    # Run both detectors
    opennsfw_score = analyze_with_opennsfw2(image_path)
    nudenet_score, nudenet_labels = analyze_with_nudenet(image_path)

    logger.info(f"OpenNSFW2 score: {opennsfw_score}, NudeNet score: {nudenet_score}")
    logger.info(f"NudeNet labels: {nudenet_labels}")

    # Take the maximum score from both detectors
    final_score = max(opennsfw_score, nudenet_score)

    # Build reason string
    reasons = []
    if opennsfw_score > NSFW_THRESHOLD:
        reasons.append(f"NSFW content detected (score: {opennsfw_score:.1%})")
    if nudenet_labels:
        reasons.append(f"Detected: {', '.join(nudenet_labels)}")

    reason = "; ".join(reasons) if reasons else "Content appears safe"

    logger.info(f"Final NSFW score: {final_score}, Reason: {reason}")

    return final_score, reason


def analyze_video_nsfw(video_path: str, sample_frames: int = 15) -> tuple[float, str]:
    """
    Analyze a video for NSFW content by sampling frames.
    Returns (max_nsfw_score, reason)
    """
    logger.info(f"Analyzing video: {video_path}")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return 0.0, "Could not analyze video"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return 0.0, "Empty video"

        # Sample frames evenly throughout the video
        frame_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)

        max_score = 0.0
        all_reasons = []
        temp_dir = UPLOAD_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Save frame temporarily
                temp_frame_path = str(temp_dir / f"temp_frame_{uuid.uuid4()}.jpg")
                cv2.imwrite(temp_frame_path, frame)

                # Analyze frame
                score, reason = analyze_image_nsfw(temp_frame_path)

                if score > max_score:
                    max_score = score
                    if reason and "safe" not in reason.lower():
                        all_reasons.append(f"Frame {idx}: {reason}")

                # Clean up temp frame
                try:
                    os.unlink(temp_frame_path)
                except:
                    pass

                # Early exit if clearly NSFW
                if max_score > 0.8:
                    logger.warning(f"High NSFW score detected at frame {idx}, stopping analysis")
                    break

        cap.release()

        final_reason = "; ".join(all_reasons[:3]) if all_reasons else "Content appears safe"
        logger.info(f"Video analysis complete. Max score: {max_score}")

        return max_score, final_reason

    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        return 0.0, f"Analysis error: {str(e)}"


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


Base.metadata.create_all(bind=engine)


class PostResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    media_type: str
    caption: Optional[str]
    nsfw_score: float
    is_safe: bool
    created_at: datetime
    file_url: str

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    success: bool
    message: str
    post: Optional[PostResponse] = None
    nsfw_score: Optional[float] = None


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


@app.post("/api/upload", response_model=UploadResponse)
async def upload_media(
    file: UploadFile = File(...),
    caption: Optional[str] = Form(None)
):
    """Upload media file with NSFW detection."""

    logger.info(f"Upload request received: {file.filename}")

    # Validate file type
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

    # Save file temporarily
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved temporarily: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Analyze for NSFW content
    try:
        if media_type == 'image':
            nsfw_score, reason = analyze_image_nsfw(str(file_path))
        else:
            nsfw_score, reason = analyze_video_nsfw(str(file_path))

        logger.info(f"Analysis complete - Score: {nsfw_score}, Reason: {reason}")

    except Exception as e:
        # Clean up file if analysis fails
        file_path.unlink(missing_ok=True)
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze content: {str(e)}")

    is_safe = nsfw_score < NSFW_THRESHOLD

    # If content is unsafe, delete the file and return warning
    if not is_safe:
        file_path.unlink(missing_ok=True)
        logger.warning(f"BLOCKED: Unsafe content detected - Score: {nsfw_score}")

        return UploadResponse(
            success=False,
            message=f"Content blocked: This media contains inappropriate content and cannot be posted. {reason}",
            nsfw_score=nsfw_score
        )

    # Save to database
    db = SessionLocal()
    try:
        post = Post(
            id=str(uuid.uuid4()),
            filename=unique_filename,
            original_filename=file.filename,
            media_type=media_type,
            caption=caption,
            nsfw_score=nsfw_score,
            is_safe=True,
            file_path=str(file_path)
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

        logger.info(f"Post created successfully: {post.id}")

        return UploadResponse(
            success=True,
            message="Content uploaded successfully!",
            post=post_response,
            nsfw_score=nsfw_score
        )
    except Exception as e:
        file_path.unlink(missing_ok=True)
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save to database: {str(e)}")
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
    """Health check endpoint."""
    # Check if NudeNet is loaded
    nudenet_status = "loaded" if nudenet_classifier is not None else "not loaded"
    return {
        "status": "healthy",
        "message": "Safe Media Social Feed API is running",
        "nudenet": nudenet_status,
        "nsfw_threshold": NSFW_THRESHOLD
    }


# Preload models on startup
@app.on_event("startup")
async def startup_event():
    """Preload ML models on startup."""
    logger.info("Starting up - loading models...")
    get_nudenet_classifier()
    logger.info("Startup complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
