"""
Robust NSFW Detection System with Multi-Model Ensemble

This module implements a sophisticated content moderation system using multiple
AI models for accurate nudity and sexual content detection.

Models used:
1. OpenNSFW2 - Yahoo's general NSFW classifier
2. NudeNet - Body part detection with bounding boxes
3. Transformers (Falconsai) - HuggingFace NSFW image classification
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from PIL import Image

logger = logging.getLogger(__name__)


class ContentCategory(str, Enum):
    """Content classification categories."""
    SAFE = "safe"
    SUGGESTIVE = "suggestive"
    PARTIAL_NUDITY = "partial_nudity"
    EXPLICIT_NUDITY = "explicit_nudity"
    SEXUAL_CONTENT = "sexual_content"


class ConfidenceLevel(str, Enum):
    """Confidence level of the detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class BoundingBox:
    """Bounding box for detected content."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class Detection:
    """Single detection result from a model."""
    class_name: str
    score: float
    box: Optional[BoundingBox] = None


@dataclass
class ModelResult:
    """Result from a single detection model."""
    model_name: str
    score: float
    label: str
    is_available: bool = True
    detections: list = field(default_factory=list)
    raw_output: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class EnsembleResult:
    """Final ensemble analysis result."""
    is_safe: bool
    final_score: float
    confidence: ConfidenceLevel
    category: ContentCategory
    threshold_used: float
    models: dict  # model_name -> ModelResult
    detections_summary: list  # List of all detected body parts
    summary: str
    recommendation: str


# Body part classification
EXPLICIT_BODY_PARTS = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
}

NUDITY_BODY_PARTS = {
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
}

PARTIAL_NUDITY_PARTS = {
    "FEMALE_BREAST_COVERED",
    "BELLY_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "FEET_EXPOSED",
    "ARMPITS_EXPOSED",
}

# Model weight configuration (higher = more influence)
MODEL_WEIGHTS = {
    "opennsfw2": 0.30,
    "nudenet": 0.40,
    "transformers": 0.30,
}


class NSFWDetector:
    """
    Multi-model ensemble NSFW detector.

    Combines multiple AI models to provide accurate and robust
    content moderation with detailed analysis reports.
    """

    def __init__(self, threshold: float = 0.2):
        """
        Initialize the NSFW detector.

        Args:
            threshold: Score threshold above which content is considered unsafe (0.0-1.0)
        """
        self.threshold = threshold
        self._opennsfw_model = None
        self._nudenet_detector = None
        self._transformers_pipeline = None
        self._models_loaded = False

    def load_models(self):
        """Preload all detection models."""
        logger.info("Loading NSFW detection models...")
        self._get_opennsfw_model()
        self._get_nudenet_detector()
        self._get_transformers_pipeline()
        self._models_loaded = True
        logger.info("All models loaded successfully")

    def _get_opennsfw_model(self):
        """Lazy load OpenNSFW2 model."""
        if self._opennsfw_model is None:
            try:
                import opennsfw2 as n2
                self._opennsfw_model = n2
                logger.info("OpenNSFW2 model loaded")
            except Exception as e:
                logger.error(f"Failed to load OpenNSFW2: {e}")
        return self._opennsfw_model

    def _get_nudenet_detector(self):
        """Lazy load NudeNet detector."""
        if self._nudenet_detector is None:
            try:
                from nudenet import NudeDetector
                self._nudenet_detector = NudeDetector()
                logger.info("NudeNet detector loaded")
            except Exception as e:
                logger.error(f"Failed to load NudeNet: {e}")
        return self._nudenet_detector

    def _get_transformers_pipeline(self):
        """Lazy load Transformers NSFW classifier."""
        if self._transformers_pipeline is None:
            try:
                from transformers import pipeline
                self._transformers_pipeline = pipeline(
                    "image-classification",
                    model="Falconsai/nsfw_image_detection"
                )
                logger.info("Transformers NSFW pipeline loaded")
            except Exception as e:
                logger.error(f"Failed to load Transformers pipeline: {e}")
        return self._transformers_pipeline

    def _analyze_with_opennsfw2(self, image_path: str) -> ModelResult:
        """Analyze image with OpenNSFW2."""
        try:
            model = self._get_opennsfw_model()
            if model is None:
                return ModelResult(
                    model_name="opennsfw2",
                    score=0.0,
                    label="unknown",
                    is_available=False,
                    error="Model not available"
                )

            nsfw_probability = float(model.predict_image(image_path))
            label = "nsfw" if nsfw_probability > 0.5 else "safe"

            return ModelResult(
                model_name="opennsfw2",
                score=nsfw_probability,
                label=label,
                is_available=True,
                raw_output={"nsfw_probability": nsfw_probability}
            )

        except Exception as e:
            logger.error(f"OpenNSFW2 analysis error: {e}")
            return ModelResult(
                model_name="opennsfw2",
                score=0.0,
                label="error",
                is_available=False,
                error=str(e)
            )

    def _analyze_with_nudenet(self, image_path: str) -> ModelResult:
        """Analyze image with NudeNet."""
        try:
            detector = self._get_nudenet_detector()
            if detector is None:
                return ModelResult(
                    model_name="nudenet",
                    score=0.0,
                    label="unknown",
                    is_available=False,
                    error="Model not available"
                )

            raw_detections = detector.detect(image_path)
            logger.info(f"NudeNet raw detections: {raw_detections}")

            if not raw_detections:
                return ModelResult(
                    model_name="nudenet",
                    score=0.0,
                    label="safe",
                    is_available=True,
                    detections=[],
                    raw_output={"detections": []}
                )

            detections = []
            max_score = 0.0
            has_explicit = False
            has_nudity = False
            has_partial = False

            for det in raw_detections:
                class_name = det.get('class', '')
                score = det.get('score', 0.0)
                box_data = det.get('box', [])

                box = None
                if box_data and len(box_data) == 4:
                    box = BoundingBox(
                        x=int(box_data[0]),
                        y=int(box_data[1]),
                        width=int(box_data[2] - box_data[0]),
                        height=int(box_data[3] - box_data[1])
                    )

                detections.append(Detection(
                    class_name=class_name,
                    score=score,
                    box=box
                ))

                # Categorize detection
                if class_name in EXPLICIT_BODY_PARTS:
                    has_explicit = True
                    max_score = max(max_score, score)
                elif class_name in NUDITY_BODY_PARTS:
                    has_nudity = True
                    max_score = max(max_score, score * 0.9)
                elif class_name in PARTIAL_NUDITY_PARTS:
                    has_partial = True
                    max_score = max(max_score, score * 0.5)

            # Determine label based on findings
            if has_explicit:
                label = "explicit"
                max_score = max(max_score, 0.95)  # Ensure high score for explicit
            elif has_nudity:
                label = "nudity"
                max_score = max(max_score, 0.80)
            elif has_partial:
                label = "partial_nudity"
            else:
                label = "safe"

            return ModelResult(
                model_name="nudenet",
                score=max_score,
                label=label,
                is_available=True,
                detections=[{
                    "class": d.class_name,
                    "score": d.score,
                    "box": {"x": d.box.x, "y": d.box.y, "w": d.box.width, "h": d.box.height} if d.box else None
                } for d in detections],
                raw_output={"raw_detections": raw_detections}
            )

        except Exception as e:
            logger.error(f"NudeNet analysis error: {e}")
            return ModelResult(
                model_name="nudenet",
                score=0.0,
                label="error",
                is_available=False,
                error=str(e)
            )

    def _analyze_with_transformers(self, image_path: str) -> ModelResult:
        """Analyze image with Transformers NSFW classifier."""
        try:
            pipeline = self._get_transformers_pipeline()
            if pipeline is None:
                return ModelResult(
                    model_name="transformers",
                    score=0.0,
                    label="unknown",
                    is_available=False,
                    error="Model not available"
                )

            # Load and process image
            image = Image.open(image_path).convert("RGB")
            results = pipeline(image)

            logger.info(f"Transformers results: {results}")

            # Parse results
            nsfw_score = 0.0
            label = "safe"

            for result in results:
                result_label = result.get('label', '').lower()
                score = result.get('score', 0.0)

                if result_label in ['nsfw', 'porn', 'sexy', 'hentai']:
                    nsfw_score = max(nsfw_score, score)
                    if score > 0.5:
                        label = "nsfw"
                elif result_label == 'normal' or result_label == 'safe':
                    if score > 0.5:
                        label = "safe"
                        nsfw_score = 1 - score

            return ModelResult(
                model_name="transformers",
                score=nsfw_score,
                label=label,
                is_available=True,
                raw_output={"predictions": results}
            )

        except Exception as e:
            logger.error(f"Transformers analysis error: {e}")
            return ModelResult(
                model_name="transformers",
                score=0.0,
                label="error",
                is_available=False,
                error=str(e)
            )

    def _calculate_ensemble_score(self, model_results: dict) -> tuple[float, float]:
        """
        Calculate weighted ensemble score from all models.

        Returns:
            tuple: (final_score, confidence_score)
        """
        total_weight = 0.0
        weighted_score = 0.0
        available_models = 0
        scores = []

        for model_name, result in model_results.items():
            if result.is_available and result.error is None:
                weight = MODEL_WEIGHTS.get(model_name, 0.25)
                weighted_score += result.score * weight
                total_weight += weight
                available_models += 1
                scores.append(result.score)

        if total_weight == 0:
            return 0.0, 0.0

        final_score = weighted_score / total_weight

        # Calculate confidence based on model agreement
        if len(scores) >= 2:
            score_variance = sum((s - final_score) ** 2 for s in scores) / len(scores)
            # Lower variance = higher confidence
            confidence_score = 1.0 - min(score_variance * 2, 1.0)
        else:
            confidence_score = 0.5  # Low confidence with single model

        # Boost confidence if multiple models agree
        if available_models >= 2:
            agreement_count = sum(1 for s in scores if abs(s - final_score) < 0.2)
            if agreement_count == available_models:
                confidence_score = min(confidence_score + 0.2, 1.0)

        return final_score, confidence_score

    def _determine_category(self, model_results: dict, final_score: float) -> ContentCategory:
        """Determine content category based on analysis results."""
        nudenet_result = model_results.get("nudenet")

        if nudenet_result and nudenet_result.is_available:
            if nudenet_result.label == "explicit":
                return ContentCategory.EXPLICIT_NUDITY
            elif nudenet_result.label == "nudity":
                return ContentCategory.PARTIAL_NUDITY
            elif nudenet_result.label == "partial_nudity":
                return ContentCategory.SUGGESTIVE

        # Fallback to score-based categorization
        if final_score >= 0.8:
            return ContentCategory.EXPLICIT_NUDITY
        elif final_score >= 0.6:
            return ContentCategory.SEXUAL_CONTENT
        elif final_score >= 0.4:
            return ContentCategory.PARTIAL_NUDITY
        elif final_score >= 0.2:
            return ContentCategory.SUGGESTIVE
        else:
            return ContentCategory.SAFE

    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if confidence_score >= 0.85:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _generate_summary(self, category: ContentCategory, final_score: float,
                          model_results: dict, confidence: ConfidenceLevel) -> str:
        """Generate human-readable summary of the analysis."""
        nudenet_result = model_results.get("nudenet")
        detected_parts = []

        if nudenet_result and nudenet_result.detections:
            for det in nudenet_result.detections:
                if det.get("score", 0) > 0.3:
                    detected_parts.append(f"{det['class']} ({det['score']:.0%})")

        if category == ContentCategory.SAFE:
            return f"Content appears safe. NSFW score: {final_score:.1%}"
        elif category == ContentCategory.SUGGESTIVE:
            parts_str = ", ".join(detected_parts[:3]) if detected_parts else "suggestive elements"
            return f"Suggestive content detected: {parts_str}. Score: {final_score:.1%}"
        elif category == ContentCategory.PARTIAL_NUDITY:
            parts_str = ", ".join(detected_parts[:3]) if detected_parts else "partial nudity"
            return f"Partial nudity detected: {parts_str}. Score: {final_score:.1%}"
        elif category == ContentCategory.EXPLICIT_NUDITY:
            parts_str = ", ".join(detected_parts[:3]) if detected_parts else "explicit content"
            return f"EXPLICIT NUDITY DETECTED: {parts_str}. Score: {final_score:.1%}"
        else:
            return f"Sexual content detected with {confidence.value} confidence. Score: {final_score:.1%}"

    def _generate_recommendation(self, is_safe: bool, category: ContentCategory,
                                  confidence: ConfidenceLevel) -> str:
        """Generate action recommendation."""
        if is_safe:
            if category == ContentCategory.SAFE:
                return "ALLOW: Content is safe to display"
            else:
                return "ALLOW WITH CAUTION: Content is borderline, consider manual review"
        else:
            if confidence in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]:
                return "BLOCK: High confidence unsafe content detected"
            else:
                return "BLOCK (REVIEW RECOMMENDED): Content flagged but confidence is not high"

    def analyze_image(self, image_path: str) -> EnsembleResult:
        """
        Analyze an image for NSFW content using all available models.

        Args:
            image_path: Path to the image file

        Returns:
            EnsembleResult with detailed analysis
        """
        logger.info(f"Starting ensemble analysis for: {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run all models
        model_results = {
            "opennsfw2": self._analyze_with_opennsfw2(image_path),
            "nudenet": self._analyze_with_nudenet(image_path),
            "transformers": self._analyze_with_transformers(image_path),
        }

        # Log individual results
        for name, result in model_results.items():
            logger.info(f"{name}: score={result.score:.3f}, label={result.label}, available={result.is_available}")

        # Calculate ensemble score
        final_score, confidence_score = self._calculate_ensemble_score(model_results)

        # Determine if safe
        is_safe = final_score < self.threshold

        # Determine category and confidence level
        category = self._determine_category(model_results, final_score)
        confidence = self._determine_confidence_level(confidence_score)

        # Collect all detections
        detections_summary = []
        nudenet_result = model_results.get("nudenet")
        if nudenet_result and nudenet_result.detections:
            for det in nudenet_result.detections:
                detections_summary.append({
                    "part": det.get("class", "unknown"),
                    "confidence": det.get("score", 0),
                    "location": det.get("box")
                })

        # Generate summary and recommendation
        summary = self._generate_summary(category, final_score, model_results, confidence)
        recommendation = self._generate_recommendation(is_safe, category, confidence)

        # Build models dict for response
        models_dict = {}
        for name, result in model_results.items():
            models_dict[name] = {
                "score": round(result.score, 4),
                "label": result.label,
                "is_available": result.is_available,
                "detections": result.detections if result.detections else [],
                "error": result.error
            }

        result = EnsembleResult(
            is_safe=is_safe,
            final_score=round(final_score, 4),
            confidence=confidence,
            category=category,
            threshold_used=self.threshold,
            models=models_dict,
            detections_summary=detections_summary,
            summary=summary,
            recommendation=recommendation
        )

        logger.info(f"Analysis complete: safe={is_safe}, score={final_score:.3f}, category={category.value}")

        return result

    def get_model_status(self) -> dict:
        """Get status of all detection models."""
        return {
            "opennsfw2": {
                "loaded": self._opennsfw_model is not None,
                "weight": MODEL_WEIGHTS["opennsfw2"]
            },
            "nudenet": {
                "loaded": self._nudenet_detector is not None,
                "weight": MODEL_WEIGHTS["nudenet"]
            },
            "transformers": {
                "loaded": self._transformers_pipeline is not None,
                "weight": MODEL_WEIGHTS["transformers"]
            },
            "threshold": self.threshold
        }


# Global detector instance
_detector_instance: Optional[NSFWDetector] = None


def get_detector(threshold: float = 0.2) -> NSFWDetector:
    """Get or create the global detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = NSFWDetector(threshold=threshold)
    return _detector_instance
