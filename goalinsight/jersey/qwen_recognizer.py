"""Jersey number recognition using Qwen VL (Vision-Language Model).

Implements BaseJerseyRecognizer interface.
"""

import base64
import os
import re
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from ..interfaces import BaseJerseyRecognizer

try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError:
    torch = None
    AutoProcessor = None
    AutoModelForImageTextToText = None

try:
    import httpx
except ImportError:
    httpx = None


class QwenJerseyRecognizer(BaseJerseyRecognizer):
    """Recognize jersey numbers using Qwen VL model.

    Supports both local model inference and API mode.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize jersey recognizer.

        Args:
            config: Configuration with keys:
                - mode: "local" or "api"
                - local: {model, device, max_new_tokens}
                - api: {base_url, model, api_key_env}
        """
        self.config = config or {}
        self.mode = self.config.get("mode", "local")

        # Local model settings
        local_config = self.config.get("local", {})
        self.model_name = local_config.get("model", "Qwen/Qwen3-VL-8B-Instruct")
        self.device = local_config.get("device", "cuda")
        self.max_new_tokens = local_config.get("max_new_tokens", 10)

        # API settings
        api_config = self.config.get("api", {})
        self.api_base_url = api_config.get("base_url")
        self.api_model = api_config.get("model", "qwen3-vl-8b")
        self.api_key_env = api_config.get("api_key_env", "QWEN_API_KEY")

        self.model = None
        self.processor = None
        self._api_client = None

    def load_model(self) -> None:
        """Load the local VLM model."""
        if torch is None or AutoModelForImageTextToText is None:
            raise ImportError(
                "transformers and torch packages are required. "
                "Install with: pip install transformers torch"
            )

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )

    def _image_to_base64(self, image: np.ndarray | Image.Image) -> str:
        """Convert image to base64 string."""
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]
            image = Image.fromarray(image)

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_jersey_number(self, response: str) -> int | None:
        """Parse jersey number from model response."""
        response = response.strip().lower()

        # Check for "no number" responses
        no_number_phrases = [
            "no number", "not visible", "cannot see", "unclear",
            "no jersey", "can't see", "unable", "none", "n/a"
        ]
        for phrase in no_number_phrases:
            if phrase in response:
                return None

        # Extract numbers from response
        numbers = re.findall(r'\b(\d{1,2})\b', response)
        if numbers:
            for num_str in numbers:
                num = int(num_str)
                if 1 <= num <= 99:
                    return num

        return None

    def _recognize_local(self, crop: np.ndarray) -> tuple[int | None, float]:
        """Recognize jersey number using local model."""
        if self.model is None:
            self.load_model()

        # Convert to PIL Image
        if len(crop.shape) == 3 and crop.shape[2] == 3:
            crop_rgb = crop[:, :, ::-1]
        else:
            crop_rgb = crop
        pil_image = Image.fromarray(crop_rgb)

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "What is the jersey number of the person in this image? Reply with only the number, or 'none' if no number is visible."}
                ]
            }
        ]

        # Process input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        jersey_number = self._parse_jersey_number(response)
        confidence = 1.0 if jersey_number is not None else 0.0

        return jersey_number, confidence

    def _recognize_api(self, crop: np.ndarray) -> tuple[int | None, float]:
        """Recognize jersey number using API."""
        if httpx is None:
            raise ImportError("httpx package is required for API mode.")

        if self.api_base_url is None:
            raise ValueError("API base_url must be configured for API mode")

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {self.api_key_env}")

        image_base64 = self._image_to_base64(crop)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.api_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        },
                        {
                            "type": "text",
                            "text": "What is the jersey number of the person in this image? Reply with only the number, or 'none' if no number is visible."
                        }
                    ]
                }
            ],
            "max_tokens": self.max_new_tokens,
        }

        if self._api_client is None:
            self._api_client = httpx.Client(timeout=30.0)

        response = self._api_client.post(
            f"{self.api_base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        result = response.json()
        response_text = result["choices"][0]["message"]["content"]

        jersey_number = self._parse_jersey_number(response_text)
        confidence = 1.0 if jersey_number is not None else 0.0

        return jersey_number, confidence

    def recognize(self, crop: np.ndarray) -> tuple[int | None, float]:
        """Recognize jersey number from player crop.

        Args:
            crop: Player image crop (BGR format).

        Returns:
            Tuple of (jersey_number, confidence).
        """
        try:
            if self.mode == "api":
                return self._recognize_api(crop)
            else:
                return self._recognize_local(crop)
        except Exception:
            return None, 0.0

    def recognize_batch(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[int | None, float]]:
        """Recognize jersey numbers from multiple crops.

        Args:
            crops: List of player image crops.

        Returns:
            List of (jersey_number, confidence) tuples.
        """
        results = []
        for crop in crops:
            try:
                result = self.recognize(crop)
                results.append(result)
            except Exception:
                results.append((None, 0.0))
        return results


def process_vision_info(messages: list[dict]) -> tuple[list, list]:
    """Process vision information from messages.

    Helper function for Qwen2-VL input processing.
    """
    image_inputs = []
    video_inputs = []

    for message in messages:
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        image = item.get("image")
                        if image is not None:
                            image_inputs.append(image)
                    elif item.get("type") == "video":
                        video = item.get("video")
                        if video is not None:
                            video_inputs.append(video)

    return image_inputs if image_inputs else None, video_inputs if video_inputs else None


class JerseyNumberAggregator:
    """Aggregate jersey number predictions over time."""

    def __init__(self, window_size: int = 5):
        """Initialize aggregator.

        Args:
            window_size: Size of sliding window for voting.
        """
        self.window_size = window_size
        self.predictions: dict[int, list[tuple[int | None, float]]] = {}

    def add_prediction(
        self,
        track_id: int,
        jersey_number: int | None,
        confidence: float,
    ) -> None:
        """Add a prediction for a track."""
        if track_id not in self.predictions:
            self.predictions[track_id] = []

        self.predictions[track_id].append((jersey_number, confidence))

        # Keep only recent predictions
        if len(self.predictions[track_id]) > self.window_size * 2:
            self.predictions[track_id] = self.predictions[track_id][-self.window_size * 2:]

    def get_consensus(self, track_id: int) -> tuple[int | None, float]:
        """Get consensus jersey number for a track."""
        if track_id not in self.predictions:
            return None, 0.0

        recent = self.predictions[track_id][-self.window_size:]

        # Filter to valid predictions
        valid = [(num, conf) for num, conf in recent if num is not None]
        if not valid:
            return None, 0.0

        # Count occurrences
        from collections import Counter
        number_counts = Counter(num for num, _ in valid)

        # Get most common number
        most_common = number_counts.most_common(1)[0]
        jersey_number = most_common[0]
        count = most_common[1]

        # Confidence based on agreement ratio
        confidence = count / len(recent)

        return jersey_number, confidence

    def clear(self, track_id: int | None = None) -> None:
        """Clear predictions."""
        if track_id is not None:
            self.predictions.pop(track_id, None)
        else:
            self.predictions.clear()


# Backwards compatibility alias
JerseyRecognizer = QwenJerseyRecognizer
