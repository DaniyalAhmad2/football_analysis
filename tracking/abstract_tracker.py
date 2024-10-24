from abc import ABC, abstractmethod
# from ultralytics import YOLO
# import torch
from typing import Any, Dict, List
# from ultralytics.engine.results import Results
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import numpy as np

class AbstractTracker(ABC):

    def __init__(self, model_id: str, conf: float = 0.1) -> None:
        """
        Initialize the inference client and set the confidence threshold.

        Args:
            model_id (str): ID of the model to use.
            conf (float): Confidence threshold for detections.
        """
        self.client = InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key="LwUcgr0hOvL7HYlvv4NI"
        ).select_model(
            model_id=model_id
        ).configure(
            InferenceConfiguration(confidence_threshold=conf)
        )
        self.conf = conf
        self.cur_frame = 0  # Initialize current frame counter

    @abstractmethod
    def detect(self, frames: List[np.ndarray]) -> List[Any]:
        """
        Abstract method for detection.

        Args:
            frames (List[np.ndarray]): List of frames for detection.

        Returns:
            List[Any]: List of detection result objects.
        """
        pass

    @abstractmethod
    def track(self, detection: Any) -> dict:
        """
        Abstract method for tracking detections.

        Args:
            detection (Any): Detection results for a single frame.

        Returns:
            dict: Tracking data.
        """
        pass