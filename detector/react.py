import torch
import torch.nn as nn
from torch import Tensor

class ReAct:
    """
    ReAct OOD Detector.

    This detector clips the activations from a backbone network to a specified threshold
    before passing them to a head (classifier). The resulting logits are then fed to a detector
    function (default is an energy-based scoring) to compute OOD scores. A higher OOD score 
    (i.e., a value closer to 0) indicates a higher likelihood that the input is out-of-distribution.
    
    Reference:
        "ReAct: Out-of-distribution Detection With Rectified Activations"
        https://arxiv.org/abs/2111.12797
    """
    def __init__(self, backbone: callable, head: callable, threshold: float = 1.0,
                 detector: callable = None):
        """
        Initialize the ReAct detector.

        Args:
            backbone (callable): The feature extractor part of the model.
            head (callable): The classifier head that outputs logits.
            threshold (float): The maximum value to clip the backbone activations.
            detector (callable): Function to compute OOD scores from logits.
                                 Default is an energy-based detector.
        """
        if backbone is None or head is None:
            raise ValueError("Both backbone and head must be provided.")
        self.backbone = backbone
        self.head = head
        self.threshold = threshold
        self.detector = detector if detector is not None else self.default_detector

    def default_detector(self, logits: Tensor) -> Tensor:
        """
        Default energy-based detector that computes OOD scores from logits.

        Args:
            logits (Tensor): Logits from the head.

        Returns:
            Tensor: OOD scores.
        """
        # Compute energy score: -logsumexp(logits)
        return -torch.logsumexp(logits, dim=1)

    def predict(self, x: Tensor) -> Tensor:
        """
        Compute the OOD score for input x.

        Process:
          1. Pass input x through the backbone to extract features.
          2. Clip the features to the specified threshold.
          3. Pass the clipped features through the head to obtain logits.
          4. Compute and return the OOD score using the detector function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: OOD scores for each sample.
        """
        features = self.backbone(x)
        # Clip the activations to the threshold.
        clipped_features = torch.clamp(features, max=self.threshold)
        logits = self.head(clipped_features)
        return self.detector(logits)
