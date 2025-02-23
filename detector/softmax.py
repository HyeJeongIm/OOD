import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MaxSoftmax:
    """
    Maximum Softmax Probability (MSP) OOD detector.

    The OOD score is computed as:
        score(x) = - max(Softmax(logits / T))
    
    A lower maximum softmax probability (after negation, a higher score)
    indicates a higher chance that the input is OOD.
    """
    def __init__(self, model: nn.Module, temperature: float = 1.0):
        if model is None:
            raise ValueError("Model must be provided.")
        self.model = model
        self.temperature = temperature

    def predict(self, x: Tensor) -> Tensor:
        """Compute the OOD score for input x."""
        logits = self.model(x)
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        # Negate the maximum probability so that a higher score indicates OOD
        return -probs.max(dim=1).values
