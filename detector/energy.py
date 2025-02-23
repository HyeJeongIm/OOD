import torch
import torch.nn as nn
from torch import Tensor, logsumexp

class EnergyBased:
    """
    Energy-Based OOD detector.

    The OOD score is computed as:
        score(x) = -T * logsumexp(logits / T)
    
    A higher energy score indicates a higher chance that the input is OOD.
    """
    def __init__(self, model: nn.Module, temperature: float = 1.0):
        if model is None:
            raise ValueError("Model must be provided.")
        self.model = model
        self.temperature = temperature

    def predict(self, x: Tensor) -> Tensor:
        """Compute the OOD score for input x."""
        logits = self.model(x)
        return -self.temperature * logsumexp(logits / self.temperature, dim=1)
