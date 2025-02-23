import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ODIN:
    """
    ODIN OOD Detector.

    This method perturbs the input using the gradient of the loss to enhance the 
    discriminability between in-distribution and out-of-distribution (OOD) data.
    
    The perturbed input is computed as:
        x̂ = x - eps * sign(∇ₓ L(f(x)/T, ŷ))
    where ŷ is the predicted label when not provided, and T is the temperature scaling factor.
    
    The OOD score is then defined as:
        score(x) = - max(Softmax(f(x̂)))
    A higher score (closer to 0) indicates a higher chance that the input is OOD.
    
    Reference:
        "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks"
        https://arxiv.org/abs/1706.02690
    """
    def __init__(self, model: nn.Module, eps: float = 0.05, temperature: float = 1000.0,
                 criterion: callable = None, norm_std: list = None):
        if model is None:
            raise ValueError("Model must be provided.")
        self.model = model
        self.eps = eps
        self.temperature = temperature
        # Default loss is negative log-likelihood if none is provided.
        self.criterion = criterion if criterion is not None else F.nll_loss
        # Optional per-channel normalization standard deviations.
        self.norm_std = norm_std

    def _preprocess(self, x: Tensor) -> Tensor:
        """
        Perturb the input x using gradient information.

        Steps:
          1. Clone x and enable gradient computation.
          2. Compute scaled logits: f(x) / T.
          3. Determine target labels as the argmax if not provided.
          4. Calculate loss and backpropagate to obtain ∇ₓ L.
          5. Compute the sign of the gradients (optionally normalized per channel).
          6. Generate the perturbed input: x̂ = x - eps * sign(gradient).

        Returns:
            Tensor: The perturbed input x̂.
        """
        # Clone input and ensure gradients can be computed.
        x = x.clone().detach().requires_grad_(True)
        logits = self.model(x) / self.temperature
        # Use predicted labels (argmax) if targets are not provided.
        y = logits.argmax(dim=1)
        loss = self.criterion(logits, y)
        loss.backward()
        grad_sign = torch.sign(x.grad)
        # Normalize gradients per channel if norm_std is provided.
        if self.norm_std is not None:
            for i, std in enumerate(self.norm_std):
                grad_sign[:, i, ...] = grad_sign[:, i, ...] / std
        # Create perturbed input.
        x_hat = x - self.eps * grad_sign
        return x_hat

    def predict(self, x: Tensor) -> Tensor:
        """
        Compute the ODIN OOD score for input x.

        Process:
          1. Obtain the perturbed input x̂ via _preprocess(x).
          2. Pass x̂ through the model and compute softmax probabilities.
          3. The OOD score is defined as the negative of the maximum softmax probability.
        
        Returns:
            Tensor: OOD scores for each sample in x.
        """
        x_hat = self._preprocess(x)
        probs = F.softmax(self.model(x_hat), dim=1)
        return -probs.max(dim=1).values
