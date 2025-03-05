import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from datetime import datetime
import matplotlib.pyplot as plt

# from datasets.image import (
#     LSUNCrop,
#     LSUNResize,
#     TinyImageNetCrop,
#     TinyImageNetResize,
# )

from detector import (
    MaxSoftmax,
    ODIN,
    EnergyBased,
)

from models.wrn import WideResNet
from utils import OODMetrics, ToUnknown, fix_random_seed

# -------------------------------
# Set device and fix random seed
# -------------------------------
device = "cuda:0"
fix_random_seed(123)

# -------------------------------
# Setup Result Saving Directory
# -------------------------------
# Get current timestamp in Canada time
canada_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create result directory
result_dir = f"/workspace/OOD/results/{canada_time}/"
os.makedirs(result_dir, exist_ok=True)

# -------------------------------
# Preprocessing Setup
# -------------------------------
trans = WideResNet.transform_for("cifar10-pt")
norm_std = WideResNet.norm_std_for("cifar10-pt")

# -------------------------------
# Dataset Setup
# -------------------------------
cifar10_test = CIFAR10(root="data", train=False, transform=trans, download=True)

ood_dataset_classes = [
    # TinyImageNetCrop,
    # TinyImageNetResize,
    # LSUNCrop,
    # LSUNResize,
    CIFAR100,
    MNIST,
    # FashionMNIST,
]

datasets = {}
for dataset_cls in ood_dataset_classes:
    ood_dataset = dataset_cls(root="data", transform=trans, target_transform=ToUnknown(), download=True)
    combined_dataset = cifar10_test + ood_dataset
    loader = DataLoader(combined_dataset, batch_size=512, num_workers=12)
    datasets[dataset_cls.__name__] = loader

# -------------------------------
# Model Setup
# -------------------------------
print("STAGE 1: Creating pre-trained WideResNet model...")
model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval().to(device)

# -------------------------------
# Detector Setup
# -------------------------------
print("STAGE 2: Creating OOD Detectors...")
detectors = {
    "MSP": MaxSoftmax(model),
    "ODIN": ODIN(model, norm_std=norm_std, eps=0.002),
    "EnergyBased": EnergyBased(model),
}

# -------------------------------
# Detector Evaluation
# -------------------------------
print(f"STAGE 3: Evaluating {len(detectors)} detectors on {len(datasets)} OOD datasets...")
results = []

with torch.no_grad():
    for det_name, detector in detectors.items():
        print(f"> Evaluating {det_name}")
        for ds_name, loader in datasets.items():
            print(f"--> {ds_name}")
            metrics = OODMetrics()
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                scores = detector.predict(x)
                print(f"[DEBUG] {det_name} - {ds_name}: Scores = {scores[:5].cpu().numpy()} | Labels = {y[:5].cpu().numpy()}")

                metrics.update(scores, y)
            result = {"Detector": det_name, "Dataset": ds_name}
            result.update(metrics.compute())
            results.append(result)

# Convert results to DataFrame
df = pd.DataFrame(results)

# -------------------------------
# Save Results
# -------------------------------
# Save individual OOD dataset results
individual_results_path = os.path.join(result_dir, "individual_results.csv")
df.to_csv(individual_results_path, index=False)
print(f"Saved individual OOD dataset results to {individual_results_path}")

# Calculate mean performance across all datasets
mean_scores = df.groupby("Detector")[["AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"]].mean() * 100
mean_results_path = os.path.join(result_dir, "mean_results.csv")
mean_scores.to_csv(mean_results_path, float_format="%.2f")
print(f"Saved mean results to {mean_results_path}")

# -------------------------------
# Visualization
# -------------------------------
# Create and save plots
for metric in ["AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"]:
    plt.figure(figsize=(10, 5))
    
    # Plot for individual datasets
    for det_name in detectors.keys():
        subset = df[df["Detector"] == det_name]
        plt.plot(
            subset["Dataset"].to_numpy(),  # ← NumPy 배열로 변환
            subset[metric].to_numpy(),  # ← NumPy 배열로 변환
            marker='o',
            label=det_name
        )

    plt.title(f"{metric} per OOD Dataset")
    plt.xlabel("OOD Dataset")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Save figure
    metric_plot_path = os.path.join(result_dir, f"{metric}_per_dataset.png")
    plt.savefig(metric_plot_path, bbox_inches="tight")
    print(f"Saved {metric} plot to {metric_plot_path}")
    plt.close()

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset

# # Import detectors from the detector folder.
# from detector.softmax import MaxSoftmax
# from detector.energy import EnergyBased
# from detector.odin import ODIN


# from models.wrn import WideResNet
# def load_model(num_classes: int = 10) -> nn.Module:
#     """
#     Create a simple neural network model for testing.
#     This model mimics a simple classifier (e.g., for MNIST).
#     """
#     return nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(28 * 28, 128),
#         nn.ReLU(),
#         nn.Linear(128, num_classes)
#     )

# def get_dummy_data(num_samples: int = 100, input_shape=(1, 28, 28)) -> DataLoader:
#     """
#     Generate dummy data for testing.
#     This creates random images.
#     """
#     x = torch.rand(num_samples, *input_shape)
#     dataset = TensorDataset(x)
#     return DataLoader(dataset, batch_size=32)

# def main():
#     # Load the model and set it to evaluation mode.
#     model = load_model()
#     model.eval()

#     # Create dummy data.
#     data_loader = get_dummy_data()

#     # Initialize detectors with the model.
#     msp_detector = MaxSoftmax(model, temperature=1.0)
#     energy_detector = EnergyBased(model, temperature=1.0)
#     odin_detector = ODIN(model, eps=0.05, temperature=1000.0)

#     # Process one batch of data.
#     for batch in data_loader:
#         x = batch[0]  # Get input tensor.
#         # Compute OOD scores from each detector.
#         msp_scores = msp_detector.predict(x)
#         energy_scores = energy_detector.predict(x)
#         odin_scores = odin_detector.predict(x)
        
#         # Print the first five scores for each detector.
#         print("MaxSoftmax OOD Scores:", msp_scores[:5])
#         print("Energy-Based OOD Scores:", energy_scores[:5])
#         print("ODIN OOD Scores:", odin_scores[:5])
#         break  # Process only one batch for demonstration.

# if __name__ == '__main__':
#     main()
