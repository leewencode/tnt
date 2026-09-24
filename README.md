# Carbon Nanotube (TNT) Microscopic Image Segmentation & Topology Analysis

## Project Overview
An advanced, multi-architecture deep learning and morphological image processing pipeline built in PyTorch to detect, segment, and reconstruct carbon nanotube (TNT) structures from high-resolution microscopic imagery. The project implements a hybrid workflow comparing U-Net variants (ResNet-34, ConvNeXt-Nano) and Vision Transformers (SegFormer-B3), utilizing specialized topology-aware losses (`clDice`), Test-Time Augmentation (TTA), hysteresis thresholding, and morphological geodesic reconstruction.

---

## Pipeline Workflow
1. **Automated Calibration & Patch Extraction:** Automatically calibrates marker HSV color spaces from annotation templates, extracts overlapping tiles, and handles background balancing (pure background vs. hard-negative edge mining).
2. **Data Augmentation & Preprocessing:** Applies domain-specific augmentations via Albumentations (CLAHE contrast enhancement, random rotations, gamma adjustments, and normalizations).
3. **Deep Learning Architectures (7 Comparative Pipelines):**
   - **Pipeline 1:** ResNet-34 U-Net baseline.
   - **Pipeline 2:** ConvNeXt-Nano U-Net for modern lightweight feature extraction.
   - **Pipeline 3:** SegFormer-B3 Vision Transformer (with multi-stage unfreezing) for global context modeling.
   - **Pipeline 4:** Probability-Weighted Ensemble Models.
   - **Pipelines 5–7:** Geodesic Morphological Reconstruction models guided by seed and boundary probability maps.
4. **Topological Evaluation Metrics:** Evaluates performance using centerline Dice score (`clDice` @ 2.5px tolerance), skeleton recall, skeleton precision, and whole-image instance-level recovery counts.

---

## Tech Stack & Dependencies
- **Language:** Python
- **Environment Management:** `uv`
- **Deep Learning Framework:** `PyTorch`, `segmentation-models-pytorch`, `timm`, `torchinfo`
- **Image Processing & Computer Vision:** `OpenCV`, `scikit-image`, `scipy`, `Albumentations`
- **Data Manipulation & Plotting:** `numpy`, `matplotlib`, `tqdm`

---

## Project Structure
```text
├── notebooks/          # Jupyter notebooks for model training and inference
├── pyproject.toml      # Local uv dependency configuration file
├── requirements.txt    # Cloud/Colab environment compatibility file
└── README.md           # Project documentation
