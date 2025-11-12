# CamoXpert: Dynamic Neural Networks for Adaptive Camouflaged Object Detection

## 📄 Overview

**CamoXpert** is a novel dynamic neural network architecture designed for camouflaged object detection (COD). It addresses the challenge of detecting objects deliberately concealed within their surroundings by introducing dynamic routing, multi-scale expert modules, and bi-level feature fusion. The architecture is lightweight, efficient, and achieves state-of-the-art performance on COD benchmarks.

---

## 🚀 Key Features

- **Dynamic Mixture-of-Experts (MoE) Routing**: Adaptive selection of specialized processing paths based on input characteristics.
- **Multi-Scale Expert Modules**: Texture, attention, and hybrid experts for comprehensive feature extraction.
- **Bi-Level Feature Fusion**: Combines low-level details and high-level semantics for better object detection.
- **EdgeNeXt Backbone**: Lightweight CNN-Transformer hybrid optimized for edge devices.
- **Real-Time Inference**: Achieves >30 FPS on mobile devices.

---

## 📊 Benchmarks

CamoXpert has been evaluated on the following datasets:
- **COD10K**
- **NC4K**
- **CAMO**

It achieves state-of-the-art performance in terms of F-measure, S-measure, and IoU.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (optional for GPU acceleration)

### Install Dependencies
1. Clone the repository:
   ```bash
   git clone https://github.com/mahi-chan/CamoXpert.git
   cd CamoXpert