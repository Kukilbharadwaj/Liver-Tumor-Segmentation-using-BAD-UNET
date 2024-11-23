
# Liver Tumor Segmentation Using BAD-UNet

This repository contains the implementation of a **BAD-UNet** model for segmenting liver tumors, merging the functionality of BADNet and UNet architectures. The model is trained on the **3D-IRCADb-01 dataset** provided by [IRCAD](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/).

By leveraging state-of-the-art data preprocessing techniques and a robust encoder-decoder architecture, this project achieves high accuracy and reliable segmentation with a **96% accuracy** and an **88 Dice coefficient**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project demonstrates an effective approach to segmenting liver tumors by combining:
- **BADNet:** A network optimized for medical image segmentation.
- **UNet:** A well-known encoder-decoder architecture for biomedical image analysis.

The 3D-IRCADb dataset is processed into 2D slices, preprocessed, and augmented to train the model efficiently.

---

## Dataset

The dataset used for this project is **3D-IRCADb-01**, available for download [here](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/).

---

## Data Preprocessing

The raw 3D dataset is converted into 2D slices for efficient processing and training. The following preprocessing steps are applied:
1. **Data Augmentation:** Techniques such as flipping, rotation, and scaling are used to increase the diversity of training data.
2. **Hounsfield Windowing:** Standardizes pixel values to focus on relevant intensities for liver tissues.
3. **Histogram Equalization:** Enhances contrast in the images.
4. **Normalization:** Scales pixel values to a uniform range for consistent input to the model.

---

## Model Architecture

The BAD-UNet model merges the best features of BADNet and UNet, forming a robust encoder-decoder architecture. Key features include:
- **Encoder:** Extracts features from the input image using convolutional layers.
- **Decoder:** Reconstructs the segmentation map from the encoded features.
- **Skip Connections:** Ensures fine-grained details are retained during reconstruction.

---

## Training and Evaluation

- **Training:** The model is trained using the processed dataset with a carefully designed loss function to maximize segmentation accuracy.
- **Metrics:**
  - **Accuracy:** 0.9660
  - **Dice Coefficient:** 0.8817
  - **loss** 0.0613

---

## Results

The BAD-UNet model demonstrates:
- High segmentation performance, achieving precise delineation of liver tumors.
- Robust generalization to unseen data.

---
