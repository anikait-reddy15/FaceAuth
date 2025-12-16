# FaceAuth: Deep Learning Facial Authentication System

![Project Banner](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-GPU-orange)

FaceAuth is a secure, biometrics-based authentication system built from scratch using **TensorFlow** and **Computer Vision**. Unlike traditional classification models that require retraining for every new user, FaceAuth utilizes a **Siamese Network** architecture (One-Shot Learning) to generate 128-dimensional embeddings. This allows the system to recognize registered users immediately without retraining the entire model.

---

## 📑 Table of Contents
1.  [Introduction](#-introduction)
2.  [The Approach (Architecture)](#-the-approach-architecture)
3.  [Dataset & Preprocessing](#-dataset--preprocessing)
4.  [Prerequisites](#-prerequisites)
5.  [Installation](#-installation)
6.  [Training the Model](#-training-the-model)
7.  [Testing & Usage](#-testing--usage)
8.  [Performance & Results](#-performance--results)
9.  [Future Scope](#-future-scope)

---

## 📖 Introduction

Standard Convolutional Neural Networks (CNNs) classify images into fixed categories (e.g., "Person A" vs. "Person B"). In a real-world login system, this is impractical because adding a new user would require retraining the entire network.

**FaceAuth solves this by learning "Similarity" instead of "Identity".** It maps facial features to a 128-dimensional vector space where images of the same person are mathematically close, and images of different people are far apart.

---

## 🏗 The Approach (Architecture)

We implemented a **Siamese Neural Network (SNN)**. This architecture consists of two identical subnetworks that share the exact same weights and parameters.



### 1. The Triplet Loss Strategy
The model trains on "Triplets" of images:
* **Anchor (A):** The baseline image of a person.
* **Positive (P):** Another image of the *same* person.
* **Negative (N):** An image of a *different* person.

**The Goal:** Minimize the distance between $(A, P)$ and maximize the distance between $(A, N)$.

### 2. The Embedding Model ("The Brain")
We designed a custom VGG-style Convolutional Network optimized for the **RTX 3060 Laptop GPU**:

* **Input Layer:** 100x100x3 RGB Images.
* **Convolutional Blocks:** 4 blocks of Conv2D (filters: 64 $\to$ 128 $\to$ 256 $\to$ 256) with ReLU activation and Max Pooling.
* **Global Average Pooling:** Replaces the traditional "Flatten" layer. This reduces the parameter count from ~150 Million to ~200,000, preventing VRAM crashes and making the model translation-invariant.
* **Output Layer:** A Dense layer with `Sigmoid` activation, outputting a normalized 128-dimensional embedding vector (values between 0 and 1).

---

## 📂 Dataset & Preprocessing

We utilized the **Labeled Faces in the Wild (LFW)** dataset for training.

* **Download Link:** [LFW Dataset (Deep Funneled Version)](http://vis-www.cs.umass.edu/lfw/)
* **Structure:** The dataset contains 13,000+ images of faces collected from the web.

### Data Augmentation Pipeline
To solve the common issue where webcam lighting differs from dataset lighting, we implemented a GPU-accelerated augmentation pipeline:
* **Random Brightness:** $\pm 0.2$ delta.
* **Random Contrast:** $0.8x$ to $1.2x$.
* **Random Saturation:** $0.8x$ to $1.2x$.
* *Note:* Augmentation is applied only to Anchor and Positive images to force the model to learn robust features.

---

## 💻 Prerequisites

Ensure your system meets the following requirements before installation:

* **Operating System:** Windows 10/11 (Native Support) or Linux.
* **Python Version:** Python 3.8, 3.9, or 3.10 (Required for TensorFlow 2.10).
* **Hardware:** NVIDIA GPU (RTX Series recommended) for training.
* **Drivers:** * [CUDA Toolkit 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)
    * [cuDNN SDK 8.1](https://developer.nvidia.com/rdp/cudnn-archive)

---

## ⚙ Installation

1.  **Clone the Repository**
    ```cmd
    git clone [https://github.com/yourusername/FaceAuth.git](https://github.com/yourusername/FaceAuth.git)
    cd FaceAuth
    ```

2.  **Create a Virtual Environment**
    It is highly recommended to use a virtual environment to avoid dependency conflicts.
    ```cmd
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```cmd
    pip install -r requirements.txt
    ```

4.  **Verify GPU Support**
    Run this quick check to ensure TensorFlow sees your GPU:
    ```cmd
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

---

## 🧠 Training the Model

The training script handles data loading, augmentation, and the training loop.

**Run via Command Prompt:**
```cmd
python Model/train_final.py
```

## 🧪 Testing & Usage

We use a real-time verification script that utilizes **OpenCV** to capture live video and compare it against a captured anchor.

**Run the Test:**
```cmd
python real_time_test_v2.py
```
**Controls**

* **Press A:** Capture your Anchor face (The reference identity).
* **Press T:** Capture a Test face (The face to verify).
* **Press Esc or Q:** Quit the application.

## 📊 Performance & Results

Based on extensive testing with the Sigmoid activation model and Margin=1.0:

| Scenario | Distance Score | Verdict |
| :--- | :--- | :--- |
| **Same Person** | 0.00 - 0.30 | ✅ MATCH |
| **Different Person** | 0.31 - 1.50+ | ❌ NO MATCH |

## 🔮 Future Scope

* **Liveness Detection:** Implement "Blink Detection" or "Head Pose Estimation" to prevent photo spoofing attacks (where a user holds up a photo of the authorized person).
* **Database Integration:** Replace the manual "Anchor" capture with a persistent SQL/MongoDB database of user embeddings.
* **Frontend UI:** Develop a user-friendly frontend using React or Electron for a seamless login experience.
