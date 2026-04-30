# Deep Learning Assignment 2: Advanced Architectures in Image Classification and Object Detection

**Author:** Abdullah Naeem  
---

## 📌 Project Overview
This repository contains a comprehensive Jupyter Notebook exploring both foundational and state-of-the-art computer vision architectures. The project is divided into two major tasks:
1. **Image Classification:** Building a customized VGG-16 Convolutional Neural Network (CNN) from scratch to classify high-dimensional images from the CIFAR-100 dataset.
2. **Real-time Object Detection:** Leveraging the modern YOLOv8 (Large) framework to detect, localize, and draw bounding boxes around multiple objects within an image using the COCO8 dataset.

This dual approach provides a deep understanding of how deep learning transitions from assigning a single label to an image (Classification) to pinpointing exact coordinates of multiple objects (Detection).

---

## 🧠 Part 1: VGG-16 Architecture from Scratch (Classification)

### What it does:
The model takes 32x32 pixel images from the CIFAR-100 dataset and classifies them into one of 100 distinct categories (e.g., apples, orchids, trains, etc.). 

### How it does it:
* **Architecture Engineering:** The network is built entirely from scratch using PyTorch (`torch.nn`). It follows the philosophy of the original VGG-16 paper by using a deep stack of small 3x3 convolutional filters, grouped together and separated by Max Pooling layers to progressively reduce spatial dimensions while increasing channel depth (up to 512 channels).
* **Modern Adaptations:** To stabilize the deep network and accelerate convergence, **Batch Normalization (`nn.BatchNorm2d`)** layers were injected after every convolution. **Dropout** layers were added in the fully connected classifier block to prevent the model from memorizing the training data (overfitting).
* **Training Pipeline:** The model is optimized using Stochastic Gradient Descent (SGD) with momentum, running through the dataset while dynamically visualizing progress using `tqdm`.

### Why it does it:
Building VGG-16 from scratch is a foundational exercise. It demonstrates the mathematical nuances of hierarchical feature extraction—how early layers learn simple edges and textures, while deeper layers construct complex semantic representations. 

---

## 🎯 Part 2: YOLOv8-Large Object Detection 

### What it does:
Unlike VGG which looks at an entire image to guess one label, YOLO (You Only Look Once) analyzes the image to find *where* objects are and *what* they are simultaneously, outputting bounding boxes and class probabilities.

### How it does it:
* **Ultralytics Framework:** The project imports the `ultralytics` library to load `yolov8l.pt` (the Large variant of YOLOv8). 
* **Custom Training:** The model is fine-tuned on the `coco8.yaml` dataset for **10 epochs** using a resolution of `imgsz=640`. Data augmentation is enabled to improve model robustness.
* **Algorithmic Logic (Manual NMS):** The notebook includes a custom, vectorized implementation of **Non-Maximum Suppression (NMS)** using PyTorch. When YOLO predicts multiple overlapping bounding boxes for the same object, NMS mathematically filters them out by calculating the Intersection over Union (IoU) and keeping only the box with the highest confidence score.

### Why it does it:
YOLO represents the pinnacle of real-time computer vision. We use it to showcase a "One-Stage Detector," which replaces the computationally expensive region-proposal steps of older models with a single unified neural network pass. This makes it highly suitable for autonomous driving, robotics, and live surveillance.

---

## 🔬 Advanced Analytics & Visualizations

To prove that the models aren't just "black boxes", the notebook includes deep analytical code:
1. **Filter Visualization:** Extracts and plots the actual mathematical weights of the first convolutional layer in VGG, showing the raw filters the network uses to process the input image.
2. **Feature Map Heatmaps:** Extracts intermediate activations from the VGG network and compresses them into a visual heatmap. This answers the "why" of the model by visually highlighting exactly which parts of the original image the neural network is paying attention to.
3. **Speed Benchmarking:** A direct head-to-head comparison measuring the inference time (in milliseconds) of passing a dummy tensor through VGG-16 vs YOLOv8L.

---

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python installed along with the following libraries:
```bash
pip install torch torchvision ultralytics matplotlib tqdm numpy