# 🔬 Corrosion Detection on Metal Surfaces (CNN)

This repository contains the implementation of a deep learning model for the **early and accurate detection and classification of corrosion** on static metal surfaces. The system replaces traditional, subjective visual inspections with an automated, high-accuracy computer vision solution.

---

## ✨ Key Features & Objectives

* **High-Accuracy Classification:** Developed a **Convolutional Neural Network (CNN)** to perform binary classification (Corrosion vs. Non-Corroded).
* **Performance:** The optimized model achieved a classification **accuracy of 98.75%** on the validation dataset.
* **Interpretability (Grad-CAM):** Provides **Gradient-weighted Class Activation Mapping (Grad-CAM)** visualizations to highlight the specific regions of rust that influenced the model's prediction.
* **Target Industry:** Designed to increase structural safety and reduce maintenance costs in critical sectors like aerospace, marine, and oil & gas.

---

## 💻 Model Architecture & Technology

The corrosion detection pipeline is built around an optimized CNN designed for spatial feature learning.

### Model Details
* **Architecture:** The model consists of three convolutional layers with **ReLU activation**, max pooling, **dropout (0.25)**, and batch normalization, followed by two dense layers.
* **Optimization:** Employed the Adam optimizer and Early Stopping to prevent overfitting.

### Technology Stack
| Component | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.10 | Core programming language. |
| **TensorFlow/Keras** | 2.12+ | Model building and training for the CNN. |
| **OpenCV** | 4.5+ | Image preprocessing (noise removal, resizing). |
| **scikit-learn** | 1.2+ | Performance evaluation (accuracy, confusion matrix). |
| **Grad-CAM** | Custom | Provides localized explanations for model predictions. |

---

## 📊 Results Overview

The optimized CNN model significantly outperformed the basic architecture.

| Metric | Basic CNN | Optimized CNN |
| :--- | :--- | :--- |
| **Accuracy (%)** | 94.50 | **98.75** |
| **Precision (%)** | 100.00 | **99.60** |
| **Recall (%)** | 91.20 | **98.40** |
| **Loss Value** | 0.3155 | **0.0986** |

---

## ⚙️ Data and Setup

* **Dataset:** Static Dataset of **2000+ images** manually labeled (corrosion / no-corrosion).
* **Resolution:** All images were resized to $224 \times 224$ pixels.
* **Data Source:** Images were web scraped from corrosion image databases and public sources.

### Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/the-rajanand/Corrosion-Detection.git](https://github.com/the-rajanand/Corrosion-Detection.git)
    cd Corrosion-Detection
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Execution Flow

The project is executed sequentially across the following Jupyter Notebooks:

1.  **Data Acquisition:** Run `1) Web Scraping.ipynb` to gather the necessary image datasets.
2.  **Preprocessing & Augmentation:** Run `2) Image Preprocessing and Augmentation.ipynb` to clean the data and prepare it for the model.
3.  **Data Split:** Run `3) Data Split Generation.ipynb` to divide the dataset into training, validation, and test sets.
4.  **Model Training & Comparison:** Run `4) Model Making 1 and 2.ipynb` and then `5) Model Making 3.ipynb` to train and evaluate the basic and optimized CNN architectures.

---

## 👤 Author

* **Raj Anand**
