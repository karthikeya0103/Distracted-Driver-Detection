# 🚗 Distracted Driver Detection using Deep Learning

This project uses a Convolutional Neural Network (CNN) to classify driver behavior and detect distractions in real-time. A Streamlit web app is provided to upload and classify images based on the driver's activity.
dataset link - [Dataset](https://www.kaggle.com/datasets/rightway11/state-farm-distracted-driver-detection)

---

## 📌 Features

- Detects 10 types of driver behaviors
- Built with TensorFlow and Keras
- Interactive web interface using Streamlit
- Real-time prediction from uploaded images
---

## 🧠 Driver Activities Detected

| Label | Activity                      |
|-------|-------------------------------|
| c0    | Normal driving                |
| c1    | Texting - right               |
| c2    | Talking on the phone - right |
| c3    | Texting - left                |
| c4    | Talking on the phone - left  |
| c5    | Operating the radio           |
| c6    | Drinking                      |
| c7    | Reaching behind               |
| c8    | Hair and makeup               |
| c9    | Talking to passenger          |

---


## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- pip
- virtualenv (optional but recommended)

### 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/karthikeya0103/distracted-driver-detection.git
cd distracted-driver-detection
```

2. Create a virtual environment (optional):
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 🖼️ How to Use
- Upload an image showing the driver.

- The model will classify the driver's action.

- View the predicted behavior label on the screen.

---

## 🧠 Model Summary
- Model type: CNN

- Layers: 5 Convolutional layers + Dense layers

- Accuracy: ~98% on validation dataset

---

## 📊 Dataset
Dataset used: State Farm Distracted Driver Detection

The dataset consists of labeled images from dashboard cameras of drivers performing different actions.

---