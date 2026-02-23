# 🧠 Neuro AI Multi-Class Cancer Detector

**Hackathon Project: HEC Generative AI Training – Cohort 2**

### 👥 Team Members:
- Hafiz Muhammad Asnan Amar
- Narmeen Bilal
- Babar Khan
- Rehan Shafique
- Afshan Batool
- Sahar Ejaz

## 📋 Project Overview
Neuro AI is a deep learning-based multi-class cancer detector for brain MRI images.
It can classify images into **Glioma, Meningioma, Pituitary Tumor, or No Tumor** categories.
The project is deployed on Hugging Face, tested on Google Colab, and used Gradio for an interactive user interface.

## 🎯 Problem Statement
Early detection of brain tumors is critical for effective treatment.
This tool provides an AI-powered solution to assist medical professionals and students in quickly identifying tumor types from MRI scans.

## ✨ Features
- 🧠 Multi-class tumor prediction (4 classes)
- 📊 Confidence scores for each prediction
- 🎨 Styled HTML result with urgency and clinical details
- 🖱️ Simple Gradio interface
- 🤗 Deployed on Hugging Face Spaces

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- Gradio
- NumPy, Pandas, Pillow
- Hugging Face (for deployment)
- Model file: `.h5` format

## 🧪 Model Details
- **Architecture:** VGG16 with custom classification head
- **Input:** 224x224 MRI image
- **Output:** Predicted class + confidence score
- **Classes:**
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor

### 🔗 Links
- **Live Demo:** [Neuro AI Detector on Hugging Face](https://huggingface.co/spaces/narmeenbilal/Neuro-AI-Detector)
- **Model File:** [Download best_multi_class_model.h5](your-model-file-link-here)

## 📦 Installation

   ```bash
   git clone [https://github.com/narmeen-hub/neuro-ai-cancer-detector.git]
   cd neuro-ai-cancer-detector
   pip install -r requirements.txt
