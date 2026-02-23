import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
import pathlib
import sys
import warnings
warnings.filterwarnings('ignore')

# Force TensorFlow to use CPU only (to avoid GPU errors)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ============================================
# Configuration
# ============================================
IMG_SIZE = 224
CURRENT_DIR = pathlib.Path(__file__).parent

print("="*50)
print("🚀 NEURO AI APP STARTING...")
print("="*50)
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {sys.version}")
print(f"Files in directory: {os.listdir('.')}")
print("="*50)

# Find model file
h5_files = list(CURRENT_DIR.glob("*.h5"))
if not h5_files:
    raise FileNotFoundError("No .h5 model file found!")
    
MODEL_PATH = h5_files[0]
print(f"✅ Using model: {MODEL_PATH.name}")
print(f"✅ Model size: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")

# ============================================
# Load Model with compatibility settings
# ============================================
print("\n🔄 Loading model with compatibility mode...")

try:
    # Method 1: Try loading with compile=False first
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        compile=False,
        custom_objects={'tf': tf}  # Add custom objects for compatibility
    )
    print("✅ Model loaded successfully with compile=False")
    
except Exception as e:
    print(f"⚠️ First loading method failed: {str(e)}")
    
    try:
        # Method 2: Try with safe_mode=False
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
        print("✅ Model loaded with safe_mode=False")
        
    except Exception as e2:
        print(f"⚠️ Second loading method failed: {str(e2)}")
        
        try:
            # Method 3: Load weights into new model
            from tensorflow.keras.applications import VGG16
            from tensorflow.keras import layers, models
            
            print("🔄 Rebuilding model architecture...")
            
            # Recreate the exact architecture from your Colab
            base_model = VGG16(
                weights=None,  # Don't load ImageNet weights
                include_top=False,
                input_shape=(IMG_SIZE, IMG_SIZE, 3)
            )
            
            # Build the model (same as your Colab)
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(4, activation='softmax')  # 4 classes
            ])
            
            # Load just the weights
            model.load_weights(MODEL_PATH)
            print("✅ Model weights loaded successfully!")
            
        except Exception as e3:
            print(f"❌ All loading methods failed: {str(e3)}")
            raise RuntimeError(f"Could not load model: {str(e3)}")

print(f"📊 Model input shape: {model.input_shape}")
print(f"📊 Model output shape: {model.output_shape}")

# ============================================
# Class names
# ============================================
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
display_names = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary Tumor'
}

# Tumor information
tumor_info = {
    'glioma': {
        'name': '🚨 Glioma',
        'description': 'High-grade brain tumor that grows rapidly and requires aggressive treatment immediately.',
        'urgency': 'IMMEDIATE ACTION REQUIRED',
        'color': '#dc3545',
        'details': '• Aggressive tumor\n• Rapidly growing\n• Requires immediate surgery/radiation'
    },
    'meningioma': {
        'name': '⚠️ Meningioma',
        'description': 'Often slow-growing but can compress brain tissue causing severe neurological damage.',
        'urgency': 'Schedule Specialist Consult',
        'color': '#fd7e14',
        'details': '• Usually benign\n• Slow growing\n• May cause pressure on brain'
    },
    'pituitary': {
        'name': '🔍 Pituitary Tumor',
        'description': 'Can cause blindness or hormonal failure if not detected and treated surgically.',
        'urgency': 'Urgent Endocrinology Consult',
        'color': '#0d6efd',
        'details': '• Affects hormone production\n• Can impact vision\n• Often treatable with surgery'
    },
    'notumor': {
        'name': '✅ No Tumor',
        'description': 'Normal brain tissue. No tumor detected.',
        'urgency': 'Regular Checkups Recommended',
        'color': '#198754',
        'details': '• Healthy brain tissue\n• No abnormalities detected\n• Continue routine screening'
    }
}

# ============================================
# Preprocessing Function
# ============================================
def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ============================================
# Prediction Function
# ============================================
def predict_tumor_type(image):
    """Predict tumor type from MRI image"""
    try:
        if image is None:
            return "<div style='color: red; text-align: center;'>⚠️ Please upload an image first.</div>", pd.DataFrame()
        
        # Preprocess image
        processed_img = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)[0]
        
        # Get top prediction
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100
        
        # Get info for predicted class
        info = tumor_info.get(predicted_class, {
            'name': predicted_class,
            'description': 'Consult a specialist.',
            'urgency': 'Further evaluation needed',
            'color': '#6c757d',
            'details': 'Please consult with a medical professional.'
        })
        
        # Create HTML result
        result_html = f'''
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: white; border-radius: 15px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="background-color: {info['color']}; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
                    <h1 style="margin: 0; font-size: 32px;">{info['name']}</h1>
                </div>
                
                <div style="margin-bottom: 25px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-weight: bold; color: #333;">Confidence:</span>
                        <span style="font-weight: bold; color: {info['color']};">{confidence:.2f}%</span>
                    </div>
                    <div style="background-color: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden;">
                        <div style="background-color: {info['color']}; width: {confidence}%; height: 100%;"></div>
                    </div>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #333; margin-bottom: 8px;">Urgency:</h4>
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                        <span style="color: {info['color']}; font-weight: bold; font-size: 18px;">{info['urgency']}</span>
                    </div>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #333; margin-bottom: 8px;">Description:</h4>
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; color: #333;">
                        {info['description']}
                    </div>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #333; margin-bottom: 8px;">Clinical Details:</h4>
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; color: #333; white-space: pre-line;">
                        {info['details']}
                    </div>
                </div>
                
                <div style="margin-top: 25px; padding-top: 15px; border-top: 1px solid #dee2e6; font-size: 12px; color: #666; text-align: center;">
                    ⚕️ AI Assistant Tool - For demonstration purposes only.
                </div>
            </div>
        </div>
        '''
        
        # Create probability dataframe
        prob_data = []
        for i, name in enumerate(class_names):
            prob_data.append({
                'Tumor Type': display_names.get(name, name),
                'Confidence (%)': f"{predictions[i]*100:.2f}%"
            })
        
        prob_df = pd.DataFrame(prob_data)
        prob_df = prob_df.sort_values('Confidence (%)', ascending=False)
        
        return result_html, prob_df
        
    except Exception as e:
        error_html = f'''
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 25px; border-radius: 10px; text-align: center;">
                <h2 style="margin-top: 0;">❌ Error</h2>
                <p>{str(e)}</p>
            </div>
        </div>
        '''
        return error_html, pd.DataFrame()

# ============================================
# Create Gradio Interface
# ============================================
with gr.Blocks(title="🧠 Neuro AI Detector", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🧠 Neuro AI Multi-Class Cancer Detector
    ### Glioma · Meningioma · Pituitary · No Tumor
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Brain MRI", height=400)
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        with gr.Column(scale=1):
            result_html = gr.HTML(
                value="<div style='text-align: center; padding: 50px;'>Awaiting Analysis</div>"
            )
    
    prob_df = gr.Dataframe(
        headers=["Tumor Type", "Confidence (%)"],
        label="Confidence Scores"
    )
    
    analyze_btn.click(
        fn=predict_tumor_type,
        inputs=input_image,
        outputs=[result_html, prob_df]
    )
    
    clear_btn.click(
        fn=lambda: ("<div style='text-align: center; padding: 50px;'>Awaiting Analysis</div>", None),
        outputs=[result_html, prob_df]
    )

# ============================================
# Launch
# ============================================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
