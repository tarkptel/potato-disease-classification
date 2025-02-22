import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Load Model
MODEL = tf.keras.models.load_model("new_cnn_model_tf.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Function to process and predict
def predict(image):
    # Convert image to numpy array
    image = np.array(image)
    
    # Expand dimensions to match model input shape
    img_batch = np.expand_dims(image, 0)
    
    # Make prediction
    predictions = MODEL.predict(img_batch)
    
    # Get class and confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return f"Prediction: {predicted_class} (Confidence: {confidence:.2f})"

# Create Gradio Interface
iface = gr.Interface(
    fn=predict,               # Function to call
    inputs=gr.Image(type="pil"),  # Image input (PIL format)
    outputs="text",           # Text output (prediction result)
    title="Potato Disease Classifier",
    description="Upload an image of a potato leaf, and the model will predict whether it's Healthy, Early Blight, or Late Blight."
)

# Launch the app
iface.launch()
