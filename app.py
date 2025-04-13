import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import pickle

# Load pre-trained VGG16 model for feature extraction
@st.cache_resource
def load_feature_extractor():
    base_model = VGG16(weights='imagenet')  # Specify weights explicitly
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    return model

# Function to preprocess image and extract features
def extract_features(image, model):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

# Load tokenizer with error handling
def load_tokenizer():
    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except FileNotFoundError:
        st.error("Tokenizer file not found. Please ensure 'tokenizer.pkl' exists.")
        return None

# Dummy caption generation (to be replaced with your model's implementation)
def generate_caption(feature, tokenizer):
    if tokenizer is None:
        return "Tokenizer not available."
    return "A caption generated for the uploaded image."

# Streamlit App
st.title("🖼️ Image Caption Generator")
st.write("Upload an image and get an AI-generated caption!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    model = load_feature_extractor()
    features = extract_features(image, model)

    tokenizer = load_tokenizer()
    if tokenizer:
        caption = generate_caption(features, tokenizer)
        st.markdown("### 📢 Generated Caption:")
        st.success(caption)
