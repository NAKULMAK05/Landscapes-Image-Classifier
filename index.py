import streamlit as st
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

 
model = load_model('resnet91.keras')
model.training = False

# Class names corresponding to the model's output
class_names = ['Coasts', 'Deserts', 'Forests', 'Glacier', 'Mountains']


# App header with a clean, professional design
st.title("🌍Landscape Image Classifier")
st.markdown("""
Welcome to the **AI-powered Landscape Classifier** for industry professionals.  
Upload a landscape image and our model will predict its category with detailed confidence scores for each class.  
The classifier provides insights into why a landscape is classified into a particular category, making it suitable for business or research analysis.
""")
# File uploader for image
uploaded_file = st.file_uploader("🖼️ Upload a landscape image...", type=["jpg", "jpeg", "png"])

# If no image is uploaded, show placeholder image
if uploaded_file is None:
    st.image("placeholder.jpg", caption="Please upload a landscape image.", use_container_width=True)

# When an image is uploaded
if uploaded_file is not None:
    # Load and preprocess the image
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Display the uploaded image in a professional manner
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Classify button with a loading spinner for real-time feedback
    if st.button("🔍 Classify Image"):
        st.markdown("### ⏳ Analyzing the image... Please wait.")
        
        with st.spinner('Running model prediction...'):
            time.sleep(1)  # Simulate a delay
            # Model prediction
            # prediction = model.predict(img_array)
            # score = tf.nn.softmax(prediction[0])
            # predicted_class_idx = np.argmax(score)
            # predicted_class = class_names[predicted_class_idx]
            # confidence_level = 100 * np.max(score)
            #score = tf.nn.softmax(prediction[0])

            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction)
            predicted_class = class_names[predicted_class_idx]
            confidence_level = prediction[0][predicted_class_idx] * 100

        # Display results in a professional manner
        st.success(f"### 🏞️ Predicted Class: **{predicted_class}**")
        st.write(f"### 🔑 Confidence Level: **{confidence_level:.2f}%**")

        # Detailed explanation section
        # helps user to understand why this class is predicted by the Landscape Images Classifier Model
        st.markdown("### 🧐 Why this prediction?")
        explanations = {
            "Coasts": "Features like water bodies, sandy beaches, and coastal elements are prevalent in coastal landscapes.",
            "Deserts": "The model detected dry, sandy terrains typical of deserts, including sand dunes and low vegetation.",
            "Forests": "Forests are characterized by dense tree coverage, greenery, and natural woodlands, all of which were identified.",
            "Glacier": "The icy, snow-covered landscape, reflecting cold tones and glacial structures, led to this prediction.",
            "Mountains": "Mountain landscapes often include rocky terrain, elevated peaks, and rugged cliffs, all detected in the image."
        }
        st.write(explanations.get(predicted_class, "No detailed explanation available for this class."))

        confidence_scores = prediction[0] * 100
# Assuming 'prediction' contains raw logits or probabilities from the model
        confidence_scores = [100 * score for score in prediction[0]]  # Convert to percentages

        confidence_data = pd.DataFrame({
            'Class': class_names,
            'Confidence (%)': confidence_scores
        })


        # Display a bar chart of confidence levels for each class
        st.markdown("### 📊 Confidence Levels for All Classes")
        st.table(confidence_data)

# Optional: Plot a bar chart for better visualization
        st.bar_chart(confidence_data.set_index('Class'))
        

    
        # Add success message at the end with clean typography
        st.markdown("---")
        st.markdown(f"🎉 **The image is classified as {predicted_class} with a confidence of {confidence_level:.2f}%**. Use this insight for your next analysis!")
        
        # Display progress for each class in detail with progress bars
        # st.markdown("### 💡 Confidence Breakdown (Progress Bars)")
        # for i, class_name in enumerate(class_names):
        #     st.write(f"- **{class_name}:** {100 * score[i]:.2f}%")
        #     st.progress(int(100 * score[i]))

