from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('model1.h5')

# Class names from your dataset
labels = ['Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha',
          'Avacado', 'Bamboo', 'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor',
          'Curry_Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Gauva', 'Geranium', 'Henna',
          'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon', 'Lemon_grass', 'Mango',
          'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 'Pappaya', 'Pepper',
          'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel']

# Function to make predictions on uploaded images
def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.sigmoid(predictions[0])
    predicted_class = labels[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

# Function to convert the image to different formats
def convert_image_format(image, format_type):
    if format_type == 'Grayscale':
        return image.convert('L')  # Convert to grayscale
    elif format_type == 'Black and White':
        return image.convert('1')  # Convert to black and white
    elif format_type == 'Sepia':
        return apply_sepia(image)
    elif format_type == 'Invert Colors':
        return apply_invert_colors(image)
    # Add more format options as needed

def apply_sepia(image):
    # Apply custom sepia tone effect
    sepia_filter = [
        0.393, 0.769, 0.189,
        0.349, 0.686, 0.168,
        0.272, 0.534, 0.131,
    ]

    sepia_image = ImageOps.colorize(image.convert('L'), "#704214", "#C0A080")
    
    return sepia_image

def apply_invert_colors(image):
    # Invert colors
    inverted_image = ImageOps.invert(image.convert("RGB"))
    return inverted_image

# Streamlit app
st.title("Plant Identification App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Original Image", use_column_width=True)
    st.write("")


    temp_file_path = "temp_img.jpg"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getvalue())

    predicted_class, confidence = predict_image(temp_file_path)

    st.title(f"Prediction of is Image: {predicted_class}")
    st.title(f"Confidence of Image is: {confidence:.2f}%")

    # Open the uploaded image using PIL
    img = Image.open(temp_file_path)

    # Display formatted images in two columns
    format_options = ['Grayscale', 'Black and White', 'Sepia', 'Invert Colors']  # Add more formats as needed

    cols = st.columns(1)

    for col in cols:
        for format_option in format_options:
            col.image(convert_image_format(img, format_option), caption=f"{format_option} Image", use_column_width=True)
