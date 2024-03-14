import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load the Keras model
model = load_model("my_model.keras")

# Define the class labels
class_labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def predict_image(image, model):
    # Preprocess the image
    img = image.resize((224, 224))  # Resize the image to match the input size of the model
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values

    # Predict the class probabilities
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]

    return predicted_class, img

def main():
    st.title('Eye Disease Prediction')

    # Display a file uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # If an image is uploaded
    if uploaded_file is not None:
        # Read the image
        img = Image.open(uploaded_file)

        # Predict the class of the uploaded image
        predicted_class, predicted_img = predict_image(img, model)

        # Create two columns to display images side by side
        col1, col2 = st.columns(2)

        # Display the original image in the first column
        with col1:
            st.image(img, caption='Uploaded Image', use_column_width=True)

        # Draw the predicted class on the predicted image
        draw = ImageDraw.Draw(predicted_img)
        font = ImageFont.load_default()
        draw.text((10, 10), f"Predicted Image: {predicted_class}", fill=(255, 255, 255), font=font)

        # Resize the predicted image to make it smaller
        predicted_img_resized = predicted_img.resize((224, 224))

        # Display the predicted image in the second column
        with col2:
            st.image(predicted_img_resized, caption=f'Predicted Image: {predicted_class}', use_column_width=True)

if __name__ == "__main__":
    main()

