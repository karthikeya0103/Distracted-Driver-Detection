import streamlit as st # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Load the trained model
@st.cache_resource
def load_driver_model():
    return load_model('model/distracted_driver.h5')

model = load_driver_model()

# Define driver action labels
driver_action = {
    0: 'normal driving',
    1: 'texting - right',
    2: 'talking on the phone - right',
    3: 'texting - left',
    4: 'talking on the phone - left',
    5: 'operating the radio',
    6: 'drinking',
    7: 'reaching behind',
    8: 'hair and makeup',
    9: 'talking to passenger'
}

# Prediction function
def predict(pil_image):
    img = pil_image.convert("RGB")
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return np.argmax(prediction[0])

# Streamlit UI
st.title("Distracted Driver Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Driver Action"):
        prediction_index = predict(img)
        prediction_label = driver_action[prediction_index]
        st.success(f"Detected Activity: **{prediction_label}**")
