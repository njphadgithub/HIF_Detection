
import streamlit as st
import numpy as np
import pywt
import cv2
import tensorflow as tf

# --- CONFIGURATION ---
FS = 10000       # Sampling Frequency (10 kHz)
IMG_SIZE = 64    # Scalogram image size for CNN
SCALES = np.arange(1, 129) # Wavelet scales (frequencies)

# --- Load the trained model ---
@st.cache_resource
def load_hif_model():
    model = tf.keras.models.load_model('hif_model.h5')
    return model

model = load_hif_model()

# --- Signal Preprocessing Function ---
def preprocess_signal_to_scalogram(raw_signal):
    if len(raw_signal) == 0:
        return np.zeros((1, IMG_SIZE, IMG_SIZE, 1)) # Return an empty image placeholder

    coeffs, freqs = pywt.cwt(raw_signal, SCALES, 'mexh', sampling_period=1/FS)
    scalogram = np.abs(coeffs)
    scalogram_resized = cv2.resize(scalogram, (IMG_SIZE, IMG_SIZE))

    scalogram_min = scalogram_resized.min()
    scalogram_max = scalogram_resized.max()
    if (scalogram_max - scalogram_min) == 0:
        scalogram_norm = np.zeros_like(scalogram_resized)
    else:
        scalogram_norm = (scalogram_resized - scalogram_min) / (scalogram_max - scalogram_min)

    return scalogram_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# --- Streamlit App Interface ---
st.set_page_config(layout="wide", page_title="HIF Detection App")

st.title("High Impedance Fault (HIF) Detection App")
st.write("Upload a CSV file containing 1D signal data to classify it as Normal, Capacitor Switching, or HIF.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        signal_data = np.loadtxt(uploaded_file, delimiter=',')
        st.success("File successfully uploaded and loaded.")

        if signal_data.ndim > 1:
            st.warning("Multi-dimensional data detected. Taking the first column as the signal.")
            raw_signal = signal_data[:, 0]
        else:
            raw_signal = signal_data

        st.subheader("Original Signal Preview")
        st.line_chart(raw_signal)

        with st.spinner('Processing signal and making prediction...'):
            scalogram_input = preprocess_signal_to_scalogram(raw_signal)
            predictions = model.predict(scalogram_input)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class_idx] * 100

            class_names = ['Normal Grid Signal', 'Capacitor Switching (Transient)', 'High Impedance Fault (Arcing/Distorted)']
            predicted_class_name = class_names[predicted_class_idx]

            st.subheader("Prediction Results:")
            st.write(f"**Predicted Class:** {predicted_class_name}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            st.subheader("Generated Scalogram:")
            st.image(scalogram_input.squeeze(), caption='Generated Scalogram', use_column_width=True, clamp=True, channels='GRAY')

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Please ensure your CSV contains a single column of numerical data.")


else:
    st.info("Upload a CSV file to get started.")

