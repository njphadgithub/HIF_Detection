# High Impedance Fault (HIF) Detection Streamlit App

This application demonstrates a Deep Learning-based solution for detecting High Impedance Faults (HIFs) in power distribution grids. It utilizes Continuous Wavelet Transform (CWT) to convert 1D signal data into 2D scalograms, which are then classified by a Convolutional Neural Network (CNN).

## Features
- Upload CSV files containing 1D signal data.
- Visualize the uploaded signal.
- Preprocess the signal into a scalogram using CWT.
- Classify the signal as 'Normal', 'Capacitor Switching', or 'HIF' using a pre-trained CNN.
- Display the predicted class and confidence.
- Show the generated scalogram image.

## Setup and Local Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd hif-detection-app
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The app will open in your web browser.

## Deployment on Render

This application can be easily deployed on Render.com as a Web Service.

### 1. Repository Setup
Ensure your project (including `app.py`, `hif_model.h5`, and `requirements.txt`) is pushed to a Git repository (GitHub, GitLab, or Bitbucket).

### 2. Create a New Web Service on Render
- Go to your Render Dashboard.
- Click 'New' -> 'Web Service'.
- Connect your Git repository.

### 3. Configuration
Use the following settings for your Render Web Service:
-   **Build Command:** `pip install -r requirements.txt`
-   **Start Command:** `streamlit run app.py --server.port $PORT --server.enableCORS false --server.enableXsrfProtection false`
-   **Runtime:** Python 3
-   **Instance Type:** Free (for testing) or choose a suitable paid tier.

### 4. Environment Variables
No specific environment variables are strictly required for this app, but you might add them if needed for future enhancements.

### 5. Deploy
Click 'Create Web Service' and Render will automatically build and deploy your application.

### 6. Notes on `hif_model.h5`
Ensure that `hif_model.h5` is committed to your repository. For larger models or production environments, consider storing the model in an object storage service (like AWS S3 or Google Cloud Storage) and downloading it during the Render build process, rather than committing it directly to Git.

## Model Details

The `hif_model.h5` file contains a pre-trained Convolutional Neural Network (CNN) based on TensorFlow/Keras. It was trained to classify 1D time-series signals (after CWT transformation into scalograms) into three categories:
-   0: Normal Grid Signal
-   1: Capacitor Switching (Transient)
-   2: High Impedance Fault (Arcing/Distorted)

The model expects input scalograms of size (1, 64, 64, 1), representing a single grayscale image with a channel dimension.

## Dependencies
- `streamlit`
- `numpy`
- `pywavelets`
- `opencv-python`
- `tensorflow`

See `requirements.txt` for specific versions.
