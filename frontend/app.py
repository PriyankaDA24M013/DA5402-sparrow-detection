import streamlit as st
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Get API endpoint from environment variables with fallback to Docker service name
API_HOST = os.environ.get("API_HOST", "api")
API_PORT = os.environ.get("API_PORT", "8000")
API_ENDPOINT = f"http://{API_HOST}:{API_PORT}/detect/"

def draw_boxes(image, detections):
    """Draw bounding boxes on image"""
    fig, ax = plt.subplots()
    ax.imshow(image)

    for det in detections:
        box = det["box"]
        score = det["confidence"]
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(xmin, ymin, f"House Sparrow: {score:.2f}",
                 color='white', fontsize=8,
                 bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    return fig


def main():
    st.title("House Sparrow Detection System")
    
    # Display the API endpoint being used (helpful for debugging)
    st.sidebar.info(f"Using API endpoint: {API_ENDPOINT}")
    
    uploaded_file = st.file_uploader(
        "Upload an image of house sparrows", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Sparrows"):
            with st.spinner("Detecting sparrows..."):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(API_ENDPOINT, files=files, timeout=30)
                    
                    if response.status_code == 200:
                        detections = response.json()["detections"]
                        if detections:
                            fig = draw_boxes(np.array(image), detections)
                            st.subheader("Detection Results")
                            st.pyplot(fig)
                            st.success(f"Found {len(detections)} sparrow(s) in the image")
                        else:
                            st.info("No sparrows detected in this image")
                    else:
                        st.error(f"Error in detection (Status code: {response.status_code}). Please try again.")
                        
                except requests.exceptions.ConnectionError:
                    st.error(f"Failed to connect to the API at {API_ENDPOINT}. Please check if the API service is running.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()