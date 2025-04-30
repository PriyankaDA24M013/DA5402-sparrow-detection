import streamlit as st
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

API_ENDPOINT = "http://localhost:8000/detect/"

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
    
    uploaded_file = st.file_uploader(
        "Upload an image of house sparrows", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Sparrows"):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_ENDPOINT, files=files)
            
            if response.status_code == 200:
                detections = response.json()["detections"]
                fig = draw_boxes(np.array(image), detections)
                st.subheader("Detection Results")
                st.pyplot(fig)
                
            else:
                st.error("Error in detection. Please try again.")

if __name__ == "__main__":
    main()
