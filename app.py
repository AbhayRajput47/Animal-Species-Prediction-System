# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from labels import label_map

# Download Model from Google Drive

import gdown
import os

file_id = "1A9Z1vPbbpZ1A01ZiSysX1Jp-RIlBGY31"

url = f"https://drive.google.com/uc?id={file_id}"

# Download model if not present
if not os.path.exists("animal_classifier_model.h5"):
    gdown.download(
        url,
        "animal_classifier_model.h5",
        quiet=False
    )
# Page Configuration
st.set_page_config(
    page_title="Animal Species Prediction",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 30px;
        background-color: #e0e0e0;
        border-radius: 15px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model_cached():
    return load_model("animal_classifier_model.h5")

model = load_model_cached()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    show_top_n = st.slider("Show Top N Predictions", 1, len(label_map), 3)
    confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 20)
    
    st.divider()
    st.subheader("📖 About")
    st.write("""
    This AI-powered tool identifies animal species from images.
    - **Supported Animals**: Dog, Horse, Elephant, Butterfly, Chicken, Cat, Cow, Sheep, Spider, Squirrel
    - **Model**: Deep Learning CNN
    - **Input Size**: 224×224 pixels (auto-adjusted)
    """)
    
    st.subheader("💡 Tips")
    st.write("""
    - Use clear, well-lit images
    - Include the entire animal in the frame
    - Avoid blurry or low-resolution images
    - Best results with single animals
    """)

# Main content
st.title("🐾 Animal Species Prediction System")
st.markdown("**Identify animals in images using advanced AI**", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🖼️ Predict", "📊 Predictions History", "ℹ️ How to Use"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an animal image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.subheader("Or Capture with Camera")
        camera_image = st.camera_input("Take a photo")
    
    # Process uploaded file
    image_to_process = uploaded_file or camera_image
    
    if image_to_process is not None:
        try:
            # Open image
            if uploaded_file:
                img = Image.open(uploaded_file)
                image_name = uploaded_file.name
            else:
                img = Image.open(camera_image)
                image_name = "Camera Capture"
            
            # Display image
            col_img, col_info = st.columns([1, 1])
            
            with col_img:
                st.image(img, caption=image_name, use_container_width=True)
            
            # Process prediction
            with col_info:
                st.subheader(" Analysis")
                
                # Prepare image for model
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Get predictions
                with st.spinner(" Analyzing image..."):
                    prediction = model.predict(img_array, verbose=0)
                
                # Get top N predictions
                top_indices = np.argsort(prediction[0])[-show_top_n:][::-1]
                
                st.success(f"✅ Analysis Complete!")
                
                # Display top prediction prominently
                top_animal = label_map[top_indices[0]]
                top_confidence = float(prediction[0][top_indices[0]] * 100)
                
                st.markdown(f"""
                ### Primary Prediction
                **{top_animal.upper()}**
                """)
                
                # Confidence visualization
                color = "green" if top_confidence >= confidence_threshold else "orange"
                st.metric("Confidence Score", f"{top_confidence:.1f}%", delta=f"Threshold: {confidence_threshold}%")
                
                # Progress bar for confidence
                st.progress(min(float(top_confidence) / 100, 1.0), text=f"{top_confidence:.1f}%")
                
                # Show all top predictions
                st.subheader("Top Predictions")
                prediction_data = []
                for i, idx in enumerate(top_indices, 1):
                    confidence = float(prediction[0][idx] * 100)
                    animal = label_map[idx]
                    prediction_data.append({
                        "Rank": i,
                        "Animal": animal.capitalize(),
                        "Confidence": f"{confidence:.2f}%"
                    })
                
                st.dataframe(prediction_data, use_container_width=True, hide_index=True)
                
                # Export prediction in multiple formats
                st.subheader(" Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                # JSON Format
                with col1:
                    import json
                    json_data = json.dumps({
                        "primary_prediction": top_animal,
                        "confidence": round(top_confidence, 2),
                        "threshold": confidence_threshold,
                        "top_predictions": prediction_data,
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, indent=2)
                    st.download_button(
                        label="📄 JSON",
                        data=json_data,
                        file_name=f"prediction_{top_animal}.json",
                        mime="application/json"
                    )
                
                # CSV Format
                with col2:
                    csv_data = pd.DataFrame(prediction_data).to_csv(index=False)
                    st.download_button(
                        label="📊 CSV",
                        data=csv_data,
                        file_name=f"prediction_{top_animal}.csv",
                        mime="text/csv"
                    )
                
                # HTML Format
                with col3:
                    html_data = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                            .container {{ max-width: 800px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                            .prediction {{ font-size: 24px; color: #27ae60; font-weight: bold; margin: 15px 0; }}
                            .confidence {{ font-size: 18px; color: #e74c3c; }}
                            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                            th {{ background-color: #3498db; color: white; padding: 10px; text-align: left; }}
                            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                            tr:hover {{ background-color: #ecf0f1; }}
                            .timestamp {{ color: #7f8c8d; font-size: 12px; margin-top: 20px; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>🐾 Animal Prediction Report</h1>
                            <div class="prediction">Predicted Animal: {top_animal.upper()}</div>
                            <div class="confidence">Confidence Score: {top_confidence:.2f}%</div>
                            <h2>Top Predictions</h2>
                            <table>
                                <tr>
                                    <th>Rank</th>
                                    <th>Animal</th>
                                    <th>Confidence</th>
                                </tr>
                    """
                    for item in prediction_data:
                        html_data += f"""
                                <tr>
                                    <td>{item['Rank']}</td>
                                    <td>{item['Animal']}</td>
                                    <td>{item['Confidence']}</td>
                                </tr>
                        """
                    html_data += f"""
                            </table>
                            <div class="timestamp">Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                        </div>
                    </body>
                    </html>
                    """
                    st.download_button(
                        label="🌐 HTML",
                        data=html_data,
                        file_name=f"prediction_{top_animal}.html",
                        mime="text/html"
                    )
                
                # Store in session state for history
                if "predictions" not in st.session_state:
                    st.session_state.predictions = []
                
                if st.button("✅ Save to History"):
                    st.session_state.predictions.append({
                        "animal": top_animal,
                        "confidence": f"{top_confidence:.2f}%",
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success("Saved to history!")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

with tab2:
    st.subheader("📊 Predictions History")
    if "predictions" in st.session_state and st.session_state.predictions:
        df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.predictions = []
            st.rerun()
    else:
        st.info("No predictions saved yet. Make a prediction and save it to history!")

with tab3:
    st.subheader("ℹ️ How to Use")
    st.write("""
    ### Getting Started
    1. **Upload an image** - Click "Choose an animal image" and select a JPG, JPEG, or PNG file
    2. **Or use camera** - Alternatively, use the camera option to capture a live photo
    3. **View results** - The AI will analyze and display predictions with confidence scores
    4. **Review predictions** - Check the top N predictions sorted by confidence
    5. **Save results** - Optionally save predictions to your history
    
    ### Supported Animals
    The model can identify: **Dog, Horse, Elephant, Butterfly, Chicken, Cat, Cow, Sheep, Spider, Squirrel**
    
    ### Best Practices
    - Use **clear, well-lit images**
    - Ensure the **animal is clearly visible**
    - Avoid **blurry or dark images**
    - Include the **entire animal** in the frame
    - Use **high-resolution images** for better accuracy
    
    ### Confidence Score
    - **>80%** - Very High Confidence
    - **60-80%** - High Confidence
    - **40-60%** - Medium Confidence
    - **<40%** - Low Confidence (image may be unclear or not contain a recognizable animal)
    """)