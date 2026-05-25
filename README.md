# 🐾 Animal Species Prediction System

An AI-powered web application that identifies animal species from images using a VGG16 deep learning model. Built with Streamlit for an intuitive, interactive user interface.
Live:https://animal-species-prediction-system.streamlit.app/

##  Table of Contents

- [Features](#features)
- [Supported Animals](#supported-animals)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Tips for Best Results](#tips-for-best-results)
- [File Descriptions](#file-descriptions)

##  Features

- **Multiple Input Methods**: Upload images or capture photos using your device camera
- **Real-time Predictions**: Instant animal species identification with confidence scores
- **Top N Predictions**: View the top predictions sorted by confidence level
- **Customizable Settings**:
  - Adjust the number of top predictions to display (1-10)
  - Set custom confidence threshold (0-100%)
- **Comprehensive Results Export**:
  - Download predictions as JSON
  - Export as CSV spreadsheet
  - Generate HTML report
- **Prediction History**: Track all predictions made during your session
- **Confidence Visualization**: Interactive progress bars and metrics
- **Responsive Design**: Works seamlessly on desktop and mobile devices

##  Supported Animals

The model can identify the following 10 animal species:

| #   | Animal       |
| --- | ------------ |
| 0   |  Dog       |
| 1   |  Horse     |
| 2   |  Elephant  |
| 3   |  Butterfly |
| 4   |  Chicken   |
| 5   |  Cat       |
| 6   |  Cow       |
| 7   |  Sheep     |
| 8   |  Spider    |
| 9   |  Squirrel  |

##  Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- 500+ MB free disk space

### Step-by-Step Setup

1. **Clone or download the project**

   ```bash
   cd Animal_Prediction_Project
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   streamlit run app.py
   ```

5. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`
   - The app will load automatically

## Project Structure

```
Animal_Prediction_Project/
├── app.py                          # Main Streamlit application
├── train.ipynb                     # Model training notebook (Google Colab)
├── labels.py                       # Label mapping for classes
├── requirements.txt                # Python dependencies
├── animal_classifier_model.h5      # Pre-trained VGG16 model
└── README.md                       # Project documentation
```

## 💻 Usage

### Basic Workflow

1. **Launch the App**

   ```bash
   streamlit run app.py
   ```

2. **Choose Input Method**
   - **Upload Image**: Click "Choose an animal image" and select a JPG, JPEG, or PNG file
   - **Camera Capture**: Use "Take a photo" to capture a live image

3. **View Results**
   - Primary prediction with confidence score
   - Top N predictions table
   - Confidence visualization bar

4. **Customize Results**
   - Use sidebar settings to adjust predictions shown and confidence threshold
   - Export results in your preferred format (JSON, CSV, or HTML)

5. **Save to History**
   - Click "✅ Save to History" to keep track of predictions
   - View all saved predictions in the "📊 Predictions History" tab

### Sidebar Controls

- **Show Top N Predictions**: Select how many top predictions to display (1-10)
- **Confidence Threshold**: Set minimum confidence level (0-100%)
- **About Section**: Information about supported animals and model details
- **Tips Section**: Best practices for optimal results

##  Model Architecture

### VGG16 Overview

This project uses **VGG16**, a deep convolutional neural network architecture that achieved excellent performance on the ImageNet dataset.

**Key Specifications:**

- **Input Size**: 224 × 224 pixels
- **Architecture**: 16 layers (13 convolutional + 3 fully connected)
- **Pre-training**: ImageNet weights
- **Output**: 10 classes (animal species)
- **Final Layer**: Softmax activation (probability distribution)

**Why VGG16?**

- Excellent accuracy for image classification tasks
- Robust feature extraction through deep layering
- Well-documented and widely adopted
- Efficient inference time for real-time predictions
- Proven performance on animal classification benchmarks

### Model Details

- **File**: `animal_classifier_model.h5`
- **Format**: Keras H5 (HDF5 format)
- **Trained on**: Custom animal species dataset (10 classes)
- **Image preprocessing**:
  - Resize to 224×224 pixels
  - Normalize pixel values (0-1 range)
  - Expand dimensions for batch processing

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

```
streamlit           # Web application framework
tensorflow-cpu     # Deep learning library (CPU version)
numpy              # Numerical computing
pillow             # Image processing
```

### Installation Details

- **Streamlit**: Creates the interactive web interface
- **TensorFlow**: Loads and runs the pre-trained VGG16 model
- **NumPy**: Handles numerical operations and array manipulations
- **Pillow**: Processes and manipulates image files

##  Tips for Best Results

### Image Quality

- ✅ Use **clear, well-lit** images
- ✅ Ensure the **animal is clearly visible**
- ✅ Include the **entire animal** in the frame
- ✅ Use **high-resolution images** (recommended: 400×400 pixels or larger)

### What to Avoid

- ❌ Blurry or low-resolution images
- ❌ Dark or poorly lit photos
- ❌ Images with multiple animals
- ❌ Extreme angles or obscured animals
- ❌ Images where the animal occupies less than 25% of the frame

### Confidence Score Interpretation

| Score Range | Confidence Level | Recommendation                   |
| ----------- | ---------------- | -------------------------------- |
| 80-100%     | Very High        | Trust the prediction             |
| 60-80%      | High             | Likely accurate                  |
| 40-60%      | Medium           | Review the image                 |
| 0-40%       | Low              | Image may be unclear or atypical |

## File Descriptions

### app.py

Main Streamlit application containing:

- UI layout and styling
- Image upload and camera input handling
- Model predictions and result processing
- Multiple export formats (JSON, CSV, HTML)
- Session state management for prediction history

### train.ipynb

Jupyter notebook for model training and development:

- Data preprocessing
- VGG16 model architecture
- Training loop and validation
- Performance metrics and visualizations

### labels.py

Simple Python file mapping numerical class indices to animal names:

- Used for displaying prediction results
- Maintains consistency across the application

### animal_classifier_model.h5

Pre-trained VGG16 model:

- Ready to use for predictions
- No training required
- Keras/TensorFlow format

##  Troubleshooting

### Issue: "Module not found" error

**Solution**: Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Issue: Model not found error

**Solution**: Ensure `animal_classifier_model.h5` is in the project directory and the path is correct in `app.py`.

### Issue: Slow predictions

**Solution**:

- Use TensorFlow GPU version for faster inference
- Reduce image resolution (still keep ≥224×224)
- Close other applications to free up system memory

### Issue: Camera not working

**Solution**:

- Check browser permissions for camera access
- Try a different browser (Chrome/Edge recommended)
- Ensure camera is not in use by another application

##  Notes

- The model is cached on first load for improved performance
- Prediction history is stored in session state and clears when the browser is refreshed
- Export features generate timestamped files for easy tracking
- The application runs locally in your browser; no data is sent to external servers

##  Support

For issues or questions:

1. Check the "ℹ️ How to Use" tab in the application
2. Review the Tips section for best practices
3. Ensure all dependencies are properly installed
4. Verify image formats are supported (JPG, JPEG, PNG)

---

**Happy Predicting! 🐾**
