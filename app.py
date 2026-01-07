import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

CLASS_NAMES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Clean names for display
DISPLAY_NAMES = [
    'Bacterial Spot',
    'Early Blight', 
    'Late Blight',
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Spider Mites',
    'Target Spot',
    'Yellow Leaf Curl Virus',
    'Mosaic Virus',
    'Healthy'
]

@st.cache_resource
def load_model():
    return keras.models.load_model('tomato_model.h5')

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

st.set_page_config(page_title="Tomato Disease Detector", layout="wide")

st.title("🍅 Tomato Leaf Disease Classifier")
st.markdown("Upload images of tomato leaves to detect diseases")


model = load_model()

with st.sidebar:
    st.header("🌿 Diseases Detected")
    for disease in DISPLAY_NAMES:
        st.write(f"- {disease}")
    
    st.header("📋 Model Info")
    st.write("**Classes:** 10 diseases")
    st.write("**Input size:** 224x224 pixels")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a tomato leaf photo",
        type=['jpg', 'jpeg', 'png'],
        help="Upload clear image of tomato leaf"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

with col2:
    if uploaded_file:
        st.subheader("🔍 Analysis Results")
        
        if st.button("Analyze Disease", type="primary"):
            with st.spinner("Processing..."):
                # Preprocess and predict
                processed = preprocess_image(image)
                predictions = model.predict(processed, verbose=0)[0]
                
                # Get top prediction
                top_idx = np.argmax(predictions)
                confidence = predictions[top_idx]
                
                # Display results
                st.success(f"**Diagnosis:** {DISPLAY_NAMES[top_idx]}")
                st.metric("Confidence", f"{confidence*100:.1f}%")
                
                # Progress bar
                st.progress(float(confidence))
                
                # Show all probabilities
                st.subheader("📊 Detailed Probabilities")
                
                # Create dataframe for sorting
                results = []
                for i, (raw_name, display_name) in enumerate(zip(CLASS_NAMES, DISPLAY_NAMES)):
                    results.append({
                        'Disease': display_name,
                        'Probability': f"{predictions[i]*100:.1f}%",
                        'Raw_Prob': predictions[i]
                    })
                
                # Sort by probability
                results.sort(key=lambda x: x['Raw_Prob'], reverse=True)
                
                # Display as bar chart
                import pandas as pd
                df = pd.DataFrame(results)
                st.bar_chart(pd.DataFrame({
                    'Probability': [r['Raw_Prob'] for r in results],
                    'Disease': [r['Disease'] for r in results]
                }).set_index('Disease'))
                
                # Display as table
                for result in results:
                    cols = st.columns([3, 1])
                    with cols[0]:
                        st.write(result['Disease'])
                    with cols[1]:
                        st.write(result['Probability'])

st.markdown("---")
st.caption("Model trained on 10,000 tomato leaf images | 10 disease classes")