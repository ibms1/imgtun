import cv2
import numpy as np
import streamlit as st
from PIL import Image as PILImage
import tempfile
import os
from pathlib import Path


# إخفاء العناصر غير المرغوب فيها
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            #stStreamlitLogo {display: none;}
            a {
                text-decoration: none;
                color: inherit;
                pointer-events: none;
            }
            a:hover {
                text-decoration: none;
                color: inherit;
                cursor: default;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def create_gif_from_transition(frames, output_path, duration=50):
    """Create GIF from transition frames using Pillow"""
    if not frames:
        st.error("No frames available to create GIF.")
        return False
    
    try:
        # Frames are already in RGB format now
        pil_frames = [PILImage.fromarray(frame) for frame in frames]
        
        # Save frames as GIF
        pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], duration=duration, loop=0)
        return True
    except Exception as e:
        st.error(f"Error creating GIF: {str(e)}")
        return False

def load_and_preprocess_image(uploaded_file, target_size=None):
    """Load and preprocess image"""
    try:
        image = PILImage.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        if target_size is not None:
            image_array = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LINEAR)
        
        return image_array
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def create_cross_dissolve(image1, image2, steps=45):
    """Create cross-dissolve transition frames"""
    frames = []
    for i in range(steps + 1):
        alpha = i / float(steps)
        # Use numpy directly for blending to preserve color accuracy
        blended = ((1 - alpha) * image1 + alpha * image2).astype(np.uint8)
        frames.append(blended)
    return frames

def main():
    st.set_page_config(page_title="Image Cross-Dissolve", layout="wide")
    
    st.markdown("""
        <style>
        .main {padding: 0rem 1rem;}
        .stButton>button {
            width: 100%;
            height: 3rem;
            margin-top: 1rem;
            background-color: #ff4b4b;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center;'>Image Cross-Dissolve</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>First Image</h3>", unsafe_allow_html=True)
        image1_file = st.file_uploader("First Image", type=["jpg", "jpeg", "png"], key="img1", label_visibility="collapsed")
        if image1_file:
            st.image(image1_file, use_column_width=True)
    
    with col2:
        st.markdown("<h3>Second Image</h3>", unsafe_allow_html=True)
        image2_file = st.file_uploader("Second Image", type=["jpg", "jpeg", "png"], key="img2", label_visibility="collapsed")
        if image2_file:
            st.image(image2_file, use_column_width=True)
    
    center_col = st.columns([1, 2, 1])[1]
    with center_col:
        convert_button = st.button("Convert")
    
    if convert_button and image1_file and image2_file:
        try:
            with st.spinner("Creating transition GIF..."):
                # Load images
                image1 = load_and_preprocess_image(image1_file)
                image2 = load_and_preprocess_image(image2_file)
                
                if image1 is None or image2 is None:
                    st.error("Failed to load one or both images. Please check the file formats.")
                    return
                
                # Resize to 720p while maintaining aspect ratio
                target_height = 720
                aspect_ratio = image1.shape[1] / image1.shape[0]
                target_width = int(target_height * aspect_ratio)
                target_width = target_width - (target_width % 2)  # Ensure even width
                
                image1 = cv2.resize(image1, (target_width, target_height))
                image2 = cv2.resize(image2, (target_width, target_height))
                
                # Create transition frames
                frames = create_cross_dissolve(image1, image2, steps=45)
                
                if not frames:
                    st.error("No frames generated for the transition.")
                    return
                
                # Create temporary file
                temp_dir = Path(tempfile.gettempdir())
                output_path = str(temp_dir / f"transition_output.gif")
                
                # Ensure file doesn't exist
                if os.path.exists(output_path):
                    os.remove(output_path)
                
                # Create GIF
                success = create_gif_from_transition(frames, output_path, duration=50)
                
                if success and os.path.exists(output_path):
                    # Read GIF file
                    with open(output_path, 'rb') as gif_file:
                        gif_bytes = gif_file.read()
                    
                    # Display GIF
                    st.markdown("<h3 style='text-align: center;'>Result</h3>", unsafe_allow_html=True)
                    st.image(gif_bytes, use_column_width=True)
                    
                    # Add download button
                    st.download_button(
                        label="Download GIF",
                        data=gif_bytes,
                        file_name="transition_output.gif",
                        mime="image/gif"
                    )
                    
                    # Clean up
                    try:
                        os.remove(output_path)
                    except:
                        pass
                else:
                    st.error("Failed to create GIF. Please try with different images.")
                    
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
    
    elif convert_button:
        st.warning("Please upload both images first.")

if __name__ == "__main__":
    main()