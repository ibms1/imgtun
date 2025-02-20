import cv2
import numpy as np
import streamlit as st
from PIL import Image as PILImage
import tempfile
import os
from pathlib import Path

def create_gif_from_transition(frames, output_path, duration=50):
    """إنشاء GIF من إطارات الانتقال باستخدام Pillow"""
    if not frames:
        st.error("No frames available to create GIF.")
        return False
    
    try:
        # تحويل الإطارات إلى تنسيق RGB
        pil_frames = [PILImage.fromarray(frame) for frame in frames]
        
        # حفظ الإطارات كـ GIF
        pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], duration=duration, loop=0)
        return True
    except Exception as e:
        st.error(f"Error creating GIF: {str(e)}")
        return False

def load_and_preprocess_image(uploaded_file):
    """تحميل وتجهيز الصورة"""
    try:
        image = PILImage.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def detect_face(image):
    """الكشف عن الوجه باستخدام OpenCV Haar cascades"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        height, width = image.shape[:2]
        return (width//4, height//4, width//2, height//2)
    
    x, y, w, h = faces[0]
    return (x, y, w, h)

def align_and_crop_faces(image1, image2, target_size=(512, 512)):
    """محاذاة واقتصاص الوجوه إلى حجم ثابت"""
    face1_bbox = detect_face(image1)
    face2_bbox = detect_face(image2)
    
    def add_padding(bbox, img_shape):
        x, y, w, h = bbox
        padding_w = int(w * 0.3)
        padding_h = int(h * 0.3)
        new_x = max(0, x - padding_w)
        new_y = max(0, y - padding_h)
        new_w = min(img_shape[1] - new_x, w + 2 * padding_w)
        new_h = min(img_shape[0] - new_y, h + 2 * padding_h)
        return (new_x, new_y, new_w, new_h)
    
    padded_bbox1 = add_padding(face1_bbox, image1.shape)
    padded_bbox2 = add_padding(face2_bbox, image2.shape)
    
    x1, y1, w1, h1 = padded_bbox1
    x2, y2, w2, h2 = padded_bbox2
    
    face1_cropped = image1[y1:y1+h1, x1:x1+w1]
    face2_cropped = image2[y2:y2+h2, x2:x2+w2]
    
    face1_resized = cv2.resize(face1_cropped, target_size, interpolation=cv2.INTER_LINEAR)
    face2_resized = cv2.resize(face2_cropped, target_size, interpolation=cv2.INTER_LINEAR)
    
    return face1_resized, face2_resized

def create_morphing_transition(image1, image2, steps=45):
    """إنشاء انتقال morphing بين صور الوجوه"""
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for morphing")
    
    frames = []
    for i in range(steps + 1):
        alpha = i / float(steps)
        blended = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)
        frames.append(blended)
    
    return frames

def main():
    st.set_page_config(page_title="Face Morphing Transition", layout="wide")
    
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
    
    st.markdown("<h1 style='text-align: center;'>Face Morphing Transition</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>First Image</h3>", unsafe_allow_html=True)
        image1_file = st.file_uploader("First Image", type=["jpg", "jpeg", "png"], key="img1", label_visibility="collapsed")
        if image1_file:
            st.image(image1_file, use_container_width=True)
    
    with col2:
        st.markdown("<h3>Second Image</h3>", unsafe_allow_html=True)
        image2_file = st.file_uploader("Second Image", type=["jpg", "jpeg", "png"], key="img2", label_visibility="collapsed")
        if image2_file:
            st.image(image2_file, use_container_width=True)
    
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            target_size = st.slider("Final Image Size", min_value=256, max_value=1024, value=512, step=64)
            steps = st.slider("Number of Transition Frames", min_value=15, max_value=90, value=45, step=5)
        with col2:
            frame_duration = st.slider("Frame Duration (ms)", min_value=20, max_value=200, value=50, step=10)
    
    center_col = st.columns([1, 2, 1])[1]
    with center_col:
        convert_button = st.button("Create Morphing Effect")
    
    if convert_button and image1_file and image2_file:
        try:
            with st.spinner("Creating morphing transition..."):
                image1 = load_and_preprocess_image(image1_file)
                image2 = load_and_preprocess_image(image2_file)
                
                if image1 is None or image2 is None:
                    st.error("Failed to load one or both images. Please check the file formats.")
                    return
                
                try:
                    st.info("Detecting and analyzing faces...")
                    image1_aligned, image2_aligned = align_and_crop_faces(
                        image1, image2, target_size=(target_size, target_size)
                    )
                    
                    aligned_col1, aligned_col2 = st.columns(2)
                    with aligned_col1:
                        st.markdown("<h4>First Face Processed</h4>", unsafe_allow_html=True)
                        st.image(image1_aligned, use_container_width=True)
                    with aligned_col2:
                        st.markdown("<h4>Second Face Processed</h4>", unsafe_allow_html=True)
                        st.image(image2_aligned, use_container_width=True)
                    
                    frames = create_morphing_transition(image1_aligned, image2_aligned, steps=steps)
                except Exception as e:
                    st.warning(f"Encountered an issue processing faces: {str(e)}. Using standard resizing.")
                    target_shape = (target_size, target_size)
                    image1_resized = cv2.resize(image1, target_shape, interpolation=cv2.INTER_LINEAR)
                    image2_resized = cv2.resize(image2, target_shape, interpolation=cv2.INTER_LINEAR)
                    frames = create_morphing_transition(image1_resized, image2_resized, steps=steps)
                
                if not frames:
                    st.error("No transition frames were generated.")
                    return
                
                temp_dir = Path(tempfile.gettempdir())
                output_path = str(temp_dir / f"transition_output.gif")
                
                if os.path.exists(output_path):
                    os.remove(output_path)
                
                success = create_gif_from_transition(frames, output_path, duration=frame_duration)
                
                if success and os.path.exists(output_path):
                    with open(output_path, 'rb') as gif_file:
                        gif_bytes = gif_file.read()
                    
                    st.markdown("<h3 style='text-align: center;'>Result</h3>", unsafe_allow_html=True)
                    st.image(gif_bytes, use_container_width=True)
                    
                    st.download_button(
                        label="Download GIF",
                        data=gif_bytes,
                        file_name="face_morphing.gif",
                        mime="image/gif"
                    )
                    
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
    
    with st.expander("Tips for Better Results"):
        st.markdown("""
        - Use face images with similar expressions for smoother transitions
        - Ensure both images have similar lighting conditions
        - For best results, use front-facing portraits
        - Increasing the number of frames gives smoother transitions but increases file size
        - Faces that look directly at the camera produce the best results
        """)

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

if __name__ == "__main__":
    main()