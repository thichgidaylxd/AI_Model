"""
AI Vehicle Classification App
Ứng dụng phân loại xe cộ sử dụng MobileNetV2
Chạy với: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Thêm path hiện tại
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# ===== CẤU HÌNH TRANG =====
st.set_page_config(
    page_title="🚗 AI Phân Loại Xe Cộ",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "Ứng dụng AI phân loại xe cộ sử dụng MobileNetV2"
    }
)

# ===== CSS STYLING =====
def load_css():
    """Load custom CSS"""
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        /* Global styles */
        .main {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Header */
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            animation: fadeInDown 1s ease-out;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            font-size: 1.3rem;
            color: #7f8c8d;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Prediction box */
        .prediction-box {
            background: linear-gradient(135deg, #f1f2f6, #ddd6fe);
            padding: 2.5rem;
            border-radius: 25px;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            animation: slideInUp 0.8s ease-out;
        }
        
        .prediction-text {
            font-size: 3rem;
            font-weight: 700;
            margin: 1rem 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .confidence-text {
            font-size: 1.8rem;
            color: #7f8c8d;
            font-weight: 400;
        }
        
        /* Animations */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* File uploader styling */
        .stFileUploader > div > div > div {
            border: 2px dashed #3498db !important;
            border-radius: 15px !important;
            padding: 20px !important;
        }
        
        /* Metrics styling */
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            text-align: center;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
    </style>
    """, unsafe_allow_html=True)

# ===== LOAD MODEL =====
@st.cache_resource
def load_vehicle_model():
    """
    Load mô hình phân loại xe cộ
    Returns:
        model: TensorFlow model hoặc None nếu lỗi
    """
    model_path = Path("vehicle_model.h5")
    
    if not model_path.exists():
        st.error(f"❌ Không tìm thấy file mô hình: {model_path}")
        st.error("📁 Vui lòng copy file 'vehicle_model.h5' vào thư mục gốc của dự án")
        st.info(f"📍 Thư mục hiện tại: {current_dir}")
        return None
    
    try:
        with st.spinner("🔄 Đang load mô hình..."):
            # Create base model
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            # Create new model
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = base_model(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Load weights
            model.load_weights(str(model_path))
            
        st.success("✅ Mô hình đã được load thành công!")
        return model
    except Exception as e:
        st.error(f"❌ Lỗi khi load mô hình: {e}")
        return None

# ===== IMAGE PREPROCESSING =====
def preprocess_image(image: Image.Image) -> tuple:
    """
    Xử lý ảnh trước khi dự đoán
    
    Args:
        image: PIL Image object
        
    Returns:
        tuple: (processed_array, resized_image) hoặc (None, None) nếu lỗi
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image_resized = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, image_resized
        
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý ảnh: {e}")
        return None, None

# ===== PREDICTION =====
def predict_vehicle(model, img_array: np.ndarray) -> np.ndarray:
    """
    Thực hiện dự đoán
    
    Args:
        model: TensorFlow model
        img_array: Processed image array
        
    Returns:
        prediction: Array dự đoán hoặc None nếu lỗi
    """
    try:
        with st.spinner("🤖 AI đang phân tích..."):
            prediction = model.predict(img_array, verbose=0)
        return prediction
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {e}")
        return None

# ===== DISPLAY RESULTS =====
def display_prediction_results(prediction: np.ndarray) -> tuple:
    """
    Hiển thị kết quả dự đoán
    
    Args:
        prediction: Array kết quả dự đoán
        
    Returns:
        tuple: (predicted_idx, confidence, class_probs)
    """
    # Class information
    class_info = {
        'names': ['Ô tô', 'Xe máy', 'Xe tải'],
        'icons': ['🚗', '🏍️', '🚛'],
        'colors': ['#3498db', '#e74c3c', '#f39c12']
    }
    
    # Get prediction results
    predicted_idx = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    predicted_class = class_info['names'][predicted_idx]
    predicted_icon = class_info['icons'][predicted_idx]
    
    # Determine status and color
    if confidence > 85:
        color, status = '#27ae60', 'Rất chính xác 🎯'
    elif confidence > 70:
        color, status = '#f39c12', 'Khá chính xác 👍'
    elif confidence > 50:
        color, status = '#e67e22', 'Tạm chấp nhận ⚠️'
    else:
        color, status = '#e74c3c', 'Không chắc chắn ❓'
    
    # Display main result
    st.markdown(f"""
    <div class="prediction-box">
        <h2>🎯 Kết Quả Dự Đoán</h2>
        <div class="prediction-text" style="color: {color};">
            {predicted_icon} {predicted_class}
        </div>
        <div class="confidence-text">
            <strong>{confidence:.1f}%</strong> - {status}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return predicted_idx, confidence, prediction[0], class_info

# ===== VISUALIZATION =====
def create_confidence_chart(prediction_probs: np.ndarray, class_info: dict) -> plt.Figure:
    """
    Tạo biểu đồ confidence
    
    Args:
        prediction_probs: Array xác suất dự đoán
        class_info: Dictionary thông tin class
        
    Returns:
        matplotlib.Figure: Biểu đồ
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    confidences = prediction_probs * 100
    
    # Create bars
    bars = ax.bar(
        class_info['names'], 
        confidences, 
        color=class_info['colors'],
        alpha=0.8,
        edgecolor='black',
        linewidth=2
    )
    
    # Customize chart
    ax.set_title('📊 Phân Tích Độ Tin Cậy Chi Tiết', 
                fontsize=18, fontweight='bold', pad=25)
    ax.set_ylabel('Xác suất (%)', fontsize=14, fontweight='600')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.1f}%\n{class_info["icons"][i]}',
               ha='center', va='bottom', 
               fontweight='bold', fontsize=12)
    
    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()
    return fig

# ===== MAIN APPLICATION =====
def main():
    """Hàm chính của ứng dụng"""
    
    # Load CSS
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">🚗 AI Phân Loại Xe Cộ 🏍️</h1>', unsafe_allow_html=True)
    st.markdown('''
    <div class="subtitle">
        🤖 Sử dụng MobileNetV2 để nhận diện xe từ ảnh<br>
        📊 Hỗ trợ phân loại: <strong>Ô tô</strong> • <strong>Xe máy</strong> • <strong>Xe tải</strong>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load model
    model = load_vehicle_model()
    if model is None:
        st.stop()
    
    # Create layout
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📤 Upload Ảnh Xe")
        
        uploaded_file = st.file_uploader(
            "Chọn ảnh xe cộ để phân loại...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Hỗ trợ: JPG, PNG, BMP, WebP. Kích thước tối đa: 200MB",
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(
    image, 
    caption=f"📸 Ảnh gốc - {uploaded_file.name}", 
    use_container_width=True
)
            
            # Image info
            st.info(f"📏 Kích thước: {image.size} | 📁 Dung lượng: {len(uploaded_file.getvalue())/1024:.1f} KB")
    
    with col2:
        st.subheader("🔮 Kết Quả AI")
        
        if uploaded_file is not None:
            # Process image
            img_array, image_resized = preprocess_image(image)
            
            if img_array is not None:
                # Show processed image
                st.image(
    image_resized, 
    caption="🖼️ Ảnh đã chuẩn hóa (224×224)", 
    use_container_width=True
)
                
                # Make prediction
                prediction = predict_vehicle(model, img_array)
                
                if prediction is not None:
                    # Display results
                    predicted_idx, confidence, class_probs, class_info = display_prediction_results(prediction)
                    
                    # Store results in session state for chart
                    st.session_state.prediction_results = {
                        'class_probs': class_probs,
                        'class_info': class_info,
                        'predicted_idx': predicted_idx,
                        'confidence': confidence
                    }
        else:
            st.info("👆 Upload ảnh để bắt đầu phân tích!")
    
    # Show detailed analysis if prediction exists
    if hasattr(st.session_state, 'prediction_results'):
        results = st.session_state.prediction_results
        
        st.markdown("---")
        st.subheader("📊 Phân Tích Chi Tiết")
        
        # Chart
        fig = create_confidence_chart(results['class_probs'], results['class_info'])
        st.pyplot(fig)
        
        # Metrics
        st.subheader("📈 Thống Kê")
        col1, col2, col3 = st.columns(3)
        
        metrics_data = [
            ("🚗 Ô tô", results['class_probs'][0] * 100, 0),
            ("🏍️ Xe máy", results['class_probs'][1] * 100, 1),
            ("🚛 Xe tải", results['class_probs'][2] * 100, 2)
        ]
        
        for i, (col, (label, value, idx)) in enumerate(zip([col1, col2, col3], metrics_data)):
            with col:
                delta = "Dự đoán chính" if idx == results['predicted_idx'] else None
                st.metric(
                    label=label,
                    value=f"{value:.1f}%",
                    delta=delta
                )
        
        # Confidence interpretation
        confidence = results['confidence']
        if confidence > 85:
            st.success("🎉 **Xuất sắc!** AI rất tự tin với dự đoán này.")
        elif confidence > 70:
            st.success("👍 **Tốt!** Dự đoán đáng tin cậy.")
        elif confidence > 50:
            st.warning("⚠️ **Khá ổn** nhưng nên kiểm tra thêm.")
        else:
            st.error("❓ **Không chắc chắn.** Thử ảnh khác với chất lượng tốt hơn.")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ Thông Tin")
        st.markdown("""
        **🤖 Mô hình:** MobileNetV2 + Transfer Learning
        
        **📊 Độ chính xác:** ~95% trên test set
        
        **⚡ Tối ưu:** Nhanh & nhẹ
        
        **📝 Hướng dẫn:**
        1. Upload ảnh xe rõ nét
        2. Chờ AI phân tích
        3. Xem kết quả chi tiết
        
        **💡 Tips:**
        - Ảnh rõ nét, góc chụp tốt
        - Xe chiếm phần lớn khung hình
        - Ánh sáng đủ, không bị mờ
        """)
        
        if st.button("🔄 Làm mới ứng dụng"):
            st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9em; margin-top: 2rem;">
        🔬 Phát triển bởi <strong>AI Team</strong> | 
        ⚡ Powered by <strong>Streamlit</strong> & <strong>TensorFlow</strong><br>
        📧 Góp ý & báo lỗi: <a href="mailto:contact@example.com">contact@example.com</a>
    </div>
    """, unsafe_allow_html=True)

# ===== RUN APP =====
if __name__ == "__main__":
    main()