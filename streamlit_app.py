import streamlit as st
from streamlit_option_menu import option_menu
import os
from datetime import datetime
import base64
import json 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout, GlobalAveragePooling1D
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
import cv2
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from keras.saving import register_keras_serializable
#-----------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Sarcasm Detection on Vietnamese Social Media Texts",
    page_icon="image.png"
)

# Custom CSS for styling
st.markdown(""" 
    <style>
    .css-1y4p8pa {
        padding-top: 0;
        padding-bottom: 0;
    }
    .banner-container {
        position: relative;
        width: 100%;
    }
    .banner-image {
        width: 100%;
        filter: brightness(0.7);  /* Make background darker */
    }
    .group-name {
        position: absolute;
        bottom: 20px;
        left: 20px;
        color: white;
        font-size: 18px;
        font-weight: bold;
        z-index: 2;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);  /* Add shadow for better readability */
    }
    /* Custom menu styling */
    .nav-link {
        background-color: #ffffff;
        color: #333333 !important;
    }
    .nav-link:hover {
        background-color: #e6e6e6 !important;
    }
    .nav-link.active {
        background-color: #AAAAAA !important;
        color: #333333 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Custom title with banner image and group name
st.markdown(""" 
    <div class="banner-container">
        <img class="banner-image" src="https://scontent.fsgn5-10.fna.fbcdn.net/v/t39.30808-6/343567058_5683586315079218_583712912555665595_n.png?_nc_cat=110&ccb=1-7&_nc_sid=cc71e4&_nc_ohc=khKFog7hPmoQ7kNvgHLKP40&_nc_zt=23&_nc_ht=scontent.fsgn5-10.fna&_nc_gid=AjVYUgFwYLBQPMso_7Cvefs&oh=00_AYARFFGGZ_XRkK93IJLRNrAkKdnBPE3qsewVZ9x3GLRwlw&oe=6771C2A6">
        <div class="group-name">Nhóm 5 - Tư duy Trí tuệ nhân tạo - AI002.P11</div>
    </div>
""", unsafe_allow_html=True)

# Menu options with custom styling
page = option_menu(
    menu_title="",
    options=["Main Posts", "Review Posts"],
    icons=["clipboard", "check-circle"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important"},
        "nav-link": {
            "font-size": "14px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#e6e6e6",
        },
        "nav-link-selected": {
            "background-color": "#AAAAAA",
        },
    }
)
#-----------------------------------------------------------------------------------------------------
class CombinedSarcasmClassifier:
    def __init__(self):
        self.model = None
        self.vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.vit_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.jina_tokenizer = AutoTokenizer.from_pretrained("uitnlp/visobert")
        self.jina_model = AutoModel.from_pretrained("uitnlp/visobert", 
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define label mapping
        self.label_mapping = {
            'not-sarcasm': 0,
            'sarcasm': 1
        }
        
        # Ensure models are in float32
        self.vit_model.to(self.device).to(torch.float32)
        self.jina_model.to(self.device).to(torch.float32)

    def encode_labels(self, labels):
        """Convert text labels to one-hot encoded format"""
        numerical_labels = [self.label_mapping[label] for label in labels]
        return tf.keras.utils.to_categorical(numerical_labels, num_classes=2)

    def decode_labels(self, one_hot_labels):
        numerical_labels = np.argmax(one_hot_labels, axis=1)
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        return [reverse_mapping[idx] for idx in numerical_labels]

    def build(self, image_dim=1000, text_dim=768):
        image_input = Input(shape=(image_dim,), name='image_input')
        text_input = Input(shape=(text_dim,), name='text_input')

        # Image processing branch
        image_dense = Dense(1024, activation='relu')(image_input)
        image_dropout = Dropout(0.3)(image_dense)
        image_dense2 = Dense(512, activation='relu')(image_dropout)

        # Text processing branch
        text_dense = Dense(512, activation='relu')(text_input)
        text_dropout = Dropout(0.3)(text_dense)
        text_dense2 = Dense(256, activation='relu')(text_dropout)

        # Combine both branches
        combined = concatenate([image_dense2, text_dense2])
        dense_combined = Dense(768, activation='relu')(combined)
        dropout_combined = Dropout(0.3)(dense_combined)
        output = Dense(2, activation='softmax', name='output')(dropout_combined)

        self.model = Model(inputs=[image_input, text_input], outputs=output)

    def preprocess_data(self, images, texts, is_test=0):
        image_features = []
        total_images = len(images)
        
        print("\nProcessing images:")
        for i, image in enumerate(images, 1):
            try:
                print(f"Processing image {i}/{total_images}: {image}", end='\r')
                temp = cv2.imread(path + image)
                inputs = self.vit_processor(images=temp, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.vit_model(**inputs)
                features = outputs.logits.cpu().numpy().squeeze()
                image_features.append(features)
            except Exception as e:
                print(f"\nError processing image {image}: {str(e)}")
                image_features.append(np.zeros(1000))  # Handle errors by adding zero vectors
        
        print("\nProcessing texts:")
        text_features = []
        total_texts = len(texts)
        for i, text in enumerate(texts, 1):
            try:
                print(f"Processing text {i}/{total_texts}", end='\r')
                inputs = self.jina_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                
                with torch.no_grad():
                    outputs = self.jina_model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                text_features.append(features)
            except Exception as e:
                print(f"\nError processing text: {str(e)}")
                text_features.append(np.zeros(768))  # Handle errors by adding zero vectors

        print("\nPreprocessing completed!")
        return np.array(image_features), np.array(text_features)


    @staticmethod
    @register_keras_serializable(package="Custom", name="f1_macro")
    def f1_macro(y_true, y_pred):
        """Custom F1 macro metric for Keras"""
        y_true_class = tf.argmax(y_true, axis=1)
        y_pred_class = tf.argmax(y_pred, axis=1)
        
        f1_scores = []
        for i in range(2):  # Update this based on the number of classes
            true_positives = tf.reduce_sum(tf.cast(
                tf.logical_and(tf.equal(y_true_class, i), tf.equal(y_pred_class, i)),
                tf.float32
            ))
            false_positives = tf.reduce_sum(tf.cast(
                tf.logical_and(tf.not_equal(y_true_class, i), tf.equal(y_pred_class, i)),
                tf.float32
            ))
            false_negatives = tf.reduce_sum(tf.cast(
                tf.logical_and(tf.equal(y_true_class, i), tf.not_equal(y_pred_class, i)),
                tf.float32
            ))
            
            precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
            recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
            f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
            f1_scores.append(f1)
        
        return tf.reduce_mean(f1_scores)


    def learning_rate_schedule(self, epoch, lr):
        """Learning rate scheduler function."""
        if 5 <= epoch :
            return lr * 0.5
        elif 20 <= epoch < 21:
            return lr * 0.01
        elif 21 <= epoch :
            return lr * 0.005
        return lr

    def train(self, x_train_images, x_train_texts, y_train):
        print("Starting preprocessing...")
        image_features, text_features = self.preprocess_data(x_train_images, x_train_texts)

        print(f"Image feature shape: {image_features.shape}")
        print(f"Text feature shape: {text_features.shape}")
        
        # Convert labels to numerical format for stratification
        numerical_labels = [self.label_mapping[label] for label in y_train]
        
        # Perform stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, val_idx = next(sss.split(image_features, numerical_labels))
        
        # Split the data
        train_image_features = image_features[train_idx]
        train_text_features = text_features[train_idx]
        val_image_features = image_features[val_idx]
        val_text_features = text_features[val_idx]
        
        # Encode labels after splitting
        y_train = np.array(y_train)  
        y_train_encoded = self.encode_labels(y_train[train_idx])
        y_val_encoded = self.encode_labels(y_train[val_idx])
        
        initial_lr = 1e-4

        print("\nCompiling model...")
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=initial_lr),
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC(), CombinedSarcasmClassifier.f1_macro]
        )
        
        class BatchProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                print(f"\nEpoch {epoch + 1} starting...")
            
            def on_batch_begin(self, batch, logs=None):
                print(f"Training batch {batch + 1}", end='\r')

        lr_scheduler = LearningRateScheduler(self.learning_rate_schedule)

        print("\nStarting training...")
        history = self.model.fit(
            [train_image_features, train_text_features],
            y_train_encoded,
            epochs=25,
            batch_size=16,
            validation_data=([val_image_features, val_text_features], y_val_encoded),
            callbacks=[BatchProgressCallback(), lr_scheduler]
        )
        
        print("\nTraining completed!")
        return history

    def predict(self, x_test_images, x_test_texts):
        print("Preprocessing test data...")
        image_features, text_features = self.preprocess_data(x_test_images, x_test_texts, 1)
        print("Making predictions...")
        predictions = self.model.predict([image_features, text_features])
        return self.decode_labels(predictions)

    def load(self, model_file):
        self.model = load_model(model_file, custom_objects={'f1_macro': self.f1_macro})

    def save(self, model_file):
        self.model.save(model_file)

    def summary(self):
        self.model.summary()
#-----------------------------------------------------------------------------------------------------
classifier = CombinedSarcasmClassifier()
classifier.build()
classifier.load('model.keras')

# Initialize session state variables if not already present
if 'pending_posts' not in st.session_state:
    st.session_state.pending_posts = []
if 'approved_posts' not in st.session_state:
    st.session_state.approved_posts = []

# Add a new post
def add_post(post):
    st.session_state.pending_posts.append(post)

# Approve a post
def approve_post(index):
    # Move the post to approved
    st.session_state.approved_posts.append(st.session_state.pending_posts.pop(index))

# Decline a post
def decline_post(index):
    st.session_state.pending_posts.pop(index)

def format_timestamp(timestamp):
    # Định dạng timestamp từ datetime string sang "Giờ:Phút, Ngày/Tháng/Năm"
    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')  # Parse string to datetime
    return dt.strftime('%H:%M, %d/%m/%Y')  # Format as Hour:Minute, Day/Month/Year

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def show_post(post, index=None, prediction=None):
    # Handle image source
    if post['image'].startswith('http'):  # Online URL
        img_src = post['image']
    else:  # Local file path
        encoded_image = encode_image(post['image'])
        img_src = f"data:image/png;base64,{encoded_image}"

    # Xác định màu và nhãn cho dự đoán
    if prediction == 0:
        prediction_label = '<span style="color: green; font-weight: bold;">Not Sarcasm</span>'
    elif prediction == 1:
        prediction_label = '<span style="color: red; font-weight: bold;">Sarcasm</span>'
    else:
        prediction_label = ''  # Không hiển thị nếu prediction là None

    # Container for the post layout
    with st.container():
        # Styled HTML post
        st.markdown(
            f"""
            <div style="
                background-color: #ffffff; 
                border: 1px solid #d3d3d3; 
                border-radius: 15px; 
                padding: 20px; 
                margin-bottom: 20px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">

            <!-- Timestamp và Prediction trên cùng một hàng -->
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-size: 15px; color: gray;">Posted at {format_timestamp(post['timestamp'])}</span>
                {prediction_label}
            </div>
            
            <!-- Caption -->
            <div style="margin-bottom: 15px;">
                <p style="font-size: 16px; font-weight: bold; margin: 0;">{post['text']}</p>
            </div>
            
            <!-- Image -->
            <div style="text-align: center;">
                <img src="{img_src}" style="max-width: 100%; border-radius: 10px;">
            </div>
            
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Buttons in container
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✔", key=f"approve_{index}", help="Approve post"):
                approve_post(index)
        with col2:
            if st.button("✖", key=f"decline_{index}", help="Decline post"):
                decline_post(index)

def display_post(post):
    # Handle image source
    if post['image'].startswith('http'):  # Online URL
        img_src = post['image']
    else:  # Local file path
        encoded_image = encode_image(post['image'])
        img_src = f"data:image/png;base64,{encoded_image}"

    # Container for the post layout
    with st.container():
        # Styled HTML post
        st.markdown(
            f"""
            <div style="
                background-color: #ffffff; 
                border: 1px solid #d3d3d3; 
                border-radius: 15px; 
                padding: 20px; 
                margin-bottom: 20px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">

            <!-- Timestamp -->
            <div style="display: flex; justify-content: flex-end; margin-bottom: 5px;">
                <span style="font-size: 15px; color: gray;">Posted at {format_timestamp(post['timestamp'])}</span>
            </div>
            
            <!-- Caption -->
            <div style="margin-bottom: 15px;">
                <p style="font-size: 16px; font-weight: bold; margin: 0;">{post['text']}</p>
            </div>
            
            <!-- Image -->
            <div style="text-align: center;">
                <img src="{img_src}" style="max-width: 100%; border-radius: 10px;">
            </div>
            
            </div>
            """, 
            unsafe_allow_html=True
        )
        
if page == 'Main Posts':
    text = st.text_input(label = "Post text", placeholder="Write something here...", label_visibility="hidden")
    if text:
        image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if image:
            if st.button("Submit"):
                if image and text:
                    # Save the uploaded image
                    image_path = os.path.join('uploads', image.name)
                    os.makedirs('uploads', exist_ok=True)
                    with open(image_path, "wb") as f:
                        f.write(image.getbuffer())

                    # Create post
                    post = {
                        "image": image_path,
                        "text": text,
                        "timestamp": str(datetime.now())
                    }
                    add_post(post)
                    st.success("Your post has been submitted for review!")
                else:
                    st.error("Please upload an image and write text.")
    if (len(st.session_state.approved_posts) > 0):
        for post in st.session_state.approved_posts:
            display_post(post)        
elif page == 'Review Posts':
    if len(st.session_state.pending_posts) == 0:
        st.title("No pending posts.")
    else:
        # Display pending posts with approve buttons
        for i, post in enumerate(st.session_state.pending_posts):
            prediction = classifier.predict(post['image'], post['text'])
            show_post(post, index=i, prediction=prediction)
            st.markdown("---")
