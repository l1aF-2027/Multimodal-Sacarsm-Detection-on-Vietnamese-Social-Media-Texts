import streamlit as st
from streamlit_option_menu import option_menu
import os
from datetime import datetime

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

def format_timestamp(timestamp):
    # Định dạng timestamp từ datetime string sang "Giờ:Phút, Ngày/Tháng/Năm"
    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')  # Parse string to datetime
    return dt.strftime('%H:%M, %d/%m/%Y')  # Format as Hour:Minute, Day/Month/Year

def show_post(post, index):
    # Container for the post layout
    with st.container():
        # Add a styled div container for the post
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
                    <img src="data:image/jpeg;base64,{post['image']}" style="max-width: 100%; border-radius: 10px;"/>
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
            
elif page == 'Review Posts':
    if len(st.session_state.pending_posts) == 0:
        st.title("No pending posts.")
    else:
        # Display pending posts with approve buttons
        for i, post in enumerate(st.session_state.pending_posts):
            show_post(post)
            if st.button(f"Approve", key=i):
                approve_post(i)
                st.experimental_rerun()
            st.markdown("---")
