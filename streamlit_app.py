import streamlit as st
from streamlit_option_menu import option_menu

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
        <div class="group-name">Ban Học Tập Công Nghệ Phần Mềm</div>
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