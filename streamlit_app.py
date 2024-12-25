import streamlit as st
from streamlit_option_menu import option_menu

# Custom CSS for styling
st.markdown("""
    <style>
    .css-1y4p8pa {
        padding-top: 0;
        padding-bottom: 0;
    }
    .menu-title {
        background-image: url('https://scontent.fsgn5-10.fna.fbcdn.net/v/t39.30808-6/343567058_5683586315079218_583712912555665595_n.png?_nc_cat=110&ccb=1-7&_nc_sid=cc71e4&_nc_ohc=khKFog7hPmoQ7kNvgHLKP40&_nc_zt=23&_nc_ht=scontent.fsgn5-10.fna&_nc_gid=AjVYUgFwYLBQPMso_7Cvefs&oh=00_AYARFFGGZ_XRkK93IJLRNrAkKdnBPE3qsewVZ9x3GLRwlw&oe=6771C2A6');
        background-size: cover;
        background-position: center;
        padding: 20px;
        position: relative;
        color: white;
        text-align: center;
    }
    .menu-title::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        z-index: 1;
    }
    .menu-title h1 {
        position: relative;
        z-index: 2;
        margin: 0;
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
        background-color: #CCCCCC !important;  /* Light blue-gray color */
        color: #333333 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Custom title with banner image
st.markdown("""
    <div style="text-align: center;">
        <img src="https://scontent.fsgn5-10.fna.fbcdn.net/v/t39.30808-6/343567058_5683586315079218_583712912555665595_n.png?_nc_cat=110&ccb=1-7&_nc_sid=cc71e4&_nc_ohc=khKFog7hPmoQ7kNvgHLKP40&_nc_zt=23&_nc_ht=scontent.fsgn5-10.fna&_nc_gid=AjVYUgFwYLBQPMso_7Cvefs&oh=00_AYARFFGGZ_XRkK93IJLRNrAkKdnBPE3qsewVZ9x3GLRwlw&oe=6771C2A6" style="width: 100%; margin-bottom: 0;">
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
            "background-color": "#CCCCCC",  # Light blue-gray when selected
        },
    }
)