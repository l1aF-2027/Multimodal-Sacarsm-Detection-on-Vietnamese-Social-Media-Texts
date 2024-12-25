import streamlit as st
from streamlit_option_menu import option_menu

# Thêm tiêu đề có ảnh bằng Markdown và HTML
st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="30" style="margin-right:10px;">
        <h1 style="display: inline;">AI002</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Menu chính
page = option_menu(
    menu_title="",
    options=["Main Posts", "Review Posts"],
    icons=["clipboard", "check-circle"],
    default_index=0,
    orientation="horizontal",
)

