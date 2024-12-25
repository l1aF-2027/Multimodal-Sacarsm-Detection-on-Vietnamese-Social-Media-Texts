import streamlit as st
from streamlit_option_menu import option_menu

page = option_menu(
    menu_title= "AI002",
    options = ["Main Posts", "Review Posts"],
    icons = ["clipboard", "check-circle"],
    default_option = "Main Posts",
    orientation="horizontal",
)