import streamlit as st

def main():
    # Thiết lập tiêu đề cho trang web
    st.title("Website Demo với Menu Chọn")

    # Tạo sidebar với selectbox
    with st.sidebar:
        st.header("Menu Lựa Chọn")
        
        # Tạo selectbox không có ô tick
        selected_option = st.selectbox(
            "Chọn một mục:",
            ["Mục 1", "Mục 2"]        )

    # Hiển thị nội dung tương ứng với lựa chọn
    if selected_option == "Mục 1":
        st.header("Nội dung Mục 1")
        st.write("Đây là nội dung của mục 1")
        
    else:
        st.header("Nội dung Mục 2")
        st.write("Đây là nội dung của mục 2")

if __name__ == "__main__":
    main()