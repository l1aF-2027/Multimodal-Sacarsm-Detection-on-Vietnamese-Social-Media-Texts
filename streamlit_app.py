import streamlit as st

def main():
    # Thiết lập tiêu đề cho trang web
    st.title("Website Demo với Navigation Links")

    # Tạo sidebar với navigation links
    with st.sidebar:
        st.header("Menu Lựa Chọn", divider="rainbow")
        
        # Tạo container cho navigation links
        with st.container():
            # Style cho các links
            st.markdown("""
                <style>
                .nav-link {
                    padding: 8px 16px;
                    text-decoration: none;
                    display: block;
                    color: #444;
                    margin: 4px 0;
                }
                .nav-link:hover {
                    background-color: #f0f2f6;
                    border-radius: 4px;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Navigation links
            selected = st.query_params.get("page", "page1")
            
            if st.markdown('<a href="?page=page1" class="nav-link">Mục 1</a>', unsafe_allow_html=True):
                selected = "page1"
            
            if st.markdown('<a href="?page=page2" class="nav-link">Mục 2</a>', unsafe_allow_html=True):
                selected = "page2"

    # Hiển thị nội dung tương ứng với lựa chọn
    if selected == "page1":
        st.header("Nội dung Mục 1")
        st.write("Đây là nội dung của mục 1")
    else:
        st.header("Nội dung Mục 2")
        st.write("Đây là nội dung của mục 2")

if __name__ == "__main__":
    main()