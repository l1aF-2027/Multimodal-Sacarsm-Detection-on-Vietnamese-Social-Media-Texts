import streamlit as st
import os
import json
from datetime import datetime

# File paths for storing data
PENDING_FILE = 'pending_posts.json'
APPROVED_FILE = 'approved_posts.json'

# Initialize files if they don't exist
def init_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([], f)

init_file(PENDING_FILE)
init_file(APPROVED_FILE)

# Load data from file
def load_posts(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Save data to file
def save_posts(file_path, posts):
    with open(file_path, 'w') as f:
        json.dump(posts, f, indent=4)

# Add a new post
def add_post(post, file_path):
    posts = load_posts(file_path)
    posts.append(post)
    save_posts(file_path, posts)

# Approve a post
def approve_post(index):
    pending_posts = load_posts(PENDING_FILE)
    approved_posts = load_posts(APPROVED_FILE)

    # Move the post to approved
    approved_posts.append(pending_posts.pop(index))
    
    # Save changes
    save_posts(PENDING_FILE, pending_posts)
    save_posts(APPROVED_FILE, approved_posts)

# Display a single post
def display_post(post):
    st.image(post['image'], width=300)
    st.write(post['text'])
    st.caption(f"Posted on: {post['timestamp']}")

# Main function
def main():
    st.title("Facebook Group Simulation")

    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Group Page", "Review Posts"])

    if page == "Group Page":
        st.header("Approved Posts")
        approved_posts = load_posts(APPROVED_FILE)

        # Display approved posts
        for post in approved_posts:
            display_post(post)
            st.markdown("---")

        # Form to create a new post
        st.subheader("Create a New Post")
        image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        text = st.text_area("Post text")
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
                add_post(post, PENDING_FILE)
                st.success("Your post has been submitted for review!")
            else:
                st.error("Please upload an image and write text.")

    elif page == "Review Posts":
        st.header("Pending Posts")
        pending_posts = load_posts(PENDING_FILE)

        # Display pending posts with approve buttons
        for i, post in enumerate(pending_posts):
            display_post(post)
            if st.button(f"Approve Post {i+1}", key=i):
                approve_post(i)
                st.experimental_rerun()
            st.markdown("---")

if __name__ == "__main__":
    main()
