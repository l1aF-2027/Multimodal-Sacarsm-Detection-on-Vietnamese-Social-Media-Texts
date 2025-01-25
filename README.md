# Multimodal Sacarsm Detection on Vietnamese Social Media Texts

[![Web demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multimodal-sacarsm-detection-on-vietnamese-social-media-texts.streamlit.app/)

<img src="https://github.com/user-attachments/assets/b6d2c448-a9b7-4409-a190-16cfde7a6476" width="300" height="auto" />

### Project Structure
1. **data_AI002:** The folder contains images.
2. **data_AI002.json:** JSON file contains images path, caption and label.
3. **code:** Contains two notebook files:
   - ocr.ipynb - Code for OCR (Optical Character Recognition) to detect text in images
   - ai002.ipynb - Main code for model training and prediction
4. **approved_post.json, pending_posts.json, predictions.json:** Contains posts and their corresponding predicted values for approved posts, pending posts, and labeled posts.
5. **model.keras**: Stores the model's architecture, weights, and parameters, allowing for model reusability, prediction, and deployment without retraining. 
6. **streamlit_app.py**: Python file containing code to deploy website using Streamlit framework.
7. **requirements.txt**: Store necessary dependencies and requirements.

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
