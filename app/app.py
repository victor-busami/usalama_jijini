import streamlit as st
import pandas as pd
from PIL import Image
import os
from transformers import pipeline
import torch
from streamlit_folium import st_folium
import folium
from ultralytics import YOLO
import base64
import requests
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import sqlite3
import time

st.set_page_config(page_title="Nairobi Crime Reporting AI App", layout="wide")

# --- Sidebar navigation with Home (Landing Page) ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# Only use sidebar radio if not redirected by button
if st.session_state['page'] == 'Home':
    page = st.sidebar.radio("Go to", ["Home", "Report Crime", "View Reports"])
else:
    page = st.session_state['page']

if page == "Home":
    # --- Custom CSS for styling and animations (from landingpage.py) ---
    st.markdown("""
        <style>
        @keyframes fadeInOut {
            0% { opacity: 0; }
            50% { opacity: 1; }
            100% { opacity: 0; }
        }
        .animated-text {
            font-size: 1.5rem;
            color: #FF4B4B;
            font-weight: bold;
            animation: fadeInOut 3s ease-in-out infinite;
            text-align: center;
            margin-bottom: 20px;
        }
        .cta-button {
            display: flex;
            justify-content: center;
            margin-top: 32px;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Header Section ---
    st.markdown("""
        <h1 style='color: #4B8BBE;'>🛡️ Usalama Jijini - Crime Reporting & Safety System</h1>
        <h4 style='color: gray;'>Helping Nairobi Residents Report, Track, and Prevent Crime in Real-Time</h4>
        <div class="animated-text">Stay Alert. Stay Safe. Report Instantly.</div>
        <hr>
    """, unsafe_allow_html=True)

    # --- Side-by-side layout for Hero Image and User Guidance ---
    col1, col2 = st.columns(2)
    with col1:
        image_path = "assets/hero_safety_image.png"
        try:
            image = Image.open(image_path)
            st.image(image, use_container_width=True, caption="Together for a Safer Nairobi")
        except Exception:
            st.warning("Hero image not found. Please add your image to the 'assets' folder with the correct name.")
        # --- Button Row with Hover Effects ---
        st.markdown("""
        <style>
        .button-row {
            display: flex;
            justify-content: center;
            gap: 18px;
            margin-top: 18px;
            margin-bottom: 8px;
        }
        .btn-main {
            background: linear-gradient(90deg, #ff4b4b 0%, #ffb347 100%);
            color: white;
            font-weight: bold;
            font-size: 1.2em;
            border: none;
            border-radius: 8px;
            padding: 14px 32px;
            box-shadow: 0 2px 8px rgba(255,75,75,0.15);
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s;
        }
        .btn-main:hover {
            background: linear-gradient(90deg, #ffb347 0%, #ff4b4b 100%);
            color: #fff;
            transform: scale(1.07);
            box-shadow: 0 4px 16px rgba(255,75,75,0.25);
        }
        .btn-secondary {
            background: #f0f2f6;
            color: #4B8BBE;
            font-weight: 500;
            font-size: 1em;
            border: 1.5px solid #4B8BBE;
            border-radius: 8px;
            padding: 12px 24px;
            cursor: pointer;
            transition: background 0.15s, color 0.15s, transform 0.15s;
        }
        .btn-secondary:hover {
            background: #4B8BBE;
            color: #fff;
            transform: scale(1.05);
        }
        </style>
        <div class="button-row">
            <button class="btn-main" id="report_crime_btn">🚨 Report Crime</button>
            <button class="btn-secondary" id="view_reports_btn">📊 View Reports</button>
            <button class="btn-secondary" id="about_btn">ℹ️ About</button>
        </div>
        <script>
        const reportBtn = window.parent.document.getElementById('report_crime_btn');
        const viewBtn = window.parent.document.getElementById('view_reports_btn');
        const aboutBtn = window.parent.document.getElementById('about_btn');
        if (reportBtn) reportBtn.onclick = () => window.parent.postMessage({type: 'streamlit:setComponentValue', key: 'page', value: 'Report Crime'}, '*');
        if (viewBtn) viewBtn.onclick = () => window.parent.postMessage({type: 'streamlit:setComponentValue', key: 'page', value: 'View Reports'}, '*');
        if (aboutBtn) aboutBtn.onclick = () => window.parent.postMessage({type: 'streamlit:setComponentValue', key: 'page', value: 'Home'}, '*');
        </script>
        """, unsafe_allow_html=True)
        # Fallback for Streamlit: use st.button for navigation
        col_btns = st.columns([1,1,1])
        with col_btns[0]:
            if st.button("🚨 Report Crime", key="report_crime_btn_fallback", help="Go to the crime reporting form"):
                st.session_state['page'] = 'Report Crime'
                st.rerun()
        with col_btns[1]:
            if st.button("📊 View Reports", key="view_reports_btn_fallback", help="Go to the reports page"):
                st.session_state['page'] = 'View Reports'
                st.rerun()
        with col_btns[2]:
            if st.button("ℹ️ About", key="about_btn_fallback", help="About this app"):
                st.session_state['page'] = 'Home'
                st.rerun()
    with col2:
        st.markdown("""
**👋 Welcome to Usalama Jijini!**

Your community-driven tool for staying informed, staying safe, and taking action.

---

### 📸 How to Report a Crime

1. **Take a clear photo** of the crime-related activity.
2. **Navigate to the sidebar** and click **Report Crime**.
3. **Upload the photo**, write a short description (optional but helpful).
4. **Enter your location details** accurately (County, Sub-County, Ward, etc).
5. **Hit the Report button** to alert local safety authorities.

---

⚠️ **Every report helps build a safer community.**

Whether it's suspicious activity, petty theft, or public disturbance — your voice matters.

---

🚀 **Let’s make Nairobi safer, one report at a time!**
        """)
    st.stop()

st.title("Nairobi County Crime Reporting System")

# Initialize session state for reports
def init_reports():
    if 'reports' not in st.session_state:
        st.session_state['reports'] = []

init_reports()

# Load sentiment analysis pipeline once
@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_pipeline = get_sentiment_pipeline()

# Load YOLOv8 model once
@st.cache_resource
def get_yolo_model():
    return YOLO('yolov8n.pt')

yolo_model = get_yolo_model()

# Add local image captioning model loader and function
@st.cache_resource
def get_captioning_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer

def get_local_image_caption(image_path):
    model, feature_extractor, tokenizer = get_captioning_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=16)  # greedy decoding only
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def map_caption_to_sentiment(caption):
    negative_keywords = [
        'violence', 'injury', 'attack', 'gun', 'knife', 'blood', 'fire', 'explosion', 'fight', 'crying', 'sad', 'danger', 'accident', 'protest', 'riot', 'dead', 'death', 'hurt', 'wound'
    ]
    positive_keywords = [
        'smile', 'happy', 'safe', 'peace', 'joy', 'celebration', 'calm', 'help', 'rescue', 'hug', 'love'
    ]
    caption_lower = caption.lower()
    if any(word in caption_lower for word in negative_keywords):
        return "Very Negative"
    elif any(word in caption_lower for word in positive_keywords):
        return "Very Positive"
    else:
        return "Neutral"

def map_image_urgency(image_sentiment):
    sentiment_to_urgency = {
        'Very Negative': ('High', '🚨'),
        'Very Positive': ('Very Low', '🎉'),
        'Neutral': ('Medium/Low', '🟡'),
    }
    return sentiment_to_urgency.get(image_sentiment, ('Medium/Low', '🟡'))

def combine_sentiment_and_urgency(text_sentiment, text_urgency, text_urgency_emoji, image_sentiment, image_urgency, image_urgency_emoji):
    # Priority: High > Medium/Low > Very Low
    urgency_order = {'High': 3, 'Medium/Low': 2, 'Very Low': 1}
    # Pick the higher urgency
    if urgency_order.get(text_urgency, 2) >= urgency_order.get(image_urgency, 2):
        combined_urgency = text_urgency
        combined_urgency_emoji = text_urgency_emoji
    else:
        combined_urgency = image_urgency
        combined_urgency_emoji = image_urgency_emoji
    # Sentiment: Very Negative > Negative > Neutral > Positive > Very Positive
    sentiment_order = {
        'Very Negative': 1, 'Negative': 2, 'Neutral': 3, 'Positive': 4, 'Very Positive': 5
    }
    # Use the more negative sentiment
    if sentiment_order.get(text_sentiment.split()[0], 3) <= sentiment_order.get(image_sentiment.split()[0], 3):
        combined_sentiment = text_sentiment
    else:
        combined_sentiment = image_sentiment
    return combined_sentiment, combined_urgency, combined_urgency_emoji

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT,
        image TEXT,
        location TEXT,
        contact TEXT,
        sentiment TEXT,
        urgency TEXT,
        objects TEXT,
        image_sentiment TEXT,
        image_caption TEXT,
        image_urgency TEXT
    )''')
    conn.commit()
    conn.close()
init_db()

def save_report_to_db(report):
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('''INSERT INTO reports (description, image, location, contact, sentiment, urgency, objects, image_sentiment, image_caption, image_urgency)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (report['description'], report['image'], report['location'], report['contact'], report['sentiment'], report['urgency'], report['objects'], report.get('image_sentiment'), report.get('image_caption'), report.get('image_urgency')))
    conn.commit()
    conn.close()

def load_reports_from_db():
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('SELECT description, image, location, contact, sentiment, urgency, objects, image_sentiment, image_caption, image_urgency FROM reports')
    rows = c.fetchall()
    conn.close()
    keys = ['description', 'image', 'location', 'contact', 'sentiment', 'urgency', 'objects', 'image_sentiment', 'image_caption', 'image_urgency']
    return [dict(zip(keys, row)) for row in rows]

# --- Auth ---
def login():
    if st.session_state.get('logged_in', False):
        if st.button('Logout', key='logout_button'):
            st.session_state['logged_in'] = False
            st.rerun()
        st.success('Login successful!')
        return True
    st.subheader('Login')
    username = st.text_input('Username', key='login_username')
    password = st.text_input('Password', type='password', key='login_password')
    # Use a less common password to avoid browser breach warnings
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD = 'Admin!2024Secure'  # Change to a strong, unique password
    if st.button('Login', key='login_button'):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state['logged_in'] = True
            st.success('Login successful!')
            st.rerun()
        else:
            st.error('Invalid credentials')
    # Display admin credentials below the login form
    st.info(f"Admin Login - Username: {ADMIN_USERNAME} | Password: {ADMIN_PASSWORD}")
    return st.session_state.get('logged_in', False)

# Sidebar navigation
# page = st.sidebar.radio("Go to", ["Report Crime", "View Reports"]) # This line is now redundant as Home handles the sidebar

# Sidebar now only contains the navigation radio button

if page == "Report Crime":
    # Add Back to Home button
    if st.button("🏠 Back to Home", key="back_to_home_from_report"):
        st.session_state['page'] = 'Home'
        st.rerun()
    # Prevent admin from submitting a report unless logged out
    if st.session_state.get('logged_in', False):
        st.warning("Admin cannot submit a report while logged in. Please log out to access the report form.")
        if st.button('Logout', key='admin_logout_button_report'):
            st.session_state['logged_in'] = False
            st.rerun()
    else:
        # Show toast if flag is set
        if st.session_state.get('show_report_toast', False):
            st.toast("Report submitted!", icon="✅")
            st.session_state['show_report_toast'] = False
        # Clear form fields if reset flag is set
        if st.session_state.get('reset_crime_form', False):
            st.session_state['crime_form_description'] = ''
            st.session_state['crime_form_location'] = None
            st.session_state['crime_form_contact'] = ''
            st.session_state['reset_crime_form'] = False
        st.header("Report a Crime")
        with st.form("crime_form"):
            description = st.text_area("Describe the incident", help="What happened? Where? When? Any details?", key="crime_form_description")
            image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"], key="crime_form_image")
            location = st.selectbox("Select location (sub-county)", [
                "Westlands", "Kasarani", "Lang'ata", "Embakasi", "Starehe", "Dagoretti", "Kamukunji", "Makadara", "Mathare", "Kibra", "Roysambu", "Ruaraka", "Other"
            ], key="crime_form_location")
            contact = st.text_input("Contact info (optional)", key="crime_form_contact")
            submitted = st.form_submit_button("Submit Report")

            if submitted:
                # Run sentiment analysis
                if description.strip():
                    sentiment_result = sentiment_pipeline(description)[0]
                    label = sentiment_result['label']
                    # Map star rating to sentiment
                    star_to_sentiment = {
                        '1 star': 'Very Negative',
                        '2 stars': 'Negative',
                        '3 stars': 'Neutral',
                        '4 stars': 'Positive',
                        '5 stars': 'Very Positive',
                    }
                    sentiment_label = star_to_sentiment.get(label, label)
                    # --- Custom logic for negative keywords ---
                    negative_keywords = [
                        'attack', 'kill', 'robbery', 'assault', 'theft', 'violence', 'murder', 'rape', 'shooting', 'stab',
                        'injury', 'danger', 'threat', 'gun', 'knife', 'explosion', 'bomb', 'terror', 'crime', 'goon', 'gang',
                        'thief', 'hijack', 'kidnap', 'abduct', 'arson', 'riot', 'fight', 'abuse', 'molest', 'harass', 'rape'
                    ]
                    if (sentiment_label in ['Positive', 'Very Positive'] and any(word in description.lower() for word in negative_keywords)):
                        sentiment_label = 'Very Negative'
                    sentiment = f"{sentiment_label} (score: {sentiment_result['score']:.2f})"
                    # Map sentiment to urgency and emoji
                    sentiment_to_urgency = {
                        'Very Negative': ('High', '🚨'),
                        'Negative': ('Medium-High', '⚠️'),
                        'Neutral': ('Medium/Low', '🟡'),
                        'Positive': ('Low', '✅'),
                        'Very Positive': ('Very Low', '🎉'),
                    }
                    urgency, urgency_emoji = sentiment_to_urgency.get(sentiment_label, ('Medium/Low', '🟡'))
                else:
                    sentiment = "No description provided"
                    urgency = "Unknown"
                    urgency_emoji = "❓"
                # Run object detection if image is uploaded
                detected_objects = "No image uploaded"
                img_path = None
                image_sentiment = "No image uploaded"
                if image:
                    img_path = os.path.join("uploads", image.name)
                    os.makedirs("uploads", exist_ok=True)
                    with open(img_path, "wb") as f:
                        f.write(image.read())
                    # Object detection using ultralytics YOLO
                    results = yolo_model(img_path)
                    labels = set()
                    for r in results:
                        if hasattr(r, 'names') and hasattr(r, 'boxes'):
                            for c in r.boxes.cls:
                                labels.add(r.names[int(c)])
                    detected_objects = ', '.join(labels) if labels else 'No objects detected'
                    # Local image captioning (as sentiment proxy)
                    caption = get_local_image_caption(img_path)
                    image_sentiment = map_caption_to_sentiment(caption)
                    image_urgency, image_urgency_emoji = map_image_urgency(image_sentiment)

                if image and description.strip():
                    # Combine sentiment and urgency
                    combined_sentiment, combined_urgency, combined_urgency_emoji = combine_sentiment_and_urgency(
                        sentiment_label, urgency, urgency_emoji, image_sentiment, image_urgency, image_urgency_emoji
                    )
                    report = {
                        "description": description,
                        "image": img_path,
                        "location": location,
                        "contact": contact,
                        "sentiment": combined_sentiment,
                        "urgency": f"{combined_urgency} {combined_urgency_emoji}",
                        "objects": detected_objects,
                        "image_sentiment": image_sentiment,
                        "image_caption": caption,
                        "image_urgency": f"{image_urgency} {image_urgency_emoji}",
                    }
                    st.session_state['reports'].append(report)
                    save_report_to_db(report)
                    st.session_state['reset_crime_form'] = True
                    st.session_state['show_report_toast'] = True
                    st.rerun()
                else:
                    report = {
                        "description": description,
                        "image": img_path,
                        "location": location,
                        "contact": contact,
                        "sentiment": sentiment,
                        "urgency": f"{urgency} {urgency_emoji}",
                        "objects": detected_objects,
                        "image_sentiment": image_sentiment,
                        "image_caption": caption if image else None,
                        "image_urgency": f"{image_urgency} {image_urgency_emoji}" if image else None
                    }
                    st.session_state['reports'].append(report)
                    save_report_to_db(report)
                    st.session_state['reset_crime_form'] = True
                    st.session_state['show_report_toast'] = True
                    st.rerun()

elif page == "View Reports":
    # Add Back to Home button
    if st.button("🏠 Back to Home", key="back_to_home_from_view_reports"):
        st.session_state['page'] = 'Home'
        st.rerun()
    if not st.session_state.get('logged_in', False):
        if not login():
            st.stop()
    # Add logout button to admin page
    if st.button('Logout', key='admin_logout_button_view_reports'):
        st.session_state['logged_in'] = False
        st.rerun()
    st.header("Crime Reports")
    reports = load_reports_from_db()
    if not reports:
        st.info("No reports yet.")
    else:
        df = pd.DataFrame(reports)
        # --- Summary Stats ---
        st.subheader("Report Summary")
        colA, colB, colC, colD = st.columns(4)
        with colA:
            st.metric("Total Reports", len(df))
        with colB:
            st.metric("High Urgency", (df['urgency'].str.contains('High')).sum())
        with colC:
            st.metric("Unique Locations", df['location'].nunique())
        with colD:
            st.metric("Very Negative Sentiment", (df['sentiment'].str.contains('Very Negative')).sum())