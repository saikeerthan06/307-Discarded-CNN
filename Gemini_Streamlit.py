import streamlit as st
st.set_page_config(
    page_title="NeuroNova",
    layout="centered",
    initial_sidebar_state="collapsed"
)
import json
import os
import hashlib
import numpy as np
from PIL import Image
import tensorflow as tf
import uuid
import openai
import google.generativeai as genai
import cv2

# --- API Configuration ---
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except (KeyError, Exception):
    pass

# --- Constants ---
USERS_FILE = 'users.json'
HISTORY_FILE = 'prediction_history.json'
CHAT_HISTORY_FILE = 'chat_history.json'
MODEL_PATH = '/Users/saikeerthan/NYP-AI/Year3/AI_Application/Cancer_Project3/Model_Files/EfficientB0/Efficient_Cancer.keras'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['brain_glioma', 'brain_menin', 'brain_tumor', 'breast_benign', 'breast_malignant', 'cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi', 'colon_aca', 'colon_bnt', 'kidney_normal', 'kidney_tumor', 'lung_aca', 'lung_bnt', 'lung_scc', 'lymph_cll', 'lymph_fl', 'lymph_mcl', 'oral_normal', 'oral_scc']
GPT_MODEL = "GPT-4.1-mini"
GEMINI_MODEL = "Gemini 2.5 Flash"
PROFILE_PICS_DIR = 'profile_pics' # Directory to store profile pictures

# --- Helper, Model Loading, and Preprocessing Functions ---
def load_json(file_path, default_value=None):
    # ... (code unchanged)
    if default_value is None:
        default_value = []
    if not os.path.exists(file_path):
        return default_value
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default_value

def save_json(data, file_path):
    # ... (code unchanged)
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        st.error(f"Could not save data to {file_path}: {e}")

@st.cache_resource
def load_tf_model(path):
    # ... (code unchanged)
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Please ensure the path is correct.")
        return None
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    # ... (code unchanged)
    try:
        img = image.resize(IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# --- Authentication Pages ---
# (Authentication functions remain here, unchanged)
def login_page():
    # ... (code unchanged)
    st.title("NeuroNova (Multi-Cancer)")
    st.header("Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        col1, col2 = st.columns([1, 1])
        with col1:
            login_button = st.form_submit_button("Log In")
        with col2:
            signup_button = st.form_submit_button("Sign Up")
    if login_button:
        handle_login(email, password)
    if signup_button:
        handle_signup(email, password)

# def handle_login(email, password):
#     # ... (code unchanged)
#     if not email or not password:
#         st.warning("Please enter both email and password.")
#         return
#     users = load_json(USERS_FILE, default_value={})
#     hashed_password = hashlib.sha256(password.encode()).hexdigest()
#     if email in users and users[email] == hashed_password:
#         st.session_state['logged_in'] = True
#         st.session_state['username'] = email
#         st.success("Login successful!")
#         st.rerun()
#     else:
#         st.error("Invalid email or password.")

# def handle_signup(email, password):
#     # ... (code unchanged)
#     if not email or not password:
#         st.warning("Please enter both email and password to sign up.")
#         return
#     users = load_json(USERS_FILE, default_value={})
#     if email in users:
#         st.error("An account with this email already exists.")
#         return
#     hashed_password = hashlib.sha256(password.encode()).hexdigest()
#     users[email] = hashed_password
#     save_json(users, USERS_FILE)
#     st.success("Sign up successful! You can now log in.")
def handle_login(email, password):
    if not email or not password:
        st.warning("Please enter both email and password.")
        return
    
    users = load_json(USERS_FILE, default_value={})
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    # Updated check for the new data structure
    if email in users and users[email]['password'] == hashed_password:
        st.session_state['logged_in'] = True
        st.session_state['username'] = email
        st.success("Login successful!")
        st.rerun()
    else:
        st.error("Invalid email or password.")

def handle_signup(email, password):
    if not email or not password:
        st.warning("Please enter both email and password to sign up.")
        return

    users = load_json(USERS_FILE, default_value={})
    if email in users:
        st.error("An account with this email already exists.")
        return

    # Create a full user object on signup
    users[email] = {
        'password': hashlib.sha256(password.encode()).hexdigest(),
        'profile_picture': None,
        'description': ''
    }
    save_json(users, USERS_FILE)
    st.success("Sign up successful! You can now log in.")

# --- Prediction and History Pages (with "Ask AI" modification) ---
def prediction_page():
    # ... (code unchanged)
    st.title("NeuroNova - Image Classification")
    uploaded_files = st.file_uploader(
        "Upload microscopic images for classification",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    if 'files_to_predict' not in st.session_state:
        st.session_state['files_to_predict'] = []
    if uploaded_files:
        st.session_state['files_to_predict'] = uploaded_files
    if st.session_state['files_to_predict']:
        st.subheader("Uploaded Images")
        cols = st.columns(4)
        for i, file in enumerate(st.session_state['files_to_predict']):
            with cols[i % 4]:
                st.image(file, caption=file.name, use_container_width=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Predict", use_container_width=True):
                run_prediction()
        with col2:
            if st.button("Clear Uploads", use_container_width=True):
                st.session_state['files_to_predict'] = []
                st.rerun()

def run_prediction():
    # ... (code unchanged)
    model = load_tf_model(MODEL_PATH)
    if not model or not st.session_state['files_to_predict']:
        return
    results = []
    new_history = []
    with st.spinner("Running predictions..."):
        for file in st.session_state['files_to_predict']:
            image = Image.open(file).convert('RGB')
            processed_image = preprocess_image(image)
            if processed_image is not None:
                prediction = model.predict(processed_image)
                predicted_index = np.argmax(prediction[0])
                confidence = np.max(prediction[0]) * 100
                predicted_class = CLASS_NAMES[predicted_index]
                results.append({
                    "name": file.name,
                    "class": predicted_class,
                    "confidence": confidence
                })
                if not os.path.exists('history_images'):
                    os.makedirs('history_images')
                img_path = os.path.join('history_images', file.name)
                image.save(img_path)
                new_history.append({
                    "id": str(uuid.uuid4()),
                    "image_path": img_path,
                    "display_name": file.name,
                    "prediction": predicted_class,
                    "confidence": float(confidence)
                })
    st.subheader("Prediction Results")
    for res in results:
        st.markdown(f"**{res['name']}**: {res['class']} ({res['confidence']:.2f}%)")
    history = load_json(HISTORY_FILE)
    history.extend(new_history)
    save_json(history, HISTORY_FILE)
    st.session_state['files_to_predict'] = []

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8) # Add epsilon for stability
    return heatmap.numpy()

def generate_gradcam_overlay(img_path, model, last_conv_layer_name='top_conv'):
    """Prepares images and generates the Grad-CAM overlay, ensuring channel compatibility."""
    # Preprocess the image for the model
    img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)

    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array_expanded, model, last_conv_layer_name)

    # Load original image with OpenCV, forcing it to be a 3-channel BGR color image
    # THIS IS THE LINE THAT FIXES THE ERROR
    original_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    original_img = cv2.resize(original_img, IMAGE_SIZE)

    # Create the color heatmap
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Resize the heatmap to match the original image dimensions
    jet = cv2.resize(jet, IMAGE_SIZE)
    
    # Blend the 3-channel original image and the 3-channel color heatmap
    superimposed_img = cv2.addWeighted(original_img, 0.6, jet, 0.4, 0)
    
    return original_img, superimposed_img

# def history_page():
#     """Displays prediction history with the new 'Ask AI' feature."""
#     st.title("Prediction History")
#     history = load_json(HISTORY_FILE)
#     if not history:
#         st.info("No prediction history found.")
#         return
#     # ... (Delete all button logic remains the same)

#     for index, item in enumerate(history):
#         with st.container(border=True):
#             col1, col2 = st.columns([1, 2])
#             with col1:
#                 if os.path.exists(item['image_path']):
#                     st.image(item['image_path'], use_container_width=True)
#             with col2:
#                 # ... (Display name and prediction info)
#                 st.markdown(f"**Predicted Label:** {item['prediction']}")
#                 st.markdown(f"**Confidence Score:** {item.get('confidence', 0):.2f}%")
                
#                 with st.popover("🔬 Ask AI"):
#                     st.markdown("Choose a model for analysis:")
#                     model_choice = st.selectbox("Model", [GPT_MODEL, GEMINI_MODEL], key=f"ask_ai_{item['id']}")
#                     if st.button("Analyze", key=f"analyze_{item['id']}"):
#                         # Prepare for transition to chat interface
#                         st.session_state.view_chat = True
#                         st.session_state.active_chat_id = None
#                         st.session_state.pre_selected_model = model_choice
#                         st.session_state.chat_type = "Advanced Analysis"
#                         st.session_state.pre_filled_prompt = (
#                             f"Provide a detailed analysis of the following cancer prediction. "
#                             f"Explain what '{item['prediction']}' is, what a confidence score of "
#                             f"{item.get('confidence', 0):.2f}% implies in this context, and outline "
#                             f"common next steps or characteristics associated with this diagnosis."
#                         )
#                         st.rerun()
def history_page():
    """Displays prediction history with 'Ask AI' and 'Grad-CAM' features."""
    st.title("Prediction History")
    history = load_json(HISTORY_FILE)

    if 'show_gradcam_for' not in st.session_state:
        st.session_state.show_gradcam_for = None

    # --- Grad-CAM Dialog Logic ---
    # This block now runs first to create the dialog if needed
    if st.session_state.show_gradcam_for:
        item_id = st.session_state.show_gradcam_for
        item = next((i for i in history if i['id'] == item_id), None)
        
        # Call st.dialog directly. It will be displayed on the rerun.
        st.dialog("Grad-CAM Analysis")
        
        if item:
            # Use 'st' directly, not 'dialog'
            st.info("The heatmap highlights the regions the model focused on for its prediction.")
            model = load_tf_model(MODEL_PATH)
            
            if model:
                with st.spinner("Generating Grad-CAM..."):
                    original, gradcam_img = generate_gradcam_overlay(item['image_path'], model)
                
                # Use 'st' directly to create columns inside the dialog
                col1, col2 = st.columns(2)
                col1.image(original, caption="Original Image", channels="BGR")
                col2.image(gradcam_img, caption="Grad-CAM Heatmap", channels="BGR")

                st.markdown(f"<p style='text-align: center;'><b>Predicted:</b> {item['prediction']} | <b>Confidence:</b> {item.get('confidence', 0):.2f}%</p>", unsafe_allow_html=True)
                
                if st.button("Close"):
                    st.session_state.show_gradcam_for = None
                    st.rerun()
            else:
                st.error("Could not load the model to generate Grad-CAM.")
        else:
            # If item not found, ensure dialog is closed on next run
            st.session_state.show_gradcam_for = None
            st.rerun()

    # --- Main History Display Loop ---
    st.markdown("---")
    for item in history:
        with st.container(border=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                if os.path.exists(item['image_path']):
                    st.image(item['image_path'], use_column_width=True)
            with col2:
                st.markdown(f"**Predicted Label:** {item['prediction']}")
                st.markdown(f"**Confidence Score:** {item.get('confidence', 0):.2f}%")
                
                button_col1, button_col2 = st.columns(2)
                with button_col1:
                    if st.button("👁️ View Grad-CAM", key=f"gradcam_{item['id']}", use_container_width=True):
                        st.session_state.show_gradcam_for = item['id']
                        st.rerun() # Rerun to trigger the dialog logic
                
                with button_col2:
                    with st.popover("🔬 Ask AI", use_container_width=True):
                        model_choice = st.selectbox("Model", [GPT_MODEL, GEMINI_MODEL], key=f"ask_ai_{item['id']}")
                        if st.button("Analyze", key=f"analyze_{item['id']}"):
                            st.session_state.view_chat = True
                            st.session_state.active_chat_id = None
                            st.session_state.pre_selected_model = model_choice
                            st.session_state.chat_type = "Advanced Analysis"
                            st.session_state.pre_filled_prompt = (
                                f"Provide a detailed analysis of the cancer prediction for '{item['prediction']}' "
                                f"with a confidence of {item.get('confidence', 0):.2f}%."
                            )
                            st.rerun()

# --- AI Assistant Section (Heavily Modified) ---

def generate_chat_title(prompt, model_choice):
    """Uses an LLM to generate a concise title for a new chat."""
    try:
        title_prompt = f"Summarize the following user query into a concise, 5-word-or-less title: '{prompt}'"
        if "GPT" in model_choice:
            client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(model="gpt-4.1-mini", messages=[{"role": "user", "content": title_prompt}])
            return response.choices[0].message.content.strip().strip('""')
        else: # Gemini
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(title_prompt)
            return response.text.strip().strip('""')
    except Exception:
        return prompt[:40] + "..." # Fallback

def assistant_page_router():
    """Routes to the correct assistant view based on session state."""
    if st.session_state.get('view_chat', False):
        display_chat_interface()
    else:
        display_chat_history()

def display_chat_history():
    """Displays the list of saved chats with advanced filtering."""
    st.title("🤖 Chat History")
    chats = load_json(CHAT_HISTORY_FILE)
    
    # --- Main Filter ---
    main_filter = st.selectbox("Filter by chat type:", ("All Chats", "General Chats", "Advanced Analysis"))
    
    # --- Conditional Secondary Filter ---
    model_filter = "All Models"
    if main_filter != "All Chats":
        model_filter = st.selectbox("Filter by model:", ("All Models", GPT_MODEL, GEMINI_MODEL))

    # --- Apply Filters ---
    filtered_chats = []
    for chat in chats:
        type_match = (main_filter == "All Chats" or 
                      (main_filter == "General Chats" and chat.get('type') != "Advanced Analysis") or
                      (main_filter == "Advanced Analysis" and chat.get('type') == "Advanced Analysis"))
        
        model_match = (model_filter == "All Models" or chat.get('model') == model_filter)

        if type_match and model_match:
            filtered_chats.append(chat)
    
    # ... (Rest of the display, open, and delete logic is the same)
    st.markdown("---")
    if not filtered_chats:
        st.info("No chats found for the selected filters.")

    for chat in reversed(filtered_chats):
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(chat.get('title', 'Untitled Chat'), key=f"open_{chat['id']}", use_container_width=True):
                st.session_state.active_chat_id = chat['id']
                st.session_state.view_chat = True
                st.rerun()
        with col2:
            with st.popover("...", use_container_width=True):
                if st.button("Delete", key=f"delete_{chat['id']}", use_container_width=True):
                    all_chats = load_json(CHAT_HISTORY_FILE)
                    all_chats = [c for c in all_chats if c['id'] != chat['id']]
                    save_json(all_chats, CHAT_HISTORY_FILE)
                    st.rerun()
    st.markdown("---")
    # ... (Bottom Buttons: New Chat, Delete All Chats)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New General Chat", use_container_width=True):
            st.session_state.active_chat_id = None
            st.session_state.view_chat = True
            st.session_state.chat_type = "General"
            st.rerun()
    with col2:
        if st.button("🗑️ Delete All Chats", use_container_width=True, type="primary"):
            if chats:
                save_json([], CHAT_HISTORY_FILE)
                st.success("All chats have been deleted.")
                st.rerun()

def get_ai_response(messages, model_choice):
    """Gets a response from the chosen AI model."""
    if "GPT" in model_choice:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
        return response.choices[0].message.content
    else: # Gemini
        model = genai.GenerativeModel('gemini-2.5-flash')
        gemini_history = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in messages[:-1]]
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(messages[-1]['content'])
        return response.text

def display_chat_interface():
    """Displays the main chat interface, now handling pre-filled prompts."""
    # --- Back Button ---
    if st.button("← Back to History"):
        st.session_state.view_chat = False
        st.rerun()

    chat_id = st.session_state.get('active_chat_id')
    all_chats = load_json(CHAT_HISTORY_FILE)
    current_chat = next((c for c in all_chats if c['id'] == chat_id), None)
    
    # --- Dynamic Title ---
    model_choice = None
    if not current_chat:
        model_choice = st.selectbox("Choose your AI model:", (GPT_MODEL, GEMINI_MODEL), key="model_selector", disabled=bool(st.session_state.get('pre_filled_prompt')))
        if st.session_state.get('pre_selected_model'):
            model_choice = st.session_state.pre_selected_model
        st.title(f'✨ New Chat with {model_choice}')
    else:
        st.title(current_chat.get('title', 'Chat'))
        st.info(f"Model: **{current_chat['model']}**")

    # --- Display Existing Messages ---
    messages = current_chat['messages'] if current_chat else []
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Auto-run for pre-filled prompts from history page ---
    if prompt := st.session_state.get('pre_filled_prompt'):
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = get_ai_response(messages, model_choice)
                st.markdown(response)
        
        messages.append({"role": "assistant", "content": response})
        
        # Create and save the new chat
        new_chat_id = str(uuid.uuid4())
        st.session_state.active_chat_id = new_chat_id
        # Generate a concise title for the analysis prompt
        generated_title = generate_chat_title(prompt, model_choice)
        new_chat = {
            "id": new_chat_id, "title": f"Analysis: {generated_title}", 
            "model": model_choice, "messages": messages, "type": "Advanced Analysis"
        }
        all_chats.append(new_chat)
        save_json(all_chats, CHAT_HISTORY_FILE)

        # Clean up session state
        del st.session_state['pre_filled_prompt']
        if 'pre_selected_model' in st.session_state:
            del st.session_state['pre_selected_model']

    # --- Handle Regular User Input ---
    elif prompt := st.chat_input("How can I help you today?"):
        is_new_chat = not current_chat
        
        if is_new_chat:
            with st.spinner("Generating title..."):
                title = generate_chat_title(prompt, model_choice)
        
        new_messages = messages + [{"role": "user", "content": prompt}]
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                full_response = get_ai_response(new_messages, model_choice)
                st.markdown(full_response)
        
        new_messages.append({"role": "assistant", "content": full_response})

        if is_new_chat:
            st.session_state.active_chat_id = str(uuid.uuid4())
            current_chat = {
                "id": st.session_state.active_chat_id, "title": title,
                "model": model_choice, "messages": new_messages, "type": "General"
            }
            all_chats.append(current_chat)
        else:
            current_chat['messages'] = new_messages
        
        save_json(all_chats, CHAT_HISTORY_FILE)
        st.rerun()
def profile_page():
    st.title("👤 My Profile")
    
    users = load_json(USERS_FILE, default_value={})
    current_user_email = st.session_state['username']
    user_data = users.get(current_user_email)

    if not user_data:
        st.error("Could not load user data.")
        return

    # --- Profile Picture and Description Section ---
    st.header("Public Profile")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pfp_path = user_data.get('profile_picture')
        if pfp_path and os.path.exists(pfp_path):
            st.image(pfp_path, width=150)
        else:
            st.markdown("**(No photo)**")

        uploaded_pfp = st.file_uploader("Upload new profile picture", type=['png', 'jpg', 'jpeg'])
        if uploaded_pfp:
            if not os.path.exists(PROFILE_PICS_DIR):
                os.makedirs(PROFILE_PICS_DIR)
            
            # Save the file with a name derived from the user's email
            file_extension = os.path.splitext(uploaded_pfp.name)[1]
            safe_email = current_user_email.replace('@', '_').replace('.', '_')
            save_path = os.path.join(PROFILE_PICS_DIR, f"{safe_email}{file_extension}")
            
            with open(save_path, "wb") as f:
                f.write(uploaded_pfp.getbuffer())
            
            users[current_user_email]['profile_picture'] = save_path
            save_json(users, USERS_FILE)
            st.success("Profile picture updated!")
            st.rerun()

    with col2:
        description = st.text_area(
            "Profile Description", 
            value=user_data.get('description', ''),
            height=150,
            placeholder="Tell us a little about yourself..."
        )
        if st.button("Save Description"):
            users[current_user_email]['description'] = description
            save_json(users, USERS_FILE)
            st.success("Description updated!")

    st.markdown("---")

    # --- Account Settings Section ---
    st.header("Account Settings")
    
    with st.expander("Change Email"):
        with st.form("change_email_form"):
            new_email = st.text_input("New Email")
            password_verify = st.text_input("Current Password", type="password")
            submitted = st.form_submit_button("Change Email")

            if submitted:
                if not new_email or not password_verify:
                    st.warning("Please fill all fields.")
                elif new_email in users:
                    st.error("This email is already taken.")
                else:
                    hashed_verify = hashlib.sha256(password_verify.encode()).hexdigest()
                    if hashed_verify == user_data['password']:
                        users[new_email] = users.pop(current_user_email) # Update the key
                        save_json(users, USERS_FILE)
                        st.session_state['username'] = new_email
                        st.success("Email changed successfully!")
                        st.rerun()
                    else:
                        st.error("Incorrect password.")

    with st.expander("Change Password"):
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            submitted = st.form_submit_button("Change Password")

            if submitted:
                hashed_current = hashlib.sha256(current_password.encode()).hexdigest()
                if hashed_current != user_data['password']:
                    st.error("Incorrect current password.")
                elif not new_password or new_password != confirm_password:
                    st.error("New passwords do not match.")
                else:
                    users[current_user_email]['password'] = hashlib.sha256(new_password.encode()).hexdigest()
                    save_json(users, USERS_FILE)
                    st.success("Password changed successfully!")

# --- Main App Router (MODIFIED) ---
# def main():
#     """Main function to run the Streamlit app."""
#     if 'logged_in' not in st.session_state:
#         st.session_state['logged_in'] = False
#         st.session_state.view_chat = False
    
#     if st.session_state['logged_in']:
#         st.sidebar.title(f"Welcome, {st.session_state['username']}")
#         page = st.sidebar.radio("Navigation", ["Predict", "History", "Assistant"])
        
#         if st.sidebar.button("Logout"):
#             st.session_state.clear()
#             st.rerun()
            
#         if page == "Predict":
#             prediction_page()
#         elif page == "History":
#             history_page()
#         elif page == "Assistant":
#             assistant_page_router()
#     else:
#         login_page()
def main():
    """Main function to run the Streamlit app."""
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state.view_chat = False
    
    if st.session_state['logged_in']:
        st.sidebar.title(f"Welcome, {st.session_state['username']}")
        
        # Add "Profile" to the navigation options
        page = st.sidebar.radio("Navigation", ["Profile", "Predict", "History", "Assistant"])
        
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()
            
        if page == "Profile":
            profile_page()
        elif page == "Predict":
            prediction_page()
        elif page == "History":
            history_page()
        elif page == "Assistant":
            assistant_page_router()
    else:
        login_page()

# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    # Make sure the profile picture directory exists
    if not os.path.exists(PROFILE_PICS_DIR):
        os.makedirs(PROFILE_PICS_DIR)
    main()

