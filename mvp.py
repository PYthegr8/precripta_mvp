import os
from openai import OpenAI
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import av
import cv2
import base64
import requests
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

# Streamlit app title
st.title("Prescripta")

# Initialize session state for capturing image
if 'capture' not in st.session_state:
    st.session_state['capture'] = False
if 'captured_image' not in st.session_state:
    st.session_state['captured_image'] = None
if 'final_output' not in st.session_state:
    st.session_state['final_output'] = ""


# Input fields for user information
name = st.text_input("Name")
age = st.number_input("Age", min_value=0)
weight = st.number_input("Weight (in pounds)", min_value=0)
height = st.number_input("Height (in feet)", min_value=0)
allergies = st.text_area("Allergies")
previous_conditions = st.text_area("Previous Health Conditions")

def transform(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")
    if st.session_state['capture']:
        st.session_state['captured_image'] = img
        st.session_state['capture'] = False
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="streamer",
    video_frame_callback=transform,
    sendback_audio=False,
    media_stream_constraints={"video": True, "audio": False},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, style={"width": "100%"}),
)

# Attach the ScriptRunContext to the media processor thread
if webrtc_ctx and webrtc_ctx.state.playing:
    ctx = get_script_run_ctx()
    if ctx:
        for t in threading.enumerate():
            if isinstance(t, threading.Thread) and not hasattr(t, '_script_run_ctx'):
                add_script_run_ctx(t)

if st.button("Capture Image"):
    st.session_state['capture'] = True

captured_image = st.session_state['captured_image']

if captured_image is not None:
    st.image(captured_image, channels="BGR", caption="Captured Image")

    if st.button("Analyze Medication"):
        # Save the image to a file
        cv2.imwrite('document.jpg', captured_image)
        
        # Encode the image to base64
        with open('document.jpg', "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare the payload for the OpenAI API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Provide the drug name in the picture. no extra information"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        # Make the API request to OpenAI
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # Display the response
        if response.status_code == 200:
            result = response.json()
            drug_name = result['choices'][0]['message']['content'].strip()

            # Scrape drug info from Drugs.com
            def scrape_drug_info(drug_name):
                drug_name = drug_name.lower().replace(' ', '-')
                url = f"https://www.drugs.com/{drug_name}.html"
                response = requests.get(url)

                if response.status_code != 200:
                    return {"error": "Failed to retrieve page"}

                soup = BeautifulSoup(response.content, 'html.parser')

                drug_info = {
                    "description": "Not found",
                    "uses": "Not found",
                    "side_effects": "Not found",
                    "warnings": "Not found",
                    "interactions": "Not found",
                    "dosage": "Not found"
                }

                def extract_description():
                    description_text = []
                    description_section = soup.find('section', class_='contentBox')
                    if description_section:
                        paragraphs = description_section.find_all('p')
                        for paragraph in paragraphs:
                            description_text.append(paragraph.get_text(strip=True))
                    return "\n\n".join(description_text) if description_text else "Not found"

                def extract_info(section_id):
                    h2_tag = soup.find('h2', {'id': section_id, 'class': 'ddc-anchor-offset'})
                    if h2_tag:
                        info_text = []
                        next_sibling = h2_tag.find_next_sibling()
                        while next_sibling and next_sibling.name == 'p':
                            info_text.append(next_sibling.get_text(strip=True))
                            next_sibling = next_sibling.find_next_sibling()
                        return "\n\n".join(info_text) if info_text else "Not found"
                    return "Not found"

                drug_info["description"] = extract_description()
                drug_info["uses"] = extract_info("uses")
                drug_info["side_effects"] = extract_info("side-effects")
                drug_info["warnings"] = extract_info("warnings")
                drug_info["interactions"] = extract_info("interactions")
                drug_info["dosage"] = extract_info("dosage")

                return drug_info

            # Fetch the drug information
            drug_info = scrape_drug_info(drug_name)
            if "error" in drug_info:
                st.error(drug_info["error"])
            else:
                # Process the combined information using OpenAI API
                user_info = f"Name: {name}, Age: {age}, Weight: {weight} pounds, Height: {height} feet, Allergies: {allergies}, Previous Health Conditions: {previous_conditions}"
                drug_details = f"Description: {drug_info['description']}\nUses: {drug_info['uses']}\nSide Effects: {drug_info['side_effects']}\nWarnings: {drug_info['warnings']}\nInteractions: {drug_info['interactions']}\nDosage: {drug_info['dosage']}"
                
                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Here is the patient's information: {user_info}. And here is the drug information: {drug_details}. Please provide tailored advice based on the user's specifics, including dosage recommendations, precautions, and interactions.talk to the person directly with direct person pronoun(using you) remove all other unimportant information"
                        }
                    ],
                    "max_tokens": 500
                }

                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                if response.status_code == 200:
                    result = response.json()
                    final_response = result['choices'][0]['message']['content'].strip()
                    st.session_state['final_output'] = final_response
                    st.write(final_response)
                else:
                    st.error("Failed to process the information. Please try again.")
        else:
            st.error("Failed to analyze the document. Please try again.")

if st.session_state['final_output']:
    client = OpenAI( api_key= 'sk-svcacct-G90szKWUkd7xTbKCwQe5T3BlbkFJOsWcgMqBnhSs4wj62PM4')
    response = client.audio.speech.create(
                    model="tts-1",
                    voice="echo",
                    input= st.session_state['final_output']
                )
    response.write_to_file("output.mp3")
    with open("output.mp3", "rb") as audio_file:
        st.audio(audio_file, format='audio/mp3')