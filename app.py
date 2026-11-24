
import streamlit as st
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

# =====================================
# 1. LOAD ENVIRONMENT VARIABLES
# =====================================

load_dotenv()

# =====================================
# 2. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="My Personal AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# =====================================
# 3. CHATBOT HEADING
# =====================================

st.title("🤖 My Personal AI Chatbot")
st.caption("Your own Gemini-powered AI assistant")

# =====================================
# 4. GET GEMINI API KEY
# =====================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Check your .env file.")
    st.stop()

# =====================================
# 5. CONNECT TO GEMINI
# =====================================

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=5,
            initial_delay=2.0,
            max_delay=30.0,
            exp_base=2.0,
            jitter=1.0,
            http_status_codes=[503]
        )
    )
)

# =====================================
# 6. INITIALIZE CHAT HISTORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# 7. CLEAR CHAT BUTTON
# =====================================

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# =====================================
# 8. DISPLAY PREVIOUS MESSAGES
# =====================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# 9. USER INPUT
# =====================================

user_input = st.chat_input("Ask me anything...")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # =====================================
    # 10. GENERATE GEMINI RESPONSE
    # =====================================

    with st.chat_message("assistant"):

        with st.spinner("Gemini is thinking..."):

            try:

                # Prepare conversation history
                chat_history = []

                for message in st.session_state.messages:
                    role = (
                        "user"
                        if message["role"] == "user"
                        else "model"
                    )

                    chat_history.append(
                        types.Content(
                            role=role,
                            parts=[
                                types.Part(text=message["content"])
                            ]
                        )
                    )

                # Generate response
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=chat_history,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are a friendly and helpful personal "
                            "AI assistant. Give clear, simple and "
                            "useful answers."
                        )
                    )
                )

                answer = response.text

                if not answer:
                    answer = "Sorry, I could not generate a response."

                # Display response
                st.markdown(answer)

                # Save response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                error_message = str(e)

                if "503" in error_message or "UNAVAILABLE" in error_message:
                    st.error(
                        "Gemini is currently busy even after automatic "
                        "retries. Please wait a little and try again."
                    )

                elif "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                    st.error(
                        "Gemini API usage limit reached. "
                        "Please check your free-tier quota."
                    )

                else:
                    st.error(
                        f"Gemini API Error: {type(e).__name__}: {e}"
                    )
