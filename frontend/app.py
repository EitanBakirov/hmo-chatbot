"""
Bilingual HMO Chatbot Frontend
Handles:
- User information collection
- HMO service inquiries
- RTL/LTR text display
"""

import streamlit as st
import requests
import re
import json
from time import time
from typing import Optional
from shared.logger_config import logger
from shared.monitoring import monitoring

# API Configuration
API_URL = "http://backend:8000/ask"


# Page configuration
st.set_page_config(page_title="Health Fund Chatbot", page_icon="💬")

# Initialize session state
def init_session_state() -> None:
    """Initialize session state variables if they don't exist"""
    if "user_info" not in st.session_state:
        st.session_state.user_info = {}
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_phase" not in st.session_state:
        st.session_state.current_phase = "collection"
    # Add initialization flag
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

# Detect text direction (Hebrew/English)
def detect_language_direction(text: str) -> str:
    """
    Detect if text is primarily Hebrew (RTL) or English/other (LTR)
    Returns 'rtl' for right-to-left text, 'ltr' for left-to-right
    """
    hebrew_chars = re.findall(r'[\u0590-\u05FF]', text)
    english_chars = re.findall(r'[A-Za-z]', text)
    total = len(hebrew_chars) + len(english_chars)
    return "rtl" if total and len(hebrew_chars) / total > 0.6 else "ltr"

# Render a message with proper text direction
def render_message(content: str, role: str) -> None:
    """
    Renders chat messages with proper text direction
    
    Args:
        content: Message text
        role: "user" or "assistant"
    """
    direction = detect_language_direction(content)
    st.markdown(f"<div dir='{direction}'>{content}</div>", unsafe_allow_html=True)

# Validate user input
def validate_input(user_message: str) -> tuple:
    """Validate user input before sending to backend"""
    if not user_message:
        return False, "Empty message"
    if len(user_message) > 1000:
        return False, "Message too long (max 1000 characters)"
    return True, ""

# Send message to backend
def send_to_backend(payload: dict) -> tuple:
    """Send payload to backend API and handle response"""
    try:
        start_time = time()
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        logger.info("Backend request successful",
            duration_ms=int((time() - start_time) * 1000),
            phase=payload.get("phase", "unknown"),
            endpoint=API_URL
        )
        return response.json(), None
        
    except requests.exceptions.Timeout:
        logger.error("Backend request timeout",
            phase=payload.get("phase", "unknown"),
            endpoint=API_URL
        )
        return None, "Request timed out. Please try again."
        
    except requests.exceptions.ConnectionError:
        logger.error("Backend connection error",
            phase=payload.get("phase", "unknown"),
            endpoint=API_URL
        )
        return None, "Connection error. Please check your internet connection."
        
    except requests.exceptions.HTTPError as e:
        logger.error("Backend HTTP error",
            status_code=e.response.status_code,
            phase=payload.get("phase", "unknown"),
            endpoint=API_URL
        )
        return None, f"Server error: {e.response.status_code}"
        
    except Exception as e:
        logger.error("Unexpected error",
            error_type=type(e).__name__,
            error_details=str(e),
            phase=payload.get("phase", "unknown")
        )
        return None, f"Error: {str(e)}"

# Prepare payload for Q&A phase
def prepare_qa_payload(user_message: str, user_info: dict, history: list) -> dict:
    """
    Prepares payload for QA phase with context
    
    Args:
        user_message: Current user input
        user_info: Collected user details
        history: Conversation history
        
    Returns:
        Formatted payload for backend
    """
    return {
        "user_info": user_info,
        "history": history,
        "question": user_message,
        "language": user_info.get("preferred_language", "he"),
        "phase": "qa",
        "hmo": user_info.get("hmo", ""),
        "tier": user_info.get("tier", "")
    }

def render_page_header() -> None:
    """Render the page title and welcome message"""
    st.title("🏥 Health Fund Chatbot")
    st.markdown("""
    #### Welcome to the HMO Services Assistant! 
    This chatbot helps you access information about Israeli health fund services based on your membership details. 
    It will guide you through:
    1. Collecting your personal information
    2. Verifying your HMO membership
    3. Answering questions about available services
    
    Let's get started!
    """)

def handle_initial_greeting() -> None:
    """Handle the initial greeting if not already initialized"""
    if not st.session_state.initialized:
        payload = {
            "user_info": {},
            "history": [],
            "question": "START_CHAT",
            "language": "he",
            "phase": "collection"
        }
        
        result, error = send_to_backend(payload)
        if not error and result:
            st.session_state.history.append({
                "role": "assistant", 
                "content": result["answer"]
            })
            st.session_state.initialized = True

def display_chat_history() -> None:
    """Displays full conversation history with proper formatting"""
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            render_message(msg["content"], msg["role"])

def process_user_input(user_message: str) -> bool:
    """Process and validate user input"""
    is_valid, error_msg = validate_input(user_message)
    if not is_valid:
        st.error(error_msg)
        return False
        
    st.session_state.history.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        render_message(user_message, "user")
    return True

def get_phase_payload(user_message: str) -> dict:
    """Get the appropriate payload based on current phase"""
    if st.session_state.current_phase == "qa":
        return prepare_qa_payload(
            user_message, 
            st.session_state.user_info,
            st.session_state.history
        )
    return {
        "user_info": st.session_state.user_info,
        "history": st.session_state.history,
        "question": user_message,
        "language": "he",
        "phase": "collection"
    }

def handle_bot_response(result: dict, error: Optional[str]) -> None:
    """Process and display bot response with clean phase transition handling"""
    if error:
        monitoring.log_conversation(
            phase=st.session_state.current_phase,
            success=False,
            language=st.session_state.user_info.get("preferred_language", "he")
        )
        st.error(error)
        return
        
    answer = result["answer"]
    
    # Handle clean phase transition
    if result.get("phase_transition", False) and "user_info" in result:
        # Update session state with collected user info
        st.session_state.user_info = result["user_info"]
        st.session_state.current_phase = "qa"
        
        logger.info("Phase transition completed",
            from_phase="collection",
            to_phase="qa",
            user_info_fields=list(result["user_info"].keys())
        )
        
        # Show success message
        st.success("✅ Data collection completed! Now you can ask about your benefits.")
    
    # Log successful interaction
    monitoring.log_conversation(
        phase=st.session_state.current_phase,
        success=True,
        language=st.session_state.user_info.get("preferred_language", "he")
    )
    
    # Add to history and display
    st.session_state.history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        render_message(answer, "assistant")

def main():
    """Main application function"""
    try:
        # Initialize session state
        init_session_state()
        logger.info("Application started", 
            initialized=st.session_state.initialized,
            phase=st.session_state.current_phase
        )
        
        # Render page header
        render_page_header()
        
        # Handle initial greeting
        handle_initial_greeting()
        
        # Display chat history
        display_chat_history()
        
        # Get user input
        user_message = st.chat_input("Type your message here...")
        
        if user_message:
            # Process user input
            if not process_user_input(user_message):
                return
                
            # Get phase-specific payload
            payload = get_phase_payload(user_message)
            
            # Send to backend and handle response
            result, error = send_to_backend(payload)
            handle_bot_response(result, error)
            
    except Exception as e:
        logger.error("Application error",
            error_type=type(e).__name__,
            error_details=str(e)
        )
        st.error("An unexpected error occurred. Please refresh the page.")

# Run the main application
if __name__ == "__main__":
    main()