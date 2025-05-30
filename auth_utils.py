
import streamlit as st
from db_utils import register_user, verify_user

def init_auth_session():
    """Initialize authentication session state"""
    if 'is_authenticated' not in st.session_state:
        st.session_state['is_authenticated'] = False
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'name' not in st.session_state:
        st.session_state['name'] = None
    if 'show_login' not in st.session_state:
        st.session_state['show_login'] = False
    if 'show_register' not in st.session_state:
        st.session_state['show_register'] = False

def login(email, password):
    """Handle user login"""
    user = verify_user(email, password)
    if user:
        st.session_state['is_authenticated'] = True
        st.session_state['user_id'] = user['id']
        st.session_state['name'] = user['name']
        return True, user['id']
    return False, "Invalid email or password"

def logout():
    """Handle user logout"""
    st.session_state['is_authenticated'] = False
    st.session_state['user_id'] = None
    st.session_state['name'] = None
    st.session_state['show_login'] = False
    st.session_state['show_register'] = False

def register(username, email, password):
    """Handle user registration"""
    user_id = register_user(username, email, password)
    if user_id:
        return True, "Registration successful"
    return False, "Registration failed"

def require_auth():
    """Check if user is authenticated, if not show login"""
    if not st.session_state['is_authenticated']:
        st.session_state['show_login'] = True
        return False
    return True
