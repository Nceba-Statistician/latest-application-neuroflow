import streamlit, os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

def sign_up(email, password):
    try:
        user = supabase.auth.sign_up({"email": email, "password": password})
        return user
    except Exception as e:
        streamlit.error(f"Registration failed: {e}")
        return None  # Important to return None on failure

def sign_in(email, password):
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return user
    except Exception as e:
        streamlit.error(f"Login failed: Please confirm your email address. {e}")
        return None  # Important to return None on failure

def sign_out():
    try:
        user = supabase.auth.sign_out()
        streamlit.session_state.user_email = None
        streamlit.rerun()
    except Exception as e:
        streamlit.error(f"Logout failed: {e}")

def main_app(user_email):
    streamlit.title("Welcome page")
    streamlit.success(f"Welcome {user_email}")
    if streamlit.button("logout"):
        sign_out()

def auth_screen():
    streamlit.title("Streamlit and Supabase Auth App")
    Options = streamlit.selectbox("Choose action", ["", "login", "sign up"])
    email = streamlit.text_input("email")
    password = streamlit.text_input("password", type="password")

    if Options == "":
        streamlit.session_state["disable"] = True
    elif Options == "sign up":
        if streamlit.button("Register"):
            user = sign_up(email, password)
            if user and user.user:  # Check if user and user attribute exist
                if user.user.email:
                    streamlit.success("Registration successful. Please log in ...")
            elif user:
                streamlit.warning("Registration successful, but email not found in response.")
            else:
                streamlit.error("Registration failed.")
    elif Options == "login":
        if streamlit.button("Login"):  # Corrected button label
            user = sign_in(email, password)
            if user and user.user:  # Check if user and user attribute exist
                if user.user.email:
                    streamlit.session_state["user_email"] = user.user.email
                    streamlit.success(f"Welcome back {email}")
                    streamlit.rerun()
                else:
                    streamlit.warning("Login successful, but email not found in response.")
            else:
                streamlit.write("")

if "user_email" not in streamlit.session_state:
    streamlit.session_state["user_email"] = None

if streamlit.session_state["user_email"]:
    main_app(streamlit.session_state["user_email"])
else:
    auth_screen()
