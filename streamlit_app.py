import streamlit, pandas, numpy, datetime, json, requests, pyodbc, os, altair 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from supabase import create_client, Client
from plotly import express

streamlit.set_page_config(page_title="neuroflow application", layout="wide", initial_sidebar_state="auto")


supabase_url = streamlit.secrets["SUPABASE_URL"] 
supabase_key = streamlit.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

def sign_up(email, password):
    try:
        user = supabase.auth.sign_up({"email": email, "password": password})
        return user
    except Exception as e:
        streamlit.error(f"Registration failed: {e}")
        return None  

def sign_in(email, password):
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return user
    except Exception as e:
        streamlit.error(f"Login failed: {e}.")
        return None 

def sign_out():
    try:
        supabase.auth.sign_out()
        streamlit.session_state["user_email"] = None
        streamlit.rerun()

    except Exception as e:
        streamlit.error(f"Logout failed: {e}")

def forgot_password(email):
    try:
        response = supabase.auth.reset_password_for_email(email)
        if response.error:
            streamlit.error(f"Password reset request failed: {response.error.message}")
            return False
        else:
            streamlit.success(f"A password reset link has been sent to {email}. Please check your inbox (and spam folder).")
            return True
    except Exception as e:
        streamlit.error(f"An unexpected error occurred: {e}")
        return False        

def main_app(user_email):

    streamlit.markdown("""<style> .title { position: absolute; font-size: 20px; right: 10px; top: 10px} </style> """,
                       unsafe_allow_html=True
                       )
    streamlit.markdown(""" <style> .footer { position: fixed; bottom: 0; left: 0; width: 20%; background-color: rgba(0, 0, 0, 0.05); padding: 10px; font-size: 12px; text-align: center;} </style> """,
                       unsafe_allow_html=True
                       )
    streamlit.markdown(""" <style> .emotion-cache {vertical-align: middle; overflow: hidden; color: inherit; fill: currentcolor; display: inline-flex; -webkit-box-align: center; align-items: center; font-size: 1.25rem; width: 1.25rem; height: 1.25rem; flex-shrink: 0; } </style>""",
                       unsafe_allow_html=True
                       )
    # streamlit.markdown("""<div class="title"> AI Architect application</div>""", unsafe_allow_html=True)

    landing_page = streamlit.Page(
        "profile_page.py", title="Profile", default=True
        )
    
    data_upload = streamlit.Page(
        "ModelFlow/data-config/data-upload.py", title="Data upload", icon=":material/upload:", default=False
        )
    manage_files = streamlit.Page(
        "ModelFlow/data-config/manage-files.py", title="Manage files", icon=":material/files:", default=False
        )
    application = streamlit.Page(
        "ModelFlow/model-history.py", title="Application", icon=":material/history:", default=False
        )
    neuro_flow = streamlit.Page(
        "ModelFlow/neuro-flow.py", title="Neuro Flow", icon=":material/analytics:", default=False
        )
    visuals = streamlit.Page(
        "Reports/visuals.py", title="Visuals", icon=":material/analytics:", default=False
        )
    dashboard = streamlit.Page(
        "Reports/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=False
        )
    data_cleaning = streamlit.Page(
        "Tools/data-cleaning.py", title="Data Cleaning", icon=":material/cleaning:", default=False
        )
    data_migration = streamlit.Page(
        "Tools/data-migration.py", title="Data Migration", icon=":material/moving:", default=False
        )
    search = streamlit.Page(
        "Tools/search.py", title="Search", icon=":material/search:", default=False
        )

    streamlit.navigation({
        "Model Flow": [landing_page, data_upload, manage_files, neuro_flow, application],
        "Reports": [dashboard, visuals],
        "Tools": [data_migration, data_cleaning, search]
        }).run()
    
    streamlit.sidebar.write("Currently logged in as:")
    streamlit.sidebar.caption(f"{user_email}")
    if streamlit.sidebar.button("Sign Out"):
        sign_out()

def auth_screen():
    streamlit.markdown("""<div style="display: flex; justify-content: center; align-items: center">Welcome</div>""", unsafe_allow_html=True)
    email = streamlit.text_input("Email", key="login_email", placeholder="Enter your email")
    password = streamlit.text_input("Password", key="login_password", placeholder="Enter your password", type="password")
    landing_for_auth = streamlit.Page("landing_page.py", title="Landing Page", icon=":material/menu:")
    if streamlit.button("Login"):
        user = sign_in(email, password)
        if user and user.user:
            if user.user.email:
                streamlit.session_state["user_email"] = user.user.email
                streamlit.success(f"Welcome back {email}")
                streamlit.rerun()
            else:
                streamlit.warning("Login successful, but email not found in response.")
        else:
            streamlit.write("")

    streamlit.markdown("""<div style="display: flex; justify-content: center; align-items: center">Don't have an account?</div>""", unsafe_allow_html=True)
    if streamlit.checkbox("Sign up", key="Register"):
        email = streamlit.text_input("Email", key="signup_email", placeholder="Enter your email")
        password = streamlit.text_input("Password", key="signup_password", placeholder="Enter your password", type="password")
        if streamlit.button("Register"):
            user = sign_up(email, password)
            if user and user.user:
                if user.user.email:
                    streamlit.success("Registration successful. Please log in ...")
                    streamlit.info("Confirmation sent to you - please confirm then log-in")
            elif user:
                streamlit.warning("Registration successful, but email not found in response.")
            else:
                streamlit.error("Registration failed.")
    streamlit.markdown("---")
    streamlit.markdown("""<div style="display: flex; justify-content: center; align-items: center">Forgot password?</div>""", unsafe_allow_html=True)
    if streamlit.checkbox("Reset password"):
        streamlit.info("Enter your email address below to receive a password reset link.")
        reset_email = streamlit.text_input("Email for password reset:", placeholder="Enter your email")
        if streamlit.button("Send Reset Link"):
            if reset_email:
                forgot_password(reset_email)
            else:
                streamlit.warning("Please enter your email address.") 

    streamlit.navigation({
        "Home": [landing_for_auth]
        }).run()
            

if "user_email" not in streamlit.session_state:
    streamlit.session_state["user_email"] = None
    

if streamlit.session_state["user_email"]:
    main_app(streamlit.session_state["user_email"])
else:
    auth_screen()
