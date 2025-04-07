import streamlit, pandas, numpy, datetime, json, requests, pyodbc
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

streamlit.markdown("""<style> .title { position: absolute; font-size: 20px; right: 10px; top: 10px} </style> """,
    unsafe_allow_html=True
)
streamlit.markdown(""" <style> .footer { position: fixed; bottom: 0; left: 0; width: 20%; background-color: rgba(0, 0, 0, 0.05); padding: 10px; font-size: 12px; text-align: center;} </style> """,
    unsafe_allow_html=True
)
streamlit.markdown(""" <style> .emotion-cache {vertical-align: middle; overflow: hidden; color: inherit; fill: currentcolor; display: inline-flex; -webkit-box-align: center; align-items: center; font-size: 1.25rem; width: 1.25rem; height: 1.25rem; flex-shrink: 0; } </style>""",
    unsafe_allow_html=True
)
streamlit.markdown("""<div class="title"> AI Architect application</div>""", unsafe_allow_html=True)

log_in = streamlit.Page(
    "Account/log-in.py", title="Log in", icon=":material/login:", default=False
)
log_out = streamlit.Page(
    "Account/log-out.py", title="Log out", icon=":material/logout:", default=False
)
users = streamlit.Page(
    "Account/users.py", title="Users", icon=":material/people:", default=False
)
data_upload = streamlit.Page(
    "ModelFlow/data-config/data-upload.py", title="Data upload", icon=":material/upload:", default=False
)
manage_files = streamlit.Page(
    "ModelFlow/data-config/manage-files.py", title="manage files", icon=":material/files:", default=False
)
model_history = streamlit.Page(
    "ModelFlow/model-history.py", title="Model History", icon=":material/history:", default=False
)
neuro_flow = streamlit.Page(
    "ModelFlow/neuro-flow.py", title="Neuro Flow", icon=":material/analytics:", default=False
)
bug_reports = streamlit.Page(
    "Reports/bug-reports.py", title="Bug Reports", icon=":material/report:", default=False
)
dashboard = streamlit.Page(
    "Reports/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=False
)
system_alerts = streamlit.Page(
    "Reports/system-alerts.py", title="System Alerts", icon=":material/warning:", default=False
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

# list

data_config_list = ("Data Configuration", [data_upload, manage_files])
if data_config_list == "data_upload":
    data_upload,
elif data_config_list == "data_conn":
    manage_files

streamlit.navigation({
    "Account": [log_in, log_out, users],
    "Model Flow": [data_upload, manage_files, neuro_flow, model_history],
    "Reports": [dashboard, bug_reports, system_alerts],
    "Tools": [data_migration, data_cleaning, search]
}).run()


