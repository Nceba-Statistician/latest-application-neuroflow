import streamlit, pandas, numpy, datetime, json, requests, pyodbc
from keras.api.models import Sequential
from keras.api.layers import Input, Dense, Dropout
from keras.api.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

getdatachoices = [
        "select your option", "Upload CSV", "Upload Excel", "Add API", "Connect to SQL"
    ]
Options = streamlit.selectbox("Upload file options: ", getdatachoices)
# CSV

if Options == "select your option":
    streamlit.session_state["disable"] = True
    streamlit.warning("Select options above if your data is ready and if not visit preprocessing")
elif Options == "Upload CSV":
    uploaded_file = streamlit.file_uploader("Upload a CSV file", type=["csv"])
    if "show_data" not in streamlit.session_state:
        streamlit.session_state.show_data = False
    if uploaded_file is not None:
        file_has_been_uploaded = pandas.read_csv(uploaded_file)
        streamlit.write("File uploaded successfully!")
    col1, col2 = streamlit.columns(2)
    with col1:
        if streamlit.button("View data"):
            streamlit.session_state.show_data = True
    with col2:
        if streamlit.button("Hide data"):
            streamlit.session_state.show_data = False
    if streamlit.session_state.show_data:        
        streamlit.write(file_has_been_uploaded.head())
# Excel

elif Options == "Upload Excel":
    uploaded_file = streamlit.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
    if "show_data" not in streamlit.session_state:
        streamlit.session_state.show_data = False
    if uploaded_file is not None:
        file_has_been_uploaded = pandas.read_excel(uploaded_file)
        streamlit.write("File uploaded successfully!")
    col1, col2 = streamlit.columns(2)
    with col1:
        if streamlit.button("View data"):
            streamlit.session_state.show_data = True
    with col2:
        if streamlit.button("Hide data"):
            streamlit.session_state.show_data = False
    if streamlit.session_state.show_data:        
        streamlit.write(file_has_been_uploaded.head())
# API
        
elif Options == "Add API":
    url = streamlit.text_input("Enter your API URL:", "")
    if url == "":
        streamlit.write("")
    else:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                API_data = pandas.DataFrame(response.json())
            else:
                streamlit.error(f"Failed to fetch data: {response.status_code}")    
            if "show_data" not in streamlit.session_state:
                streamlit.session_state["show_data"] = False
            elif API_data is not None:
                API_data_loaded = API_data
                streamlit.write("Server response was successful!")
            col1, col2 = streamlit.columns(2)
            with col1:
                if streamlit.button("View data"):
                    streamlit.session_state.show_data = True
            with col2:
                if streamlit.button("Hide data"):
                    streamlit.session_state.show_data = False
            if streamlit.session_state.show_data:
                streamlit.write(API_data_loaded.head())
        except requests.exceptions.RequestException as e:
            streamlit.write("Server response failed!")
            if "show_error" not in streamlit.session_state:
                streamlit.session_state.show_error = False
            col_left, col_right = streamlit.columns(2)
            with col_left:
                if streamlit.button("See what was the error"):
                    streamlit.session_state.show_error = True
            with col_right:
                if streamlit.button("You can hide the error"):
                    streamlit.session_state.show_error = False
            if streamlit.session_state.show_error:
                streamlit.write(f"Error fetching data: {e}")
# SQL

elif Options == "Connect to SQL":
    Conn_SQL = streamlit.selectbox(
        "Connect to SQL Server", 
        ["", "Connect to SQL Server"]
    )
    
    def SQLServerDetails():
        Driver_Options = ["ODBC Driver 17 for SQL Server", "ODBC Driver 18 for SQL Server", "Other"]
        Select_Driver = streamlit.selectbox("Select Driver", Driver_Options)
        Driver = streamlit.text_input("Driver:", Select_Driver if Select_Driver != "Other" else "")
        Server = streamlit.text_input("Server:", "")
        Database = streamlit.text_input("Database:", "")
        UserID = streamlit.text_input("UserID:", "")
        Password = streamlit.text_input("Password:", type="password")
        if streamlit.button("Connect to SQL Server"):
            if not all ([Driver, Server, Database, UserID, Password]):
                streamlit.warning("Please fill in all fields")
            else:
                try:
                    conn = pyodbc.connect(f"Driver={Driver};Server={Server};Database={Database};UID={UserID};PWD={Password};TrustServerCertificate=yes")
                    conn.commit()
                    streamlit.success("Connected to SQL Server successfully!")
                except Exception as e:
                    if streamlit.checkbox("Preview error"):
                        streamlit.write(f"Issue on your connection:/n{e}")         
        return Driver, Server, Database, UserID, Password
    if Conn_SQL == "":
        streamlit.session_state["disable"] = True
    if Conn_SQL == "Connect to SQL Server":
        streamlit.markdown(
            "[Download SQL Server ODBC Driver](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver16)"
        )
        streamlit.write("Enter your SQL connection details:")  
        Driver, Server, Database, UserID, Password = SQLServerDetails()

# http://127.0.0.1:8000/items_processed
# localhost\DEMODEV2022

