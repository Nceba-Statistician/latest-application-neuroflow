import streamlit, os, pandas, numpy, datetime, json, requests, pyodbc
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.callbacks import TensorBoard
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
    uploaded_CSV_file = streamlit.file_uploader("Upload a CSV file", type=["csv"])
    if "show_data" not in streamlit.session_state:
        streamlit.session_state["show_data"] = False
    if uploaded_CSV_file is not None:
        original_file_name = uploaded_CSV_file.name
        CSV_Object = pandas.read_csv(uploaded_CSV_file)
        streamlit.success(f"{original_file_name} uploaded successfully!")
        if streamlit.checkbox(f"Preview {original_file_name}"):
            streamlit.write(CSV_Object.head())
        Save_options = streamlit.selectbox(
            "Save your file:",
            ["Select format?", "CSV", "Excel"]
            )
        if Save_options == "Select format?":
            streamlit.session_state["disable"] = True
            streamlit.warning("Select format of your choice!")
        elif Save_options == "CSV":
            if streamlit.button(f"Save {original_file_name}"):
                save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, original_file_name)
                CSV_Object.to_csv(full_path, index=False)
                streamlit.success(f"✅ {original_file_name} successfully saved at manage-files")
        elif Save_options == "Excel":
            if streamlit.button(f"Save {original_file_name}"):
                save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, original_file_name)
                CSV_Object.to_excel(full_path, index=False)
                streamlit.success(f"✅ {original_file_name} successfully saved at manage-files")      
# Excel

elif Options == "Upload Excel":
    uploaded_Excel_file = streamlit.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
    original_file_name = uploaded_Excel_file.name
    if "show_data" not in streamlit.session_state:
        streamlit.session_state["show_data"] = False
    if uploaded_Excel_file is not None:
        Excel_Object = pandas.read_csv(uploaded_Excel_file)
        streamlit.success(f"{original_file_name} uploaded successfully!")
        if streamlit.checkbox(f"Preview {original_file_name}"):
            streamlit.write(Excel_Object.head())
        Save_options = streamlit.selectbox(
            "Save your file:",
            ["Select format?", "CSV", "Excel"]
            )
        if Save_options == "Select format?":
            streamlit.session_state["disable"] = True
        elif Save_options == "CSV":
            if streamlit.button(f"Save {original_file_name}"):
                save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, original_file_name)
                Excel_Object.to_csv(full_path, index=False)
                streamlit.success(f"✅ {original_file_name} successfully saved at manage-files")
        elif Save_options == "Excel":
            if streamlit.button(f"Save {original_file_name}"):
                save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, original_file_name)
                Excel_Object.to_excel(full_path, index=False)
                streamlit.success(f"✅ {original_file_name} successfully saved at manage-files")       
# API
        
elif Options == "Add API":
    url = streamlit.text_input("Enter your API URL:", "")
    if url == "":
        streamlit.write("")
    else:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                API_Object = pandas.DataFrame(response.json())
            else:
                streamlit.error("🚨 Failed to fetch data")
                if streamlit.checkbox("Preview error"):
                    streamlit.write(f"{response.status_code}")
            if "show_data" not in streamlit.session_state:
                streamlit.session_state["show_data"] = False
            elif API_Object is not None:
                streamlit.success("Server response was successful!")
                if streamlit.checkbox("Preview object"):
                    streamlit.write(API_Object.head())  
                Save_options = streamlit.selectbox(
                    "Save your file:",
                    ["Select format?", "CSV", "Excel"]
                )
                if Save_options == "Select format?":
                    streamlit.session_state["disable"] = True
                elif Save_options == "CSV":
                    file_name_input = streamlit.text_input("Enter a file name (without extension):", "")
                    if streamlit.button(f"Save {file_name_input}"):
                        if not file_name_input.strip():
                            streamlit.warning("Please enter a valid file name.")
                        else:    
                            save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                            os.makedirs(save_path, exist_ok=True)
                            full_path = os.path.join(save_path, f"{file_name_input}.csv")
                            API_Object.to_csv(full_path, index=False)
                            streamlit.success(f"{file_name_input} successful saved at manage-files")
                elif Save_options == "Excel":
                    file_name_input = streamlit.text_input("Enter a file name (without extension):", "")
                    if streamlit.button(f"Save {file_name_input}"):
                        if not file_name_input.strip():
                            streamlit.warning("Please enter a valid file name.")
                        else:    
                            save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                            os.makedirs(save_path, exist_ok=True)
                            full_path = os.path.join(save_path, f"{file_name_input}.xlsx")
                            API_Object.to_excel(full_path, index=False)
                            streamlit.success(f"✅ {file_name_input} successful saved at manage-files")        

        except requests.exceptions.RequestException as e:
            streamlit.write("🚨 Server response failed!")
            if "show_error" not in streamlit.session_state:
                streamlit.session_state["show_error"] = False
                if streamlit.checkbox("Preview error"):
                    streamlit.write(f"API url error: \n{e}")

# SQL

elif Options == "Connect to SQL":
    streamlit.write("Enter your SQL connection details:")
    streamlit.markdown(
        "[Download SQL Server ODBC Driver](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver16)"
    )
    Driver_Options = [
        "ODBC Driver 17 for SQL Server", "ODBC Driver 18 for SQL Server"
    ]
    Select_Driver = streamlit.selectbox("Select Driver", Driver_Options)
    if Select_Driver in ["ODBC Driver 17 for SQL Server", "ODBC Driver 18 for SQL Server", "Other"]:
        Driver = streamlit.text_input("Driver:", Select_Driver)
    else:
        Driver = streamlit.text_input("Driver", "")    
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
                streamlit.success("Connected to SQL Server successfully!")        
                if "show_data_tables" not in streamlit.session_state:
                    streamlit.session_state["show_data_tables"] = False  
                query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG = ?"
                col_exec_view_tables, col_exec_not_view_tables = streamlit.columns(2)
                # show_data_tables
                with col_exec_view_tables:
                    if streamlit.button(f"View list of tables on this database {Database}"):
                        streamlit.session_state["show_data_tables"] = True
                with col_exec_not_view_tables:
                    if streamlit.button(f"Hide list of tables on this database {Database}"):
                        streamlit.session_state["show_data_tables"] = False      
                if streamlit.session_state["show_data_tables"]:
                        list_tables = pandas.DataFrame(conn.cursor().execute(query, (Database,)).fetchall())
                        conn.commit()
                        streamlit.write(f"List of tables:/n {list_tables}")
                        Data_extract = streamlit.text_input("Write your script here to extract data: ","")
                        if Data_extract is not None:
                            try:
                                pandas.DataFrame(conn.cursor().execute(Data_extract).fetchall()).head()
                                conn.commit()
                            except Exception as e:
                                streamlit.write("Data was not fetched!")
                                if "show_fetch_data_error" not in streamlit.session_state:
                                    streamlit.session_state["show_fetch_data_error"] = False
                                view_fetch_data_error, hide_fetch_data_error = streamlit.columns(2)
                                with view_fetch_data_error:
                                    if streamlit.button("See error on your query"):
                                        streamlit.session_state["show_fetch_data_error"] = True
                                with hide_fetch_data_error:
                                    if streamlit.button("Hide this error on your query"):
                                        streamlit.session_state["show_fetch_data_error"] = False
                                if streamlit.session_state["show_fetch_data_error"]:
                                    streamlit.write(f"Error on your query: {e}")
            except Exception as e:
                streamlit.write("Connection to SQL Server failed!")


# http://127.0.0.1:8000/items_processed
