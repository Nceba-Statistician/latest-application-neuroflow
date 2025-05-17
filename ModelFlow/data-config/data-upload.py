import streamlit, os, pandas, numpy, datetime, json, requests, pyodbc, gspread
from gspread_dataframe import set_with_dataframe

getdatachoices = [
        "", "Upload CSV", "Upload Excel", "Add API", "Connect to SQL", "Connect to Googlesheet"
    ]
Options = streamlit.selectbox("Upload file options: ", getdatachoices)
# CSV

if Options == "":
    streamlit.session_state["disable"] = True
    streamlit.warning("Select above to continue.")
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
                full_path = os.path.join(save_path, os.path.splitext(original_file_name)[0] + ".csv")
                CSV_Object.to_csv(full_path, index=False)
                streamlit.success(f"✅ {original_file_name} successfully saved at manage-files")
        elif Save_options == "Excel":
            if streamlit.button(f"Save {original_file_name}"):
                save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, os.path.splitext(original_file_name)[0] + ".xlsx")
                CSV_Object.to_excel(full_path, index=False, engine='openpyxl')
                streamlit.success(f"✅ {original_file_name} successfully saved at manage-files")      
# Excel

elif Options == "Upload Excel":
    uploaded_Excel_file = streamlit.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
    if "show_data" not in streamlit.session_state:
        streamlit.session_state["show_data"] = False
    if uploaded_Excel_file is not None:
        original_file_name = uploaded_Excel_file.name
        Excel_Object = pandas.read_excel(uploaded_Excel_file)
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
                full_path = os.path.join(save_path, os.path.splitext(original_file_name)[0] + ".csv")
                Excel_Object.to_csv(full_path, index=False)
                streamlit.success(f"✅ {original_file_name} successfully saved at manage-files")
        elif Save_options == "Excel":
            if streamlit.button(f"Save {original_file_name}"):
                save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, os.path.splitext(original_file_name)[0] + ".xlsx")
                Excel_Object.to_excel(full_path, index=False, engine='openpyxl')
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
                            API_Object.to_excel(full_path, index=False, engine='openpyxl')
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

elif Options == "Connect to Googlesheet":
    Spreadsheet_title = streamlit.text_input("Add your spreadsheet title", placeholder="Spreadsheet title at the top left of your spreadsheet")
    Sheet_tab_name = streamlit.text_input("Add your sheet tab name", placeholder="Sheet tab name at the bottom")
    Json_key_file = streamlit.text_area(
        "Paste your json key",
        placeholder="""
        {
          "type": "",
          "project_id": "",
          "private_key_id": "",
          "private_key": "",
          "client_email": "",
          "client_id": "",
          "auth_uri": "",
          "token_uri": "",
          "auth_provider_x509_cert_url": "",
          "client_x509_cert_url": "",
          "universe_domain": ""
        }
                    """, height=200
                    )
    if Spreadsheet_title and Sheet_tab_name and Json_key_file:
        if streamlit.button("Connect Googlesheet"):
            try:
                json_key = json.loads(Json_key_file)
                with open("json_key.json", "w") as file:
                    json.dump(json_key, file)
                worksheet = gspread.service_account("json_key.json").open(Spreadsheet_title).worksheet(Sheet_tab_name)
                dict_records = worksheet.get_all_records()
                streamlit.success("Connected successfully to your Spreadsheet!")
                streamlit.write(pandas.DataFrame(dict_records).head())
            except json.JSONDecodeError:
                streamlit.error("Invalid JSON format. Please check your pasted key.")
            except gspread.exceptions.SpreadsheetNotFound:
                streamlit.error(f"Spreadsheet '{Spreadsheet_title}' not found.")
            except gspread.exceptions.WorksheetNotFound:
                streamlit.error(f"Sheet tab '{Sheet_tab_name}' not found in '{Spreadsheet_title}'.")
            except Exception as e:
                streamlit.error(f"An error occurred: {e}")

    else:
        streamlit.warning("Please fill in all the connection details.")    


# http://127.0.0.1:8000/items_processed
