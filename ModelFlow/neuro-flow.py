import streamlit, tensorflow, os, pandas, numpy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
import datetime

streamlit.markdown("""<style> .font {font-size: 5px; font-weight: bold; background-color: green} </style> """,
    unsafe_allow_html=True
)
streamlit.markdown("""<style> .titlemodel { position: absolute; font-size: 20px; left: 10px; top: 10px} </style> """,
    unsafe_allow_html=True
)
streamlit.markdown(""" <style> .footer { position: fixed; bottom: 0; left: 0; width: 20%; background-color: rgba(0, 0, 0, 0.05); padding: 10px; font-size: 12px; text-align: center;} </style> """,
    unsafe_allow_html=True
)
streamlit.markdown(""" <style> .emotion-cache {vertical-align: middle; overflow: hidden; color: inherit; fill: currentcolor; display: inline-flex; -webkit-box-align: center; align-items: center; font-size: 1.25rem; width: 1.25rem; height: 1.25rem; flex-shrink: 0; } </style>""",
    unsafe_allow_html=True
)
streamlit.markdown("""<div class="titlemodel">Neural Network Model Builder for Prediction</div>""", unsafe_allow_html=True)

streamlit.subheader("")
Action_options = ["Select an action", "Select model fields", "Transform field values", "Update field data types"]
selected_action_option = streamlit.selectbox("Choose an action:", Action_options, key="selectbox_action")

if selected_action_option == "Select an action":
    streamlit.session_state["disable"] = True
    streamlit.info("Please select an action to continue.")
elif selected_action_option == "Select model fields": 
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = ["Select file to adjust fields"] + saved_files
    selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices, key="selectbox_gen")
    if selected_file == "Select file to adjust fields":
        streamlit.session_state["disable"] = True
        streamlit.info("Please select file to continue.")
    else:
        try:
            file_path = os.path.join(save_path, selected_file)
            if selected_file.endswith(".csv"):
                records = pandas.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                records = pandas.read_excel(file_path)
            if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_object_action"):
                streamlit.write(records.head())          

            columns = records.columns.tolist()
            select_columns = streamlit.multiselect(
                "Select columns for your model", columns
                )
            if select_columns:
                selected_records = records[select_columns]
                if selected_records is not None:
                    streamlit.success(
                        "Done selecting? Continue selecting fields."
                        )
                file_name_input = streamlit.text_input("Enter a file name (without extension):", "")
                
                if streamlit.button(f"Store fields {file_name_input}"):
                    if not file_name_input.strip():
                        streamlit.warning("Please enter a valid file name.")
                    else:    
                        save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                        os.makedirs(save_path, exist_ok=True)
                        full_path = os.path.join(save_path, f"{file_name_input}.csv")
                        selected_records.to_csv(full_path, index=False)
                        streamlit.success(f"✅ {file_name_input} successfully saved! You can find file at manage-files.")
        except Exception as e:
                streamlit.error(f"Failed to load file: {e}")

elif selected_action_option == "Transform field values":
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = ["Select file to transform"] + saved_files
    selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices, key="selectbox_trans")
    if selected_file == "Select file to transform":
        streamlit.session_state["disable"] = True
        streamlit.info("Please select file to continue.")
    else:
        try:
            file_path = os.path.join(save_path, selected_file)
            if selected_file.endswith(".csv"):
                records = pandas.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                records = pandas.read_excel(file_path)
            if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_transform__object_before"):
                streamlit.write(records.head())   
            columns = records.columns.tolist()
            select_columns_trans = streamlit.multiselect(
                "Choose fileds you want to transform", columns
                )
            if select_columns_trans:
                column_to_map = streamlit.selectbox("Choose a column to transform values in", select_columns_trans, key="selectbox_map_column")
                if column_to_map:
                    streamlit.info("Define value mappings (e.g. Male → 1, Female → 0)")
                    if "value_map" not in streamlit.session_state:
                        streamlit.session_state["value_map"] = {}
                    with streamlit.form(key="mapping_form"):
                        new_key = streamlit.text_input("Enter original value (e.g. Male)", key="map_key")
                        new_value = streamlit.text_input("Enter new value (e.g. 1)", key="map_value")
                        submitted = streamlit.form_submit_button("➕ Add mapping")
                        
                        if submitted:
                            if new_key.strip() and new_value.strip():
                                streamlit.session_state["value_map"][new_key] = new_value
                                streamlit.success(f"Added: {new_key} → {new_value}")
                            else:
                                streamlit.warning("Please fill the fields!") 

                    if streamlit.session_state["value_map"]:
                        streamlit.write("Current mappings:") 
                        streamlit.json(streamlit.session_state["value_map"])
                        if streamlit.button(f"Apply mapping to {column_to_map}"):
                            records[f"{column_to_map}"] = records[f"{column_to_map}"].replace(streamlit.session_state["value_map"])
                            save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                            os.makedirs(save_path, exist_ok=True)
                            full_path = os.path.join(save_path, selected_file)
                            records.to_csv(full_path, index=False)
                            streamlit.success(f"✅ Field updated successfully!")
                        if streamlit.checkbox(f"Preview updated {selected_file}", key=f"preview_{selected_file}_transform_object_after"):
                            streamlit.write(records.head())
        except Exception as e:
            streamlit.error(f"Field transformation failed: {e}") 

elif selected_action_option == "Update field data types":
    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
    os.makedirs(save_path, exist_ok=True)
    saved_files = [
        files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
        ]
    file_choices = ["Select file to update data types"] + saved_files
    selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices, key="selectbox_trans")
    if selected_file == "Select file to update data types":
        streamlit.session_state["disable"] = True
        streamlit.info("Please select file to continue.")
    else:
        try:
            file_path = os.path.join(save_path, selected_file)
            if selected_file.endswith(".csv"):
                records = pandas.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                records = pandas.read_excel(file_path)
            if streamlit.checkbox(f"📄 Preview {selected_file}", key=f"preview_{selected_file}_dtypes_object"):
                streamlit.write(records.head())
            if streamlit.checkbox(f"Preview {selected_file} data types", key=f"preview_{selected_file}_dtypes_before"):
                streamlit.write(records.dtypes.reset_index().rename(columns={"index": "Fields", 0: "Data Type"})) 

            streamlit.info("Data types look good? Skip this step. Otherwise, update them below!")
            columns = records.columns.tolist()
            select_columns_dtypes = streamlit.multiselect(
                "Choose fileds you want to change data types", columns
                )
            if select_columns_dtypes:
                column_to_update = streamlit.selectbox("Choose a column to change data type in", select_columns_dtypes, key="selectbox_dtype_column")
                if column_to_update:
                    data_type_options = ["select data type", "int", "float", "str", "bool", "Datetime", "Time", "Date"]
                    selected_data_type = streamlit.selectbox("Choose data type", data_type_options, key="data_type_options_key")
                    try:
                        if selected_data_type == "select data type":
                            streamlit.session_state["disable"] = True
                            streamlit.info("Select data type to continue")
                        elif selected_data_type == "int":
                                if streamlit.button(f"Apply data type to {column_to_update}"):
                                     records[f"{column_to_update}"] = pandas.to_numeric(records[f"{column_to_update}"], errors="raise").astype("int")
                                     save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                     os.makedirs(save_path, exist_ok=True)
                                     full_path = os.path.join(save_path, selected_file)
                                     records.to_csv(full_path, index=False)         
                                     streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "float":
                                precision = streamlit.slider("Select decimal precision", min_value=0, max_value=10, value=2)
                                streamlit.write(f"Precision selected: {precision}")
                                if precision:
                                    if streamlit.button(f"Apply data type to {column_to_update}"):
                                        records[f"{column_to_update}"] = pandas.to_numeric(records[f"{column_to_update}"], errors="raise").astype("float").round(precision)
                                        save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                        os.makedirs(save_path, exist_ok=True)
                                        full_path = os.path.join(save_path, selected_file)
                                        records.to_csv(full_path, index=False)         
                                        streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "str":
                                if streamlit.button(f"Apply data type to {column_to_update}"):
                                     records[f"{column_to_update}"] = pandas.to_numeric(records[f"{column_to_update}"], errors="raise").astype("str")
                                     save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                     os.makedirs(save_path, exist_ok=True)
                                     full_path = os.path.join(save_path, selected_file)
                                     records.to_csv(full_path, index=False)         
                                     streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "bool":
                                if streamlit.button(f"Apply data type to {column_to_update}"):
                                     records[f"{column_to_update}"] = pandas.to_numeric(records[f"{column_to_update}"], errors="raise").astype("bool")
                                     save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                     os.makedirs(save_path, exist_ok=True)
                                     full_path = os.path.join(save_path, selected_file)
                                     records.to_csv(full_path, index=False)         
                                     streamlit.success(f"✅ {column_to_update} updated successfully!") 
                    # '2024-12-31 14:45'
                        elif selected_data_type == "Datetime":
                            if streamlit.button(f"Apply data type to {column_to_update}"):
                                records[f"{column_to_update}"] = pandas.to_datetime(records[f"{column_to_update}"], errors="raise")
                                save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                os.makedirs(save_path, exist_ok=True)
                                full_path = os.path.join(save_path, selected_file)
                                records.to_csv(full_path, index=False)         
                                streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "Time":
                            if streamlit.button(f"Apply data type to {column_to_update}"):
                                time_input = streamlit.text_input("Enter time in HH:MM:SS format", "")
                                if not time_input.strip():
                                    streamlit.warning("Please enter time")
                                else:
                                    records[f"{column_to_update}"] = pandas.to_datetime([time_input], format='%H:%M:%S').time[0]
                                    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                    os.makedirs(save_path, exist_ok=True)
                                    full_path = os.path.join(save_path, selected_file)
                                    records.to_csv(full_path, index=False)         
                                    streamlit.success(f"✅ {column_to_update} updated successfully!") 
                        elif selected_data_type == "Date":
                                if streamlit.button(f"Apply data type to {column_to_update}"):
                                    records[f"{column_to_update}"] = pandas.to_datetime(records[f"{column_to_update}"], errors="raise")
                                    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                                    os.makedirs(save_path, exist_ok=True)
                                    full_path = os.path.join(save_path, selected_file)
                                    records.to_csv(full_path, index=False)         
                                    streamlit.success(f"✅ {column_to_update} updated successfully!") 
                    except Exception as e:
                            streamlit.error(f"⚠️ Conversion failed: {e}")
                if streamlit.checkbox(f"Preview {selected_file} data types", key=f"preview_{selected_file}_dtypes_after"):
                    streamlit.write(records.dtypes.reset_index().rename(columns={"index": "Fields", 0: "Data Type"}))                        

        except Exception as e:
            streamlit.error(f"Field transformation failed: {e}")  
# To add data value range an example 0 - 10 group it as "0 to 10"                   
