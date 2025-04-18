import streamlit, os, pandas

save_path = os.path.join("ModelFlow", "data-config", "saved-files")
os.makedirs(save_path, exist_ok=True)
streamlit.subheader("")
saved_files = [
    files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
]
file_choices = ["Select a saved file"] + saved_files
selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices)

if selected_file == "Select a saved file":
    streamlit.session_state["disable"] = True
    if file_choices is None:
        streamlit.info("You haven’t saved any files yet.")
    else:
        streamlit.info("Please select a file to view its contents.")
else:
    file_path = os.path.join(save_path, selected_file)
    try:
        if selected_file.endswith(".csv"):
            records = pandas.read_csv(file_path)
            if streamlit.checkbox(f"📄 Preview {selected_file}"):
                streamlit.success(f"{selected_file}")
                streamlit.write(records.head())
            if streamlit.button(f"🗑️ Delete {selected_file}"):
                os.remove(file_path)
                streamlit.success(f"{selected_file} deleted successfully!") 

            records_columns = records.columns.tolist()
            select_columns_dist = streamlit.multiselect(
                "Choose fields to delete", records_columns
                )
            if select_columns_dist:
                column_to_dist = streamlit.selectbox("Choose a column to continue", select_columns_dist, key="selectbox_delete_column")
                if column_to_dist:
                    streamlit.write("")
                    if streamlit.button(f"Delete {column_to_dist}"):
                        column_deleted = records.drop(column_to_dist, axis=1)
                        save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                        os.makedirs(save_path, exist_ok=True)
                        full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".csv")
                        column_deleted.to_csv(full_path, index=False)
                        streamlit.success(f"✅ You have successfully deleted {column_to_dist}!")

        elif selected_file.endswith(".xlsx"):
            records = pandas.read_excel(file_path)
            if streamlit.checkbox(f"📄 Preview {selected_file}"):
                streamlit.success(f"{selected_file}")
                streamlit.write(records.head())
            if streamlit.button(f"🗑️ Delete {selected_file}"):
                os.remove(file_path)
                streamlit.success(f"{selected_file} deleted successfully!")  
           
            records_columns = records.columns.tolist()
            select_columns_dist = streamlit.multiselect(
                "Choose fields to delete", records_columns
                )
            if select_columns_dist:
                column_to_dist = streamlit.selectbox("Choose a column to continue", select_columns_dist, key="selectbox_delete_column")
                if column_to_dist:
                    streamlit.write("")
                    if streamlit.button(f"Delete {column_to_dist}"):
                        column_deleted = records.drop(column_to_dist, axis=1)
                        save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                        os.makedirs(save_path, exist_ok=True)
                        full_path = os.path.join(save_path, os.path.splitext(selected_file)[0] + ".xlsx")
                        column_deleted.to_excel(full_path, index=False, engine='openpyxl')
                        streamlit.success(f"✅ You have successfully deleted {column_to_dist}!")

    except Exception as e:
        streamlit.error(f"Failed to load file: {e}")
