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
    streamlit.info("Please select a file to view its contents.")
else:
    file_path = os.path.join(save_path, selected_file)
    try:
        if selected_file.endswith(".csv"):
            records = pandas.read_csv(file_path)
        elif selected_file.endswith(".xlsx"):
            records = pandas.read_excel(file_path)
        streamlit.success(f"📄 Preview of {selected_file}")
        streamlit.write(records.head())
    except Exception as e:
        streamlit.error(f"Failed to load file: {e}")