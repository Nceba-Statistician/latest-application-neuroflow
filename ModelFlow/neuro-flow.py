import streamlit, tensorflow, os, pandas, numpy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
import datetime

streamlit.markdown("""<style> .title { position: absolute; font-size: 20px; right: 10px; top: 10px} </style> """,
    unsafe_allow_html=True
)
streamlit.markdown(""" <style> .footer { position: fixed; bottom: 0; left: 0; width: 20%; background-color: rgba(0, 0, 0, 0.05); padding: 10px; font-size: 12px; text-align: center;} </style> """,
    unsafe_allow_html=True
)
streamlit.markdown(""" <style> .emotion-cache {vertical-align: middle; overflow: hidden; color: inherit; fill: currentcolor; display: inline-flex; -webkit-box-align: center; align-items: center; font-size: 1.25rem; width: 1.25rem; height: 1.25rem; flex-shrink: 0; } </style>""",
    unsafe_allow_html=True
)
streamlit.markdown("""<div class="title">Neural Network Model Builder for Prediction</div>""", unsafe_allow_html=True)

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
        if streamlit.checkbox(f"📄 Preview object"):
            streamlit.write(records.head())

        columns = records.columns.tolist()
        select_columns = streamlit.multiselect(
            "Select columns for your model", columns
        )
        if select_columns:
            new_object = records[select_columns]
            if new_object is not None:
                streamlit.success("Great! Once you have selected all your fields store them to the new object")
                if streamlit.button("Store records"):
                    save_path = os.path.join("ModelFlow", "data-config", "saved-files")
                    os.makedirs(save_path, exist_ok=True)
                    new_file_name = f"updated_fields_{new_object}.csv"
                    full_path = os.path.join(save_path, new_file_name)
                    new_object.to_csv(full_path, index=False)
                    streamlit.success("Records stored! Check manage-files")
                    if streamlit.checkbox("📄 Preview updated file"):
                        streamlit.success(f"{new_file_name}")
                        streamlit.write(new_object.head())
        
    except Exception as e:
        streamlit.error(f"Failed to load file: {e}")

        