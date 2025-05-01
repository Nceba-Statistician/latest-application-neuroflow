import streamlit, pandas, numpy, datetime, json, requests, pyodbc, os, seaborn
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

save_path = os.path.join("ModelFlow", "data-config", "saved-files")
os.makedirs(save_path, exist_ok=True)
saved_files = [
    files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
    ]
file_choices = ["Select file to view fields"] + saved_files

col1_dash, col2_dash, col3_dash, col4_dash, col5_dash = streamlit.columns(5)
with col5_dash:
    selected_file = streamlit.selectbox("📂 Choose from your saved files:", file_choices, key="selectbox_dashboard", label_visibility="collapsed")
    if selected_file == "Select file to view fields":
        streamlit.session_state["disable"] = True
    else:
        try:
            file_path = os.path.join(save_path, selected_file)
            if selected_file.endswith(".csv"):
                records = pandas.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                records = pandas.read_excel(file_path)
            if streamlit.checkbox(f"📄 Fields {selected_file}", key=f"preview_{selected_file}_dashboard_object"):
                for column in records.columns:
                    streamlit.write(column)
        except Exception as e:
            streamlit.error(f"Failed to load file: {e}") 
                       
with col1_dash:
    if "records" in streamlit.session_state:
        streamlit.metric("Total Records", len(streamlit.session_state["records"]))
   
        streamlit.session_state["records"] = records
        numeric_columns = records.select_dtypes(include=numpy.number).columns.tolist()
        all_columns = records.columns.tolist()
        
        chart_type = streamlit.selectbox("Select Chart Type:", ["Scatter Plot", "Bar Chart", "Line Chart"])
        x_axis = streamlit.selectbox("Select X-axis:", all_columns)
        y_axis = streamlit.selectbox("Select Y-axis:", all_columns)
        color_option = streamlit.selectbox("Optional Color Field:", [None] + all_columns)
        if x_axis and y_axis:
            try:
                fig, ax = pyplot.subplots(figsize=(6, 4))
                if chart_type == "Scatter Plot":
                    if color_option:
                        seaborn.scatterplot(x=x_axis, y=y_axis, hue=color_option, data=records, ax=ax)
                    else:
                        ax.scatter(records[x_axis], records[y_axis])
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    
                streamlit.pyplot(fig)
            except Exception as e:
                streamlit.error(f"Error generating chart: {e}")    
        else:
            streamlit.write("There is an error on axis")
    else:
        streamlit.info("Please select a data file to begin.")             
                 
with col2_dash:
    "Second"       
with col3_dash:
    "Third"                    
with col4_dash:
    "Fourth"            
        