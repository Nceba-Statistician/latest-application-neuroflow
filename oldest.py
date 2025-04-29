import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Load Data Function ---
def load_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    return None

# --- Initialize Session State for Loaded Data ---
if 'loaded_data' not in st.session_state:
    st.session_state['loaded_data'] = None

# --- File Selection and Data Loading ---
save_path = os.path.join("ModelFlow", "data-config", "saved-files")
os.makedirs(save_path, exist_ok=True)
saved_files = [
    files for files in os.listdir(save_path) if files.endswith(".csv") or files.endswith(".xlsx")
]
file_choices = ["Select file to view fields"] + saved_files

col1_dash, col2_dash, col3_dash, col4_dash, col5_dash = st.columns(5)

with col5_dash:
    selected_file = st.selectbox(
        "📂 Choose data file:",
        file_choices,
        key="selectbox_dashboard",
        label_visibility="collapsed",
        index=0,  # Set default to "Select file..."
    )
    if selected_file != "Select file to view fields" and st.session_state['loaded_data'] is None:
        try:
            file_path = os.path.join(save_path, selected_file)
            data = load_data(file_path)
            if data is not None:
                st.session_state['loaded_data'] = data
                st.session_state['selected_file_name'] = selected_file  # Store the filename
                st.success(f"Data from '{selected_file}' loaded successfully!")
            else:
                st.error("Failed to load data.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    elif st.session_state['loaded_data'] is not None and selected_file != st.session_state['selected_file_name']:
        # If a new file is selected after data is already loaded, clear the loaded data
        st.session_state['loaded_data'] = None
        try:
            file_path = os.path.join(save_path, selected_file)
            data = load_data(file_path)
            if data is not None:
                st.session_state['loaded_data'] = data
                st.session_state['selected_file_name'] = selected_file
                st.success(f"Data from '{selected_file}' loaded successfully!")
            else:
                st.error("Failed to load data.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# --- Display Available Fields (Conditional on Data Loading) ---
if st.session_state['loaded_data'] is not None:
    with col5_dash:
        if st.checkbox(f"📄 Show Fields in {st.session_state['selected_file_name']}", key=f"preview_{st.session_state['selected_file_name']}_dashboard_object"):
            st.subheader("Available Fields:")
            for column in st.session_state['loaded_data'].columns:
                st.write(column)

# --- Dashboard Content (Conditional on Data Loading) ---
with col1_dash:
    st.title("Data Exploration Dashboard")
    if st.session_state['loaded_data'] is not None:
        st.metric("Total Records", len(st.session_state['loaded_data']))
    else:
        st.info("Please select a data file to begin.")

with col2_dash:
    st.subheader("Filters (Coming Soon)")
    if st.session_state['loaded_data'] is not None:
        st.info("You'll be able to add filters here to subset your data.")
        # Add actual filter widgets here when you implement them
    else:
        st.info("Select a file to enable filters.")

with col3_dash:
    st.subheader("Interactive Charts")
    if st.session_state['loaded_data'] is not None:
        records = st.session_state['loaded_data']
        numeric_columns = records.select_dtypes(include=np.number).columns.tolist()
        all_columns = records.columns.tolist()

        st.sidebar.header("Chart Configuration")
        chart_type = st.sidebar.selectbox("Select Chart Type:", ["Scatter Plot", "Bar Chart", "Line Chart"])
        x_axis = st.sidebar.selectbox("Select X-axis:", all_columns)
        y_axis = st.sidebar.selectbox("Select Y-axis:", all_columns)
        color_option = st.sidebar.selectbox("Optional Color Field:", [None] + all_columns)

        st.subheader(f"{chart_type} Visualization")

        if x_axis and y_axis:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                # ... (rest of your charting code as before, using st.session_state['loaded_data'])
                if chart_type == "Scatter Plot":
                    if color_option:
                        sns.scatterplot(x=x_axis, y=y_axis, hue=color_option, data=records, ax=ax)
                    else:
                        ax.scatter(records[x_axis], records[y_axis])
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                elif chart_type == "Bar Chart":
                    # ... (bar chart logic)
                    pass
                elif chart_type == "Line Chart":
                    # ... (line chart logic)
                    pass
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error generating chart: {e}")
    else:
        st.info("Select a file to configure charts.")

with col4_dash:
    st.subheader("More Controls (Coming Soon)")
    if st.session_state['loaded_data'] is not None:
        st.info("Additional data manipulation options will be added.")
        # Add more controls here later
    else:
        st.info("Select a file to enable more controls.")
        