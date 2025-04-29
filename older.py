import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For more advanced chart types
import os
# --- Load Data (Your Existing Code) ---
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
    )
    if selected_file != "Select file to view fields":
        try:
            file_path = os.path.join(save_path, selected_file)
            if selected_file.endswith(".csv"):
                records = pd.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                records = pd.read_excel(file_path)
            st.session_state['records'] = records
            if st.checkbox(f"📄 Show Fields in {selected_file}", key=f"preview_{selected_file}_dashboard_object"):
                st.subheader("Available Fields:")
                for column in records.columns:
                    st.write(column)
        except Exception as e:
            st.error(f"Failed to load file: {e}")

# --- Charting Section ---
if 'records' in st.session_state:
    records = st.session_state['records']
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
            if chart_type == "Scatter Plot":
                if color_option:
                    sns.scatterplot(x=x_axis, y=y_axis, hue=color_option, data=records, ax=ax)
                else:
                    ax.scatter(records[x_axis], records[y_axis])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
            elif chart_type == "Bar Chart":
                # For bar charts, usually one axis is categorical
                if records[x_axis].dtype == 'object' or records[y_axis].dtype == 'object':
                    if records[x_axis].dtype == 'object':
                        if color_option:
                            sns.barplot(x=x_axis, y=y_axis, hue=color_option, data=records, ax=ax)
                        else:
                            sns.barplot(x=x_axis, y=y_axis, data=records, ax=ax)
                        ax.set_xlabel(x_axis)
                        ax.set_ylabel(y_axis)
                        plt.xticks(rotation=45, ha='right')
                    elif records[y_axis].dtype == 'object':
                        if color_option:
                            sns.barplot(y=y_axis, x=x_axis, hue=color_option, data=records, ax=ax)
                        else:
                            sns.barplot(y=y_axis, x=x_axis, data=records, ax=ax)
                        ax.set_ylabel(y_axis)
                        ax.set_xlabel(x_axis)
                        plt.yticks(rotation=45, ha='right')
                    else:
                        st.warning("For Bar Chart, at least one selected axis should be categorical (non-numeric).")
                else:
                    st.warning("For Bar Chart, at least one selected axis should be categorical (non-numeric).")
            elif chart_type == "Line Chart":
                # Line charts often work best with a time-based x-axis
                if color_option:
                    sns.lineplot(x=x_axis, y=y_axis, hue=color_option, data=records, ax=ax)
                else:
                    ax.plot(records[x_axis], records[y_axis])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                plt.xticks(rotation=45, ha='right')

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error generating chart: {e}")

# --- Sidebar Content ---
with col1_dash:
    st.title("Data Exploration Dashboard")
    st.markdown("Explore your uploaded data with interactive charts.")

with col2_dash:
    st.subheader("Filters (Coming Soon)")
    st.info("You'll be able to add filters here to subset your data.")

with col3_dash:
    st.subheader("Insights (Coming Soon)")
    st.info("Key insights and summaries will appear here.")

with col4_dash:
    st.subheader("More Controls (Coming Soon)")
    st.info("Additional data manipulation options will be added.")
    
    # elif chart_type == "Bar Chart":
    # seaborn.barplot(x=x_axis, y=y_axis, hue=color_option, data=records, ax=ax) if color_option else seaborn.barplot(x=x_axis, y=y_axis, data=records, ax=ax)

    