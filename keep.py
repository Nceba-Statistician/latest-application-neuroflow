import streamlit as st

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Account", "Model Flow", "Reports", "Tools"])

if menu == "Account":
    st.sidebar.button("Log In")
    st.sidebar.button("Log Out")
    st.sidebar.button("Users")

elif menu == "Model Flow":
    st.sidebar.button("Data Config")
    st.sidebar.button("Neuro Flow")
    st.sidebar.button("Model History")

elif menu == "Reports":
    st.sidebar.button("Dashboard")
    st.sidebar.button("Bug Reports")
    st.sidebar.button("System Alerts")

elif menu == "Tools":
    st.sidebar.button("Data Migration")
    st.sidebar.button("Data Cleaning")
    st.sidebar.button("Search")

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="footer" style="position: fixed; bottom: 10px; width: 20%; font-size: 12px; text-align: center;">
        © 2025 Your Company
    </div>
    """,
    unsafe_allow_html=True
)

