import streamlit

streamlit.write("Nceba Gagaza")
streamlit.important("Profile coming soon!")
streamlit.write("Education history, work experience, projects, etc")
if streamlit.download("Download CV"):
  streamlit.write("To be added.")
