import pandas as pd
import streamlit as st

st.title("Automated plant disease detection using deep learning and cloud deployment")

ipynb_url = "https://raw.githubusercontent.com/tayabba-19/Automated-Plant-Disease-Detection-Using-Deep-Learning-and-Cloud-Deployment/refs/heads/main/Final_Project.ipynb"
df = pd.read_ipynb(ipynb_url)

st.success("Dataset loaded successfully!")
st.dataframe(df.head())
        

