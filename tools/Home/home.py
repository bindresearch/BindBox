import streamlit as st

# This is the homepage of the Bind NMR tools and applications page.

st.set_page_config(layout="wide")



def make_homepage():
    # Add logo to the sidebar
    st.logo('./bind-logo-alpha.svg', size="large", link="https://bindresearch.org", icon_image='./bind-logo-alpha.svg')

    st.title("BindBox: Bind Research NMR Tools and Applications")

   

    st.text("This page contains tools used for NMR spectroscopy studies at Bind Research. Where tools are developed by Bind Research, we ask that you please cite our work (where relevant). For open-source tools developed by external researchers but included within these tools, please cite the relevant publications shown for each tool.")
    st.markdown("Our GitHub page contains repositories for the tools developed at Bind Research (https://github.com/orgs/bindresearch)")
    st.markdown("For more information on Bind Research, please visit https://bindresearch.org")

    st.markdown('')
    st.markdown('')

    col1, col2,col3,col4, col5 = st.columns([1,1,0.5,1,1])

    with col3:

        st.image('./BindBoxLogo.svg', use_container_width=True)
    


make_homepage()
