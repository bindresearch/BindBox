import streamlit as st

# This page creates a front end tool where users can download SpinExplorer and read its documentation


st.set_page_config(layout="wide")


class SpinExplorer_page():
    """
    This class will produce a page for the SpinExplorer page
    """
    def __init__(self):

        self.create_page()
        

    def create_page(self):
        """
        Create the page for the download files and documentation
        """
        st.title("SpinExplorer: A graphical interface for NMR processing and analysis")

        st.markdown('SpinExplorer is a Python-based graphical interface that allows raw Bruker/Varian NMR data to be converted to processed NMR spectra. Processing can be performed using nmrglue or nmrPipe (if installed).')

        st.markdown('Pre-built SpinExplorer application can be downloaded from the following page: https://github.com/bindresearch/SpinExplorer/releases')
        st.markdown('Alternatively, the GitHub repo can be cloned and installed manually using pip (https://github.com/bindresearch/SpinExplorer)')
        st.markdown('Documentation is available on the GitHub page and video tutorials will be made available at https://www.youtube.com/@BindResearch')


        





# Create the page
page = SpinExplorer_page()