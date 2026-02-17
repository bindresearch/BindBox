import streamlit as st

# This is the homepage of the Bind NMR tools and applications page.

st.set_page_config(layout="wide")



def make_tools_list():

    st.subheader("Current implemented tools (Bind):")
    st.text("SpinExplorer - A graphical user interface for NMR data processing and analysis (link to download)")
    st.text("BMRB Chemical Shifts - An interactive dashboard of Biomolecular Magnetic Resonance Database (BMRB) chemical shift distributions for single atoms/groups of atoms for the whole BMRB, disordered residues only, and structured residues only. Filters based on sample conditions and experimental conditions can be applied. Data points plotted in histograms of chemical shifts can be downloaded.")
    st.text("SpinForecast - A chemical shift assignment tool for disordered residues based on a Bayesian implementation of SpinForecast.")

    st.subheader("Current implemented tools (external):")
    st.text("POTENCI - random coil chemical shift predictor (Frans Mulder lab)")
    st.text("Temperature/pH adjustment predictor (Frans Mulder lab)")


make_tools_list()
