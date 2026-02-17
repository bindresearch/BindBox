import streamlit as st

# This is the homepage of the Bind NMR tools and applications page.

st.set_page_config(layout="wide")


home_page = st.Page("tools/Home/home.py", title="About")
available_tools = st.Page("tools/Home/available_tools.py", title="Available tools")


bmrb_page = st.Page("tools/ChemicalShiftTools/BMRB_Chemical_Shifts.py", title="BMRB Chemical Shifts")
POTENCI_page = st.Page("tools/ChemicalShiftTools/POTENCI.py", title="POTENCI (Frans Mulder lab)")
Temp_pH_correction_page = st.Page("tools/ChemicalShiftTools/Temperature_pH_Adjust.py", title="Temperature/pH adjustment predictor (Frans Mulder lab)")
SpinForecast_page = st.Page("tools/ChemicalShiftTools/SpinForecast.py", title="SpinForecast (Bind Research)")
SpinExplorer_page = st.Page("tools/GeneralNMRTools/SpinExplorerBindBox.py", title="SpinExplorer (Bind Research/University of Oxford)")

pg = st.navigation(
        {
            "Home": [home_page, available_tools],
            "General NMR tools": [SpinExplorer_page],
            "Chemical shift tools": [bmrb_page, SpinForecast_page, POTENCI_page, Temp_pH_correction_page],
        }
    )

pg.run()


