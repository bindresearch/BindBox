import streamlit as st
import polars as pl
import plotly.graph_objects as go
from tools.ChemicalShiftTools.Potenci.potenci import potenci_backend

# This page creates a front end tool where users can run POTENCI random coil chemical shift predictions for a sequence of their choice.
# The user can then download a csv of the results
# The page also produces plots of the predicted two-dimensional NMR spectra in each case (e.g. H-N, HA-CA, CA-N, C-N, CA-C etc)
# The backend code is in Pages/Potenci


st.set_page_config(layout="wide")

residue_1_letter_codes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

class potenci_page():
    """
    This class will produce a page to perform potenci random coil
    chemical shift prediction.
    """
    def __init__(self):

        self.create_page()
        

    def create_page(self):
        """
        Create the page for the user to insert the amino acid sequence and conditions to
        perform the POTENCI random coil chemical shift predictor.
        """
        st.title("POTENCI: disordered protein chemical shift predictor")
        st.text("POTENCI: prediction of temperature, neighbor and pH-corrected chemical shifts for intrinsically disordered proteins. This open-source tool was developed by the Frans Mulder lab. If you use this tool, please cite the papers linked below. The original POTENCI code used for this tool is available on the Frans Mulder lab github page (protein-nmr) shown below.")
        st.markdown("**References**")
        st.markdown("POTENCI code: https://github.com/protein-nmr/POTENCI")
        st.markdown("POTENCI (Nielsen and Mulder, 2018): https://doi.org/10.1007/s10858-018-0166-5")
        st.markdown("Temperature corrections (Kjaergaard et al. 2011): https://doi.org/10.1007/s10858-011-9508-2, https://doi.org/10.1007/s10858-011-9472-x")
        st.markdown("pH corrections (Platzer et al. 2014): https://doi.org/10.1007/s10858-014-9862-y")

        # Add logo to the sidebar
        st.logo('./bind-logo-alpha.svg', size="large", link="https://bindresearch.org", icon_image='./bind-logo-alpha.svg')
        
        st.subheader('Perform POTENCI prediction')
        
        self.aa_sequence = st.text_area(label='Sequence of amino acids (1 letter codes)', value='', help='At least 5 residues must be provided. Note, the terminal 2 residues at the N and C terminus are excluded from the results.')
        
        col1, col2, col3 = st.columns(3)
        self.pH_value = col1.text_input(label="pH", value='7.0')
        self.temp_value = col2.text_input(label="Temperature (K)", value='298.0')
        self.ionic_strength = col3.text_input(label="Ionic Strength (M)", value='0.1')
        st.button('Perform POTENCI prediction', on_click=self.run_potenci)


        if "df" not in st.session_state:
            st.session_state.df = pl.DataFrame(schema={"residue number":str, "amino acid": str, "N (ppm)": float, "C (ppm)": float, "CA (ppm)": float, "CB (ppm)": float, "H (ppm)":float, "HA (ppm)": float, "HB (ppm)": float})
 
        self.placeholder = st.empty()
        self.placeholder.dataframe(st.session_state.df)

        # Create a download button to download the predicted chemical shifts
        st.download_button(
            label="Download predicted chemical shifts",
            data=st.session_state.df.write_csv().encode("utf-8"),
            file_name="potenci_prediction.csv",
            mime="text/csv",
            icon=":material/download:",
        )

        if(st.session_state.df.height > 1):
            st.subheader("Predicted spectra")
            self.combination = st.selectbox("Select atom combination:", ['H-N', 'HA-CA', 'HB-CB', 'CA-N', 'C-N (1-bond)', 'C-N (2-bond)', 'CA-CB','HA-C'])

            self.plot_predicted_spectra()

    def run_potenci(self):
        """
        This code runs POTENCI on the 1 letter amino acid sequence provided. 
        """

        if(self.perform_checks()==True):
            potenci = potenci_backend()
            potenci_result = potenci.perform_potenci_calc(self.aa_sequence, [self.pH_value, self.temp_value, self.ionic_strength], pH_temperature_predict_page=True)
            self.format_results(potenci_result)
        else:
            return
        
        
    def format_results(self, shift_dict: dict):
        """
        Update the table with POTENCI predicted random coil chemical shifts
        """

        dataframe = pl.DataFrame(schema={"residue number":str, "amino acid": str, "N": float, "C": float, "CA": float, "CB": float, "H":float, "HA": float, "HB": float})
        for residue, residue_dict in shift_dict.items():
            if(residue_dict == {}):
                continue
            residue_list = []
            residue_list.append(residue[0])
            residue_list.append(residue[1])
            order = ['N', 'C', 'CA', 'CB', 'H', 'HA', 'HB']
            for atom in order:
                try:
                    residue_list.append(residue_dict[atom])
                except:
                    residue_list.append(None)

            from copy import deepcopy
            df = pl.DataFrame([residue_list], schema={"residue number":str, "amino acid": str, "N": float, "C": float, "CA": float, "CB": float, "H":float, "HA": float, "HB": float}, orient='row')
            dataframe = pl.concat([deepcopy(dataframe), df], how='vertical')

        st.session_state.df = dataframe



    def plot_predicted_spectra(self):

        dataframe = st.session_state.df
        if(self.combination == 'C-N (1-bond)' or self.combination == 'C-N (2-bond)'):
            atom1 = 'C'
            atom2 = 'N'
        else:
            atom1, atom2 = self.combination.split('-')
        dataframe = dataframe.select(['residue number', 'amino acid', atom1, atom2])
        if(self.combination == 'C-N (1-bond)'):
            # Need to shift the C column + 1 (as the i-1 C atom is linked to the i N atom)
            dataframe = dataframe.with_columns(pl.col('C').shift(1).alias('C'))

        dataframe.drop_nulls()
        x = dataframe[atom1]
        y = dataframe[atom2]
        residues = dataframe['residue number']
        aa = dataframe['amino acid']
        labels = []
        for i, residue in enumerate(residues):
            labels.append(residue+aa[i])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x.to_numpy(), y=y.to_numpy(), hovertext=labels, mode='markers'))

        xlabel = atom1 + ' Chemical Shift (ppm)'
        ylabel = atom2 + ' Chemical Shift (ppm)'

        fig.update_layout(xaxis = dict(title=xlabel, autorange='reversed'))
        fig.update_layout(yaxis = dict(title=ylabel, autorange='reversed'))
            
        st.plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"format": "svg","height": 600,"width": 800,"scale": 1}})

            
            

    def perform_checks(self):
        """
        Perform the following checks before running POTENCI
        - The amino acid code provided is checked to be of the correct format.
        - There must be at least 5-residues.
        - Check that pH, temperature and ionic strength values are valid numbers
        """

        self.aa_sequence.replace('\n', '')
        self.aa_sequence.replace(' ', '')
        
        # Check to see that all supplied values are correct
        for character in self.aa_sequence:
            if(character not in residue_1_letter_codes):
                st.error(body = f'Character {character} is not a standard 1-letter amino acid code. Please change this and try again.',icon="🚨")
                return False
        
        if(len(self.aa_sequence)<5):
            st.error(body = f'The amino acid sequence must be at least 5 residues long. The current length is {len(self.aa_sequence)}. Please try again.',icon="🚨")
            return False
        
        # Check that the user-supplied conditions are valid
        try:
            float(self.pH_value)
        except:
            st.error(body = f'The pH value {self.pH_value} could not be converted to a float. Please try again.',icon="🚨")
            return False
        try:
            float(self.temp_value)
        except:
            st.error(body = f'The temperature value {self.temp_value} could not be converted to a float. Please try again.',icon="🚨")
            return False
        try:
            float(self.ionic_strength)
        except:
            st.error(body = f'The ionic strength value {self.ionic_strength} could not be converted to a float. Please try again.',icon="🚨")
            return False
        
        return True
        





# Create the page
page = potenci_page()