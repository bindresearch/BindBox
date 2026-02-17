import streamlit as st
import polars as pl
import plotly.graph_objects as go
import polars as pl
import re
from copy import deepcopy
from pathlib import Path
from io import StringIO
from tools.ChemicalShiftTools.Potenci.potenci import potenci_backend

# This page creates a front end tool where users can predict temperature and pH changes from one condition to another for a given sequence
# The user can then download a csv of the results
# The backend code is in Pages/Potenci (the temperature and pH corrections are from POTENCI)


st.set_page_config(layout="wide")

residue_1_letter_codes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
atoms = ['H', 'HN', 'N', 'CA', 'CB', 'CO', 'C']

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
        calculate the temperature and pH adjustments for each atom of each residue
        """
        st.title("Temperature and pH chemical shift adjustments")
        st.text("A tool to predict the changes to chemical shifts when the pH and/or temperature is adjusted from one condition to another. Adjustments are not made for terminal residues of the sequence. The predictions assume that the protein exists as a random coil, and that it remains a random coil upon the change in conditions. The temperature and pH adjustments are calculated using POTENCI, an open-source tool developed by the Frans Mulder lab. If you use this tool, please cite the papers linked below. The original POTENCI code used for this tool is available on the Frans Mulder lab github page (protein-nmr) shown below.")
        st.markdown("**References**")
        st.markdown("POTENCI code: https://github.com/protein-nmr/POTENCI")
        st.markdown("POTENCI (Nielsen and Mulder, 2018): https://doi.org/10.1007/s10858-018-0166-5")
        st.markdown("Temperature corrections (Kjaergaard et al. 2011): https://doi.org/10.1007/s10858-011-9508-2, https://doi.org/10.1007/s10858-011-9472-x")
        st.markdown("pH corrections (Platzer et al. 2014): https://doi.org/10.1007/s10858-014-9862-y")
        # Add logo to the sidebar
        st.logo('./bind-logo-alpha.svg', size="large", link="https://bindresearch.org", icon_image='./bind-logo-alpha.svg')
        
        st.subheader('Insert sequence/input peaklist:')
        
        self.aa_sequence = st.text_area(label='Sequence of amino acids (1 letter codes)', value='', help='At least 5 residues must be provided. Note, the terminal 2 residues at the N and C terminus are excluded from the results.')
        

        help_message = '''The peaklist should be in tabular format (.tab) or csv format. The residue column must contain the residue number to link it to the protein sequence.\n
        e.g., (for .tab):\n
        residue\tH\tN\tCA\n
        \tA2\t8.314\t125.242\t52.348\n
        \tR3\t8.450\t121.893\t56.297\n
        e.g., (for .csv)\n
        residue,H,N,CA\n
        A2,8.314,125.242,52.348\n
        R3,8.450,121.893,56.297\n
        \n
        '''
        self.uploaded_file = st.file_uploader('Load peaklist associated with \"initial condition\"', type=['tab','csv'], help=help_message)

        if "df" not in st.session_state or self.uploaded_file is None:
            st.subheader('Loaded chemical shifts (initial condition):')
            st.session_state.df = pl.DataFrame(schema={"residue":str, "H (ppm)": float, "N (ppm)": float, "C (ppm)": float, "CA (ppm)": float, "CB (ppm)": float, "HA (ppm)": float, "HB (ppm)": float})       
        
        if self.uploaded_file is not None:
            if(len(self.aa_sequence)<5):
                st.error(body = f'Please insert an amino acid sequence of 5 or more residues before loading a peaklist.',icon="🚨")
            else:
                error = False
                filetype = Path(self.uploaded_file.name).suffix
                if(filetype == 'csv'):
                    self.dataframe = pl.read_csv(self.uploaded_file)
                elif(filetype == '.tab'):
                    text = self.uploaded_file.getvalue().decode()

                    # Replace arbitrary whitespace with a single tab
                    cleaned = re.sub(r"[ \t]+", "\t", text)

                    # Put cleaned text into a text buffer
                    buffer = StringIO(cleaned)

                    self.dataframe = pl.read_csv(buffer, separator="\t")

                else:
                    error = True
                    st.error(body = f'The peaklist file was not read correctly, please ensure it has the correct format and try again.',icon="🚨")

                if(error==False):
                    columns = self.dataframe.columns
                    columns_stripped = []
                    for i, column in enumerate(columns):
                        if(i==0):
                            columns_stripped.append(column)
                        elif(column in atoms):
                            columns_stripped.append(column)
                    self.dataframe = self.dataframe.select(columns_stripped)
                    if(self.dataframe.height > len(self.aa_sequence)):
                        st.error(body = f'The peaklist has more rows than the number of residues in the inputted sequence. Please reduce the number of rows in the peaklist to the length of the protein sequence and try again.',icon="🚨")
                    else:
                        numbers = []
                        names = self.dataframe.to_series(0).to_list()
                        zero_numbers = []
                        more_than_one_number = []
                        for name in names:
                            number = re.findall(r'(?<!\d)([1-9]\d{0,2}|1000)(?!\d)', name)
                            if(len(number)==0):
                                zero_numbers.append(name)
                            elif(len(number)>1):
                                more_than_one_number.append(name)
                            else:
                                numbers.append(number)

                        if(len(zero_numbers)!=0):
                            error=True
                            st.error(body = f'The following residue labels have no identifying numbers ({zero_numbers}). Please use numbers to indicate the residue position in the context of the protein sequence and try again.',icon="🚨")
                        if(len(more_than_one_number)!=0):
                            error=True
                            st.error(body = f'The following residue labels contain more than one identifying number ({more_than_one_number}). Please use one number in the residue name to indicate the residue position in the context of the protein sequence and try again.',icon="🚨")

                        if(error==False):
                            st.subheader('Loaded chemical shifts (initial condition):')
                            st.session_state.df = self.dataframe

        self.placeholder = st.empty()
        self.placeholder.dataframe(st.session_state.df)

        st.subheader('Initial condition:')
        col1, col2, col3 = st.columns(3)
        self.pH_value = col1.text_input(label="pH", value='')
        self.temp_value = col2.text_input(label="Temperature (K)", value='')
        self.ionic_strength = col3.text_input(label="Ionic Strength (M)", value='', help='The predictions rely on the ionic strength remaining unchanged upon the pH or temperature change.')

        
        self.output = st.selectbox('Calculation to perform', ['Single new condition', 'Temperature ramp (plot only)', 'pH titration (plot only)'])
        
        if(self.output == 'Single new condition'):
            st.subheader('New condition:')
            col11, col12, col13 = st.columns(3)
            self.pH_value1 = col11.text_input(label="pH", key=11, value='')
            self.temp_value1 = col12.text_input(label="Temperature (K)", key=12, value='')

        elif(self.output == 'Temperature ramp (plot only)'):
            st.subheader('Additional temperatures to calculate (K):')
            self.temperature_list = st.text_input(label='Temperatures (K)', value='', help='Input temperatures as a list of strings (in Kelvin) separated by commas (e.g., 298, 303, 308)')
        
        else:
            st.subheader('Additional pH values:')
            self.pH_list = st.text_input(label='pH values', value='', help='Input pH values as a list of strings separated by commas (e.g., 7.0, 6.5, 6.0, 5.5)')

        
        st.button('Calculate pH/temperature correction', on_click=self.run_corrections)

        if(self.output == 'Single new condition'):
            st.subheader('New condition peaklist:')
            # Add a table showing the updated chemical shifts to the page
            if "df2" not in st.session_state:
                st.session_state.df2 = pl.DataFrame(schema={"residue":str, "H (ppm)": float, "N (ppm)": float, "C (ppm)": float, "CA (ppm)": float, "CB (ppm)": float, "HA (ppm)": float, "HB (ppm)": float})       
            
            self.placeholder2 = st.empty()
            self.placeholder2.dataframe(st.session_state.df2)

        try:
            len(st.session_state.df_array)
        except:
            st.session_state.df_array = []
        
        if(self.output == 'Single new condition'):
            if(st.session_state.df2.height > 1):
                self.predicted_spectra_layout()
        else:
            if(self.output == 'Temperature ramp (plot only)'):
                if(self.temperature_list==''):
                    st.session_state.df_array = []
            if(self.output == 'pH titration (plot only)'):
                    if(self.pH_list==''):
                        st.session_state.df_array = []
            if(len(st.session_state.df_array)>1):
                self.predicted_spectra_layout()




    def run_potenci_calculation(self, potenci, aa_sequence, vals, residue_column, name_conversion_dictionary, correction=True, pH_temperature_predict_page=True, sign=1):
        pH_value, temp_value, ionic_strength = vals
        potenci_result = potenci.perform_potenci_calc(aa_sequence, [pH_value, temp_value, ionic_strength], correction, pH_temperature_predict_page)
        potenci_result = self.format_results(potenci_result, sign = sign)
        # Rename columns in potenci_result1 dataframe to match that of the residue_column
        potenci_result = potenci_result.rename({"residue": residue_column})
        potenci_result = potenci_result.with_columns(pl.col(residue_column).replace(name_conversion_dictionary))

        return potenci_result
                

    def run_corrections(self):
        """
        This code runs POTENCI (temperature/pH corrections only) on the 1 letter amino acid sequence provided. 
        """

        if(self.perform_checks()==True):
            potenci = potenci_backend()
            residue_column = list(self.dataframe.columns)[0]

            # Rename residue names in potenci_result1 and potenci_result2 to match that in the loaded peaklist
            self.name_conversion_dictionary = {}
            for name in self.dataframe[residue_column].to_list():
                number = str(re.findall(r'(?<!\d)([1-9]\d{0,2}|1000)(?!\d)', name)[0])
                self.name_conversion_dictionary[number] = name

            potenci_result1 = self.run_potenci_calculation(potenci, self.aa_sequence, [self.pH_value, self.temp_value, self.ionic_strength], residue_column, self.name_conversion_dictionary, correction=True, pH_temperature_predict_page=True, sign=1)
            dataframe_filtered = self.dataframe.join(potenci_result1, on=residue_column, how="semi")

            potenci_results_2 = []
            if(self.output == 'Temperature ramp (plot only)'):
                self.pH_value1 = self.pH_value
                for temp in [float(x) for x in self.temperature_list.split(',')]:
                    potenci_result = self.run_potenci_calculation(potenci, self.aa_sequence, [self.pH_value1, temp, self.ionic_strength], residue_column, self.name_conversion_dictionary, correction=True, pH_temperature_predict_page=True, sign=-1)
                    potenci_results_2.append(potenci_result)
            elif(self.output == 'pH titration (plot only)'):
                self.temp_value1 = self.temp_value
                for pH in [float(x) for x in self.pH_list.split(',')]:
                    potenci_result = self.run_potenci_calculation(potenci, self.aa_sequence, [pH, self.temp_value1, self.ionic_strength], residue_column, self.name_conversion_dictionary, correction=True, pH_temperature_predict_page=True, sign=-1)
                    potenci_results_2.append(potenci_result)
            else:
                potenci_results_2.append(self.run_potenci_calculation(potenci, self.aa_sequence, [self.pH_value1, self.temp_value1, self.ionic_strength], residue_column, self.name_conversion_dictionary, correction=True, pH_temperature_predict_page=True, sign=-1))

            

            potenci1_filtered = potenci_result1.join(dataframe_filtered, on=residue_column, how="semi")

            potenci_results_2_filtered = []
            for potenci_results2 in potenci_results_2:
                potenci2_filtered = potenci_results2.join(dataframe_filtered, on=residue_column, how="semi")
                potenci_results_2_filtered.append(potenci2_filtered)

            numeric_cols = [c for c, dt in self.dataframe.schema.items() if dt.is_numeric()]

            # The updated peaklist = original_peaklist + potenci_result1 - potenci_result2 (potenci_result2 is already negative so just add here)

            st.session_state.df_array = []
            for potenci2 in potenci_results_2_filtered:

                dataframe_new = dataframe_filtered.select(numeric_cols) + potenci1_filtered.select(numeric_cols) + potenci2.select(numeric_cols)
                dataframe_new = dataframe_new.with_columns(pl.Series(residue_column, dataframe_filtered[residue_column]))
                # Need to update order of columns
                columns = dataframe_new.columns
                columns_updated = [columns[-1]]+columns[:-1]
                dataframe_new = dataframe_new.select(columns_updated)
                
                
                st.session_state.df_array.append(dataframe_new)
            

                if(self.output == 'Single new condition'):
                    st.session_state.df2 = dataframe_new

        else:
            return
        
        
    def format_results(self, shift_dict: dict, sign = 1):
        """
        Update the table with POTENCI predicted random coil chemical shifts

        sign - either +1/-1 and it dictates whether to add or subtract the obtained correction
        """

        atom_columns = self.dataframe.columns[1:]

        schema={"residue":str}
        for atom in atom_columns:
            schema[atom] = float

        df_new = pl.DataFrame(schema = schema, orient='row')
        for residue, residue_dict in shift_dict.items():
            if(residue_dict == {}):
                continue
            residue_list = []
            residue_list.append(residue[0])
            for atom in atom_columns:
                if(atom == 'HN'):
                    atom = 'H'
                elif(atom == 'CO'):
                    atom = 'C'
                try:
                    residue_list.append(residue_dict[atom]*sign)
                except:
                    residue_list.append(None)

            df = pl.DataFrame([residue_list], schema=schema, orient='row')
            df_new = pl.concat([deepcopy(df_new), df], how='vertical')


        # the first column of this dataframe is residue number, the dataframe needs to be stripped and reordered
        # to only include the residues that are also in the dataframe from the loaded peaklist.

        dataframe_new = pl.DataFrame(schema = schema, orient='row')
        names = self.dataframe.to_series(0).to_list()
        for name in names:
            try:
                number = str(re.findall(r'(?<!\d)([1-9]\d{0,2}|1000)(?!\d)', name)[0])
                row = list(df_new.filter(pl.col('residue')==number).row(0))
                df = pl.DataFrame([row], schema=schema, orient='row')
                dataframe_new = pl.concat([deepcopy(dataframe_new), df], how='vertical')
            except:
                # The residue is not in the initial peaklist is the most likely failure here
                pass


        return dataframe_new
    

    

            
            

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
        

        # Check that the initial peaklist has been supplied and loaded correctly
        if self.uploaded_file is None:
            st.error(body = f'A peaklist for the initial condition has not been added correctly. Please add a suitable peaklist and try again.',icon="🚨")
            return False
        
        # Check that the user-supplied conditions are valid
        try:
            float(self.pH_value)
        except:
            st.error(body = f'The pH value {self.pH_value} for the initial condition could not be converted to a float. Please try again.',icon="🚨")
            return False
        try:
            float(self.temp_value)
        except:
            st.error(body = f'The temperature value {self.temp_value} for the initial condition could not be converted to a float. Please try again.',icon="🚨")
            return False
        try:
            float(self.ionic_strength)
        except:
            st.error(body = f'The ionic strength value {self.ionic_strength} for the initial condition could not be converted to a float. Please try again.',icon="🚨")
            return False
        
        if(self.output == 'Single new condition'):
            try:
                float(self.pH_value1)
            except:
                st.error(body = f'The pH value {self.pH_value1} for the new condition could not be converted to a float. Please try again.',icon="🚨")
                return False
            try:
                float(self.temp_value1)
            except:
                st.error(body = f'The temperature value {self.temp_value1} for the new condition could not be converted to a float. Please try again.',icon="🚨")
                return False
        elif(self.output == 'Temperature ramp (plot only)'):
            # Check to see that the teperatures can be split up and that the float of each value can be taken
            self.temp_list = []
            try:
                t_list = self.temperature_list.split(',')
                self.temp_list = [float(x) for x in t_list]

            except:
                st.error(body = f'The list of temperatures supplied could not be converted to floats. Please insert values separated by commas. e.g. 298, 303, 308',icon="🚨")
                return False
            
            if(len(self.temp_list) < 1):
                st.error(body = f'The list of temperatures has less than 1 value, please insert more values separated by commas and try again.',icon="🚨")
            

        else:
            # Check to see that the pH values can be split up and that the float of each value can be taken
            self.pH_list_vals = []
            try:
                pH_list = self.pH_list.split(',')
                self.pH_list_vals = [float(x) for x in pH_list]

            except:
                st.error(body = f'The list of pH values supplied could not be converted to floats. Please insert values separated by commas. e.g. 7.2, 7.4, 7.6',icon="🚨")
                return False
            
            if(len(self.pH_list_vals) < 1):
                st.error(body = f'The list of pH values has less than 1 value, please insert more values separated by commas and try again.',icon="🚨")
            

        
        return True
        

    def get_data_array(self, dataframe, pH, temperature):
        if('bond' in self.combination):
                if('CO' in self.combination):
                    atom1, atom2 = 'CO', 'N'
                else:
                    atom1, atom2 = 'C', 'N'
        else:
            atom1, atom2 = self.combination.split('-')
        dataframe = dataframe.select([dataframe.columns[0], atom1, atom2])

        residues = dataframe[dataframe.columns[0]]

        

        if(self.combination == 'C-N (1-bond)' or self.combination == 'CO-N (1-bond)'):
            # Need to shift the C/CO chemical shifts to be matched with their corresponding pair (need to also delete entries which do not have an i-1 CO residue assigned)
            residues2 = dataframe[dataframe.columns[0]].to_list()
            residues2_new = []
            # Rename residue names in potenci_result1 and potenci_result2 to match that in the loaded peaklist
            self.name_conversion_dictionary = {}
            for name in self.dataframe[dataframe.columns[0]].to_list():
                number = str(re.findall(r'(?<!\d)([1-9]\d{0,2}|1000)(?!\d)', name)[0])
                self.name_conversion_dictionary[int(number)] = name
            name_conversion_dictionary2 = dict((v,k) for k,v in self.name_conversion_dictionary.items())
            for res in residues2:
                residues2_new.append(name_conversion_dictionary2[res])
            residues2_new = sorted(residues2_new)
            consecutive_runs = [residues2_new[0]]
            non_consecutive_runs = [residues2_new[0]]
            for i in range(1, len(residues2_new)):
                if residues2_new[i] == residues2_new[i-1] + 1:
                    consecutive_runs.append(residues2_new[i])
                else:
                    non_consecutive_runs.append(residues2_new[i])

            # Remove the non-consecutive values from consecutive_runs
            consecutive_runs = [v for v in consecutive_runs if v not in non_consecutive_runs or v == residues2_new[0]]
            consecutive_runs2 = [self.name_conversion_dictionary[v] for v in consecutive_runs]
            dataframe = dataframe.filter(pl.col(dataframe.columns[0]).is_in(consecutive_runs2))

            dataframe = dataframe.with_columns(pl.col(atom1).shift())

        dataframe = dataframe.drop_nulls()


        x = dataframe[atom1]
        y = dataframe[atom2]
        text1 = f'pH={pH}, T={temperature}K'

        return x, y, residues, text1, atom1, atom2
    
    def colours(self,n):
        def lerp(a, b, t):
            return a + (b - a) * t

        def interp_color(c1, c2, t):
            return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))

        def rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(*rgb)

        def gradient_hex(n, start="#0000ff", end="#ff0000"):
            # convert hex → RGB
            s = tuple(int(start[i:i+2], 16) for i in (1, 3, 5))
            e = tuple(int(end[i:i+2], 16) for i in (1, 3, 5))

            return [
                rgb_to_hex(interp_color(s, e, i/(n-1)))
                for i in range(n)
            ]

        # Example:
        colors = gradient_hex(n)
        return colors

    
    def plot_predicted_spectra(self):
        
        x, y, residues, text, atom1, atom2 = self.get_data_array(dataframe=st.session_state.df, pH=self.pH_value, temperature=self.temp_value)
        
        # Get a temperature gradient of colours
        if(self.output == 'Temperature ramp (plot only)'):
            colors = self.colours(len(self.temperature_list.split(','))+1)
        elif(self.output == 'pH titration (plot only)'):
            colors = self.colours(len(self.pH_list.split(','))+1)
        else:
            colors = self.colours(2)

        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x.to_numpy(), y=y.to_numpy(), hovertext=residues, mode='markers', name=text, marker_color=colors[0]))
        xlabel = atom1 + ' Chemical Shift (ppm)'
        ylabel = atom2 + ' Chemical Shift (ppm)'
        fig.update_layout(xaxis = dict(title=xlabel, autorange='reversed'))
        fig.update_layout(yaxis = dict(title=ylabel, autorange='reversed'))
        
        for i, dataframe in enumerate(st.session_state.df_array):
            if(self.output == 'Temperature ramp (plot only)'):
                x, y, residues, text, atom1, atom2 = self.get_data_array(dataframe=dataframe, pH=self.pH_value, temperature=[float(x) for x in self.temperature_list.split(',')][i])
            elif(self.output == 'pH titration (plot only)'):
                x, y, residues, text, atom1, atom2 = self.get_data_array(dataframe=dataframe, pH=[float(x) for x in self.pH_list.split(',')][i], temperature=self.temp_value)
            else:
                x, y, residues, text, atom1, atom2 = self.get_data_array(dataframe=dataframe, pH=self.pH_value1, temperature=self.temp_value1)
            
            fig.add_trace(go.Scatter(x=x.to_numpy(), y=y.to_numpy(), hovertext=residues, mode='markers', name=text, marker_color=colors[i+1]))
            

        st.plotly_chart(fig, use_container_width=True)
        


    def predicted_spectra_layout(self):
        st.subheader("Predicted spectra")
        atom_combination = []
        columns = st.session_state.df_array[0].columns
        if('H' in columns and 'N' in columns):
            atom_combination.append('H-N')
        if('CA' in columns and 'N' in columns):
            atom_combination.append('CA-N')
        if('CA' in columns and 'C' in columns):
            atom_combination.append('CA-C')
        if('CA' in columns and 'CO' in columns):
            atom_combination.append('CA-CO')
        if('C' in columns and 'N' in columns):
            atom_combination.append('C-N (1-bond)')
            atom_combination.append('C-N (2-bond)')
        if('CO' in columns and 'N' in columns):
            atom_combination.append('CO-N (1-bond)')
            atom_combination.append('CO-N (2-bond)')
        if('CA' in columns and 'CB' in columns):
            atom_combination.append('CA-CB')
        if('HA' in columns and 'CA' in columns):
            atom_combination.append('HA-CA')
        if('HB' in columns and 'CB' in columns):
            atom_combination.append('HB-CB')
        if('HA' in columns and 'HB' in columns):
            atom_combination.append('HA-HB')
        self.combination = st.selectbox("Select atom combination:", atom_combination)
        self.plot_predicted_spectra()

# Create the page
page = potenci_page()