import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import darkdetect
from pathlib import Path
import re
import json
from io import StringIO
import pynmrstar
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tools.ChemicalShiftTools.Potenci.potenci import potenci_backend
from tools.ChemicalShiftTools.SpinForecast.SpinForecast import SpinForecastBackend

# This page creates a front end tool where users can run SpinForecast for sequence specific chemical shift distributions
# The page also produces plots of the predicted two-dimensional NMR spectra in each case (e.g. H-N, HA-CA, CA-N, C-N, CA-C etc)
# The backend code is in SpinForecast/SpinForecast.py


st.set_page_config(layout="wide")

# Defining useful information used thbroughout code
residue_1_letter_codes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
residue_dict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W','TYR': 'Y','VAL': 'V', '': 'X'}
residue_dict_reversed = {(v,k) for k,v in residue_dict.items()}
colors = px.colors.qualitative.Alphabet
color_dict = {}
colormap_dict = {}

if(darkdetect.isDark()==False):
    for i, r in enumerate(residue_1_letter_codes):
        color_dict[r] = colors[i]
        colormap_dict[r] = ['white', colors[i]]
else:
    for i, r in enumerate(residue_1_letter_codes):
        color_dict[r] = colors[i]
        colormap_dict[r] = ['#0f1117', colors[i]]

class SpinForecast():
    """
    This class will produce a page to perform potenci random coil
    chemical shift prediction.
    """
    def __init__(self):

        self.create_page()
        

    def create_page(self):
        """
        Create the page for the user to insert the amino acid sequence and conditions to
        perform the DisAssign
        """
        st.title("SpinForecast: Sequence Specific Disordered Chemical Shift Distributions and Assignment")
        st.markdown("SpinForecast will plot the probability distribution functions for the chemical shift of each atom of each residue in a user-inputed sequence.")
        st.markdown("If you use this tool, please cite the references shown below.")
        st.markdown("**References**")
        st.markdown("SpinForecast (Bind Research, 2026)")
        st.markdown("BMRB: https://doi.org/10.1093/nar/gkac1050, https://bmrb.io")
        st.markdown("POTENCI (Nielsen and Mulder, 2018): https://doi.org/10.1007/s10858-018-0166-5")
        st.markdown("Temperature corrections (Kjaergaard et al. 2011): https://doi.org/10.1007/s10858-011-9508-2, https://doi.org/10.1007/s10858-011-9472-x")
        st.markdown("pH corrections (Platzer et al. 2014): https://doi.org/10.1007/s10858-014-9862-y")

        # Add logo to the sidebar
        st.logo('./bind-logo-alpha.svg', size="large", link="https://bindresearch.org", icon_image='./bind-logo-alpha.svg')
        
        st.subheader('Sequence and conditions')
        
        if('aa_sequence' in st.session_state):
            value = st.session_state.aa_sequence
        else:
            value=''
        self.aa_sequence = st.text_area(label='Sequence of amino acids (1 letter codes)', value=value, help='At least 5 residues must be provided. Note, the terminal 2 residues at the N and C terminus are excluded from the results.')
        
        col1, col2, col3 = st.columns(3)
        self.pH_value = col1.text_input(label="pH", value='7.4')
        self.temp_value = col2.text_input(label="Temperature (K)", value='298.0')
        self.ionic_strength = col3.text_input(label="Ionic Strength (M)", value='0.15')
        st.button('Plot predicted distributions', on_click=self.run_spinforecast)


        if "distribution_dictionary" not in st.session_state:
            st.session_state.distribution_dictionary =  {}
        

        if(len(list(st.session_state.distribution_dictionary.keys())) > 1):
            st.subheader("Predicted spectra")
            self.combination = st.selectbox("Select atom combination:", ['H-N', 'HA-CA', 'HB-CB', 'CA-N', 'C-N (1-bond)', 'C-N (2-bond)', 'CA-CB', 'HA-C'])

            self.plot_predicted_distributions()


        if(len(list(st.session_state.distribution_dictionary.keys())) > 1):
            # Add assignment tool
            self.add_assignment_probabilities()

        if('possible_assignments' in st.session_state):
            self.plot_predicted_assignments()

    def run_spinforecast(self):
        """
        This code runs SpinForecast on the 1 letter amino acid sequence provided. 
        """

        if(self.perform_checks()==True):

            st.session_state.aa_sequence = self.aa_sequence
            
            # Get temperature and pH corrections using POTENCI
            potenci = potenci_backend()
            potenci_result = potenci.perform_potenci_calc(self.aa_sequence, [self.pH_value, self.temp_value, self.ionic_strength], correction=True, pH_temperature_predict_page=True)
            potenci_result_df = self.format_results(potenci_result)

            predict_distributions = SpinForecastBackend()
            distributions_result = predict_distributions.perform_calc(st.session_state.aa_sequence, potenci_result_df)
            
            # Set the dataframe to the new state
            st.session_state.distribution_dictionary = distributions_result


        else:
            return
        
    
        
        
        
    def format_results(self, shift_dict: dict):
        """
        Create a dataframe of the predicted values
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

        return dataframe
    

    def plot_predicted_distributions(self):
        """
        Plotting a normal distribution of the chemical shift for each residue using the predicted mean and uncertainty values.

        Colour coding these by residue type and allow in the legend for individual residue types to be toggled on and off
        Also plottomg a scatter plot of the mean values (colour coded by residue type) and hover over this for peak information.

        Also adding the ability for a user to add in a peak list of their own to plot as a scatter plot and compare (colour = black)
        """

        dictionary = st.session_state.distribution_dictionary
        if(self.combination == 'C-N (1-bond)' or self.combination == 'C-N (2-bond)'):
            atom1 = 'C'
            atom2 = 'N'
        else:
            atom1, atom2 = self.combination.split('-')
        dictionary1 = dictionary[atom1]
        dictionary2 = dictionary[atom2]
        if(self.combination == 'C-N (1-bond)'):
            # Need to shift the C residue numbers + 1 (as the i-1 C atom is linked to the i N atom)
            dictionary1_new = {}
            for number in dictionary1.keys():
                dictionary1_new[str(int(number)+1)]= dictionary1[number]
            import copy
            dictionary1 = copy.deepcopy(dictionary1_new)

        

        fig = go.Figure()
        for i, residue in enumerate(list(dictionary2.keys())):
            if(residue not in list(dictionary1.keys())):
                continue
            if(residue not in list(dictionary2.keys())):
                continue
                
            xvals = dictionary1[residue]
            yvals = dictionary2[residue]

            x = xvals
            y = yvals

            kde_x = gaussian_kde(x)
            kde_y = gaussian_kde(y)

            # nx = int((max(x)-min(x))/bin_width1)
            # ny = int((max(y)-min(y))/bin_width2)

            nx=50
            ny=50

            rangex = max(x)-min(x)
            rangey = max(y)-min(y)
            x_counts, x_edges = np.histogram(x, bins=nx, range=(min(x)-rangex/2, max(x)+rangex/2))
            y_counts, y_edges = np.histogram(y, bins=ny, range=(min(y)-rangey/2, max(y)+rangey/2))

            x_centers = 0.5*(x_edges[:-1] + x_edges[1:])
            y_centers = 0.5*(y_edges[:-1] + y_edges[1:])


            x_smooth = kde_x(x_centers)
            y_smooth = kde_y(y_centers)

            # Adding 2D grid
            Z = np.outer(y_smooth, x_smooth)
            Z = Z / np.sum(Z)

            max_x = np.argmax(Z, axis=1)
            max_y = np.argmax(Z, axis=0)
                

            label = residue + self.aa_sequence[int(residue)-1]
            color = color_dict[self.aa_sequence[int(residue)-1]]
            colormap = colormap_dict[self.aa_sequence[int(residue)-1]]

            # Plotting a 2D histogram based on the xvals and yvals distributions
            fig.add_trace(go.Contour(
                x=x_centers,
                y=y_centers,
                z=Z,
                contours=dict(coloring="lines"),
                colorscale=colormap,
                name=label,
                showscale=False,
                showlegend=True,
                hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(x=[x_centers[max_x][0]], y=[y_centers[max_y][0]], hovertext=label, text=label,  mode='markers', marker_color=color, showlegend=True, name=label))



        xlabel = atom1 + ' Chemical Shift (ppm)'
        ylabel = atom2 + ' Chemical Shift (ppm)'

        fig.update_layout(xaxis = dict(title=xlabel, autorange='reversed'))
        fig.update_layout(yaxis = dict(title=ylabel, autorange='reversed'))
            
        st.plotly_chart(fig, use_container_width=True)




    def plot_predicted_spectra(self):

        dataframe = st.session_state.distribution_dictionary
        atom1, atom2 = self.combination.split('-')
        dataframe = dataframe.select(['residue number', 'amino acid', atom1, atom2])
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
            
        st.plotly_chart(fig, use_container_width=True)


    def add_assignment_probabilities(self):
        """
        Add the functionality for users to input a peaklist and use this for probabilistic assignments
        """

        help_message = '''The peaklist should be in tabular format (.tab/.list), csv or nef (CCPN) format. For tabular and csv formats, the residue column must contain the residue number to link it to the protein sequence.\n
        e.g., (for .tab/.list):\n
        peak_name\tH\tN\tCA\n
        \t1H-N\t8.314\t125.242\t52.348\n
        \t2H-N\t8.450\t121.893\t56.297\n
        e.g., (for .csv)\n
        1H-N,8.314,125.242,52.348\n
        2H-N,8.450,121.893,56.297\n
        \n
        '''
        self.uploaded_file = st.file_uploader('Load peaklist"', type=['tab','csv','list','nef'], help=help_message)

        if "df_peaklist" not in st.session_state or self.uploaded_file is None:
            st.subheader('Loaded chemical shifts:')
            st.session_state.df_peaklist = pl.DataFrame(schema={"residue":str, "atom 1 (ppm)": float, "atom 2 (ppm)": float, "atom 3 (ppm)": float})       

        if self.uploaded_file is not None:
                error = False
                filetype = Path(self.uploaded_file.name).suffix
                if(filetype == '.csv'):
                    self.dataframe_peaklist = pl.read_csv(self.uploaded_file)
                elif(filetype == '.tab' or filetype == '.list'):
                    text = self.uploaded_file.getvalue().decode()

                    # Replacing arbitrary whitespace with a single tab (to allow for incorrectly formatted peaklist)
                    cleaned = re.sub(r"[ \t]+", "\t", text)
                    buffer = StringIO(cleaned)
                    self.dataframe_peaklist = pl.read_csv(buffer, separator="\t")

                elif(filetype == '.nef'):
                    self.dataframe_peaklist, error = self.read_nef_file()

                else:
                    error = True
                    st.error(body = f'The peaklist file was not read correctly, please ensure it has the correct format and try again.',icon="🚨")

                if(error==False):
                    columns = self.dataframe_peaklist.columns
                    columns_stripped = []
                    for i, column in enumerate(columns):
                        if(i==0):
                            columns_stripped.append(column)
                        elif(column in ['H', 'N', 'CO', 'C', 'CO(i-1)', 'C(i-1)', 'CA', 'CA(i-1)', 'CB','CB(i-1)', 'HA', 'HB']):
                            columns_stripped.append(column)
                        
                    self.dataframe_peaklist = self.dataframe_peaklist.select(columns_stripped)
                    st.subheader('Loaded chemical shifts:')
                    st.session_state.df_peaklist = self.dataframe_peaklist

        self.placeholder2 = st.empty()
        if(len(list(st.session_state.df_peaklist)) > 1):
            self.placeholder2.dataframe(st.session_state.df_peaklist)

            st.markdown('Missing values in the peaklist are denoted 999 in the table')

            st.markdown('')

            # Adding a row of tickboxes so the atoms being used for assessing probabilities can be adjusted
            st.markdown('Select which atom classifications to use for calculating the assignment predictions:')
            columns = st.session_state.df_peaklist.columns
            number_of_cols = len(columns)-1
            cols = st.columns(number_of_cols)
            self.column_checkboxes = []
            for j, col in enumerate(cols):
                self.column_checkboxes.append(col.checkbox(label=columns[j+1], value=True))

            st.markdown('Add a referencing correction (if required) for each atom:')
            dictionary = {}
            for atom in st.session_state.df_peaklist.columns[1:]:
                dictionary[atom] = 0.0
            df = pd.DataFrame([dictionary])
            self.reference_df = st.data_editor(df)

            st.button("Assess probabilities", on_click=self.assess_probabilities)


    def read_nef_file(self):
        """
        Reading a CCPN style nef file to be used as a peaklist for SpinForecast to perform predictions on
        """
        try:
            nef_dictionary = json.loads(self.get_json_data(self.uploaded_file))['saveframes']
            index = 2
            for j in range(len(nef_dictionary)):
                if nef_dictionary[j]['name'] == 'nef_chemical_shift_list_default':
                    index = j

            nef_dictionary = nef_dictionary[index]
            

        except:
            nef_dictionary = None
            error = True
            return nef_dictionary, error
        

        shift_data = nef_dictionary['loops'][0]['data']
        peaklist_dictionary = {}

        atom_list = []
        
        for row in shift_data:
            peak_name = row[1].split('-1')[0]
            iminus1 = ''
            if('-1' in row[1]):
                iminus1 = '(i-1)'
            if('+1' in peak_name):
                # Not implemented currently
                continue
            if(peak_name not in peaklist_dictionary):
                peaklist_dictionary[peak_name] = {}
            atom = row[3]+iminus1
            if(atom not in atom_list):
                atom_list.append(atom)
            shift = float(row[4])
            peaklist_dictionary[peak_name][atom] = shift


        schema = {'peak_name':str}
        for atom in atom_list:
            schema[atom] = float

        from copy import deepcopy


        for k, (peak_name, peak_values) in enumerate(peaklist_dictionary.items()):
            row = []
            row.append(peak_name)
            for atom in atom_list:
                if(atom in peak_values):
                    row.append(peak_values[atom])
                else:
                    row.append(999)
            if(k==0):
                df = pl.DataFrame([row], schema=schema, orient='row')
            else:
                dataframe = pl.DataFrame([row], schema=schema, orient='row')
                df = pl.concat([deepcopy(df), dataframe], how='vertical')

        error=False



        return df, error

    @staticmethod
    def get_json_data(infile):
        entry = pynmrstar.Entry.from_file(infile)

        # Convert to dictionary-like structure
        data = entry.get_json()
        return data
    

    def assess_probabilities(self):
        
        dictionary = st.session_state.distribution_dictionary

        
        columns = self.dataframe_peaklist.columns[1:]
        self.atoms = []
        indexes = []

        # Defining weights to describe the contribution of each atom to the overall prediction
        # H chemical shifts are downweighted as they are highly variable compared to other nuclei
        weights = {'CA':1, 'CA(i-1)':1, 'CB':1, 'CB(i-1)':1, 'N':1, 'C':1,'C(i-1)':1,'CO':1,'CO(i-1)':1, 'H':0.5}

        for j, checkbox in enumerate(self.column_checkboxes):
            if(checkbox==True):
                self.atoms.append(columns[j])
                indexes.append(j+1)

        
        

        dictionaries = {}
        for atom in self.atoms:
            atom_stripped = atom.strip('(i-1)')
            dictionaries[atom] = dictionary[atom_stripped]

            if('(i-1)' in atom):
                # Need to shift residue numbers + 1 (as the i-1 residue atom is linked to the i residue)
                dictionary_new = {}
                for number in dictionaries[atom].keys():
                    dictionary_new[str(int(number)+1)]= dictionaries[atom][number]
                import copy
                dictionaries[atom]= copy.deepcopy(dictionary_new)


        kdes = {}
        kdes_support = {}

        lists = []
        for atom in self.atoms:
            residues = list(dictionaries[atom].keys())
            if('i-1' in atom):
                residues = residues[:-1]
            lists.append(residues)

        
        list_of_residues = []
        for k,list_ in enumerate(lists):
            list_of_residues = set(list_+list(list_of_residues))


        for i, residue in enumerate(list_of_residues):
            kdes[residue] = {}
            kdes_support[residue] = {}
            for atom in self.atoms:
                try:
                    vals = dictionaries[atom][residue]
                    kde = gaussian_kde(vals)
                    kdes[residue][atom] = kde
                    support_percentile = 0.1
                    lo = np.percentile(vals, support_percentile * 100)
                    hi = np.percentile(vals, (1 - support_percentile) * 100)
                    margin = (hi - lo) * 0.2
                    kdes_support[residue][atom]=(lo - margin, hi + margin)
                except:
                    pass
            
        
        
        classes = list(list_of_residues)
        priors = {cls: 1 / len(classes) for cls in classes}

        

        def probability_calc(shifts):
            log_scores = {}
            counts = {}
            log_scores_per_atom = {}
            log_evidence_score = {}

            for cls in classes:
                try:
                    probability = np.log(priors[cls])
                    evidence = 0
                    count=0
                    for i, atom in enumerate(self.atoms):
                        if(shifts[i]==999):
                            # No chemical shift inputted
                            continue
                        else:
                            p = kdes[cls][atom](shifts[i])[0]
                            probability = probability+np.log(p)*weights[atom]
                            evidence = evidence+np.log(p)*weights[atom]
                            count+=1


                            if(atom not in log_scores_per_atom):
                                log_scores_per_atom[atom] = {}
                            log_scores_per_atom[atom][cls] = np.log(priors[cls])
                            
                            for j, atom2 in enumerate(self.atoms):
                                if(atom2!=atom):
                                    if(shifts[j]==999):
                                        # No chemical shift inputted
                                        continue
                                    else:
                                        p = kdes[cls][atom2](shifts[j])[0]
                                        log_scores_per_atom[atom][cls] += np.log(p)*weights[atom2]


                    log_scores[cls] = probability
                    log_evidence_score[cls] = evidence
                except:
                    pass

            max_val = max(log_scores.values())
            
            exp_scores = {c: np.exp(v - max_val) for c, v in log_scores.items()}
            total = sum(exp_scores.values())

            posteriors = {c: v / total for c, v in exp_scores.items()}




            max_val_evidence = max(log_evidence_score.values())
            exp_evidence_scores = {c: np.exp(v - max_val) for c, v in log_evidence_score.items()}
            total_evidence = sum(exp_evidence_scores.values())

            log_evidence = max_val_evidence + np.log(total_evidence)


            # Posteriors calculated with each atom removed from the prediction sequentially
            posteriors_per_atom = {}
            for atom in list(log_scores_per_atom.keys()):
                max_val_atom = max(log_scores_per_atom[atom].values())
                exp_scores_atom = {c: np.exp(v - max_val_atom) for c, v in log_scores_per_atom[atom].items()}
                total_atom = sum(exp_scores_atom.values())

                posteriors_per_atom[atom] = {c: v / total_atom for c, v in exp_scores_atom.items()}



            return posteriors, log_evidence, posteriors_per_atom
        


        peaks = []
        peak_names = []
        for row in st.session_state.df_peaklist.iter_rows():
            peak_names.append(row[0])
            peak_values = []
            for k in indexes:
                try:
                    reference_correction = self.reference_df.iloc[0][k-1]
                    reference_correction = float(reference_correction)
                except:
                    # reference factor was not loaded correctly
                    st.error(body = f'One of the referencing corrections is not a valid number, please fix this and try again.',icon="🚨")
                    return
                peak_values.append(row[k]+reference_correction)
            peaks.append(peak_values)

        probabilities = {}
        posteriors_per_atom = {}
        posterior_max = []
        total_log_evidence = []
        for i, peak in enumerate(peaks):
            probabilities[peak_names[i]], log_evidence, posteriors_per_atom[peak_names[i]]= probability_calc(peak)
            total_log_evidence.append(log_evidence)
            posterior_max.append(max(list(probabilities[peak_names[i]].values())))
   
        


        st.session_state.possible_assignments = {}
        st.session_state.out_of_distribution_score = {}
        st.session_state.possible_assignments_atom_confidence = {}
        st.session_state.assignment_report = {}
        for k, peak_name in enumerate(peak_names):

            probs = probabilities[peak_name]
            possibilities = []
            for key, val in probs.items():
                if(val>0.01):
                    # If the likelihood is larger than 0.01 then these values are included in the list of possible assignments
                    possibilities.append([key,val])

        
        


            names = []
            numbers = []
            likelihoods = []
        
            for i, d in enumerate(possibilities):
                names.append(d[0] + self.aa_sequence[int(d[0])-1])
                numbers.append(d[0])
                likelihoods.append(d[1])
            
        
            atom_set_evidence = {}
            total_set_evidence = {}

            for set_member in possibilities:
                total_set_evidence[set_member[0]]=set_member[1]

                
            percentile = {}

            for atom, atom_vals in posteriors_per_atom[peak_name].items():
                atom_set_evidence[atom] = {}
                percentile[atom] = {}
                for set_member in possibilities:
                    n = set_member[0]
                    atom_set_evidence[atom][set_member[0]]=atom_vals[n]
                
                    atom_set_evidence[atom][set_member[0]] = atom_set_evidence[atom][set_member[0]]- total_set_evidence[set_member[0]]


                    # Getting the percentile of the KDE predicted distribution per atom that the experimental value lies at
                    def cdf(x: float) -> float:
                        lo, hi = kdes_support[set_member[0]][atom]
                        lo = min(lo, x - 1e-6)
                        hi = max(hi, x + 1e-6)
                        
                        x_clamped = float(np.clip(x, lo, hi))
                        result, _ = quad(kdes[set_member[0]][atom], lo, x_clamped, limit=400)
                        norm, _  = quad(kdes[set_member[0]][atom], lo, hi, limit=400)
                        
                        if norm < 1e-300:
                            return 0.0 if x < (lo + hi) / 2 else 1.0
                
                        return float(np.clip(result / norm, 0.0, 1.0))

                        
                    if(atom not in percentile):
                        percentile[atom] = {}
                    # find index of atom in self.atoms
                    index = self.atoms.index(atom)
                    percentile[atom][set_member[0]] = cdf(peaks[k][index])

            
            # storing the dictionary value as the peak_name is going to be changed below
            atom_score = posteriors_per_atom[peak_name]


            # Taking the prediction with the largest posterior and see if more than 2 values are out of distribution
            try:
                index = likelihoods.index(max(likelihoods))
                most_likely_assignment = possibilities[index][0]
                count = 0
                for atom in list(percentile.keys()):
                    if(percentile[atom][most_likely_assignment]<0.05):
                        count+=1
                    elif(percentile[atom][most_likely_assignment]>0.95):
                        count+=1
                    else:
                        pass
            except:
                # if likelihoods is empty it means that no assignments could be predicted for this peak
                count = 10
            




            # Assigning a colour based on the Jeffreys scale
            try:
                if(count >= 2):
                    peak_name = peak_name + ' (out of distribution detection)'
                elif(max(likelihoods)>0.75):
                    peak_name = peak_name + ' (higher likelihood)'
                elif(max(likelihoods)>0.5):
                    peak_name = peak_name + ' (medium likelihood)'
                else:
                    peak_name = peak_name + ' (small likelihood)'
            except:
                # usually because no likelihoods were above 0.01 (rare but possible)
                peak_name = peak_name + ' (small likelihood)'


            st.session_state.assignment_report[peak_name] = {}
            st.session_state.assignment_report[peak_name]['names'] = names
            st.session_state.assignment_report[peak_name]['likelihoods'] = likelihoods
            st.session_state.assignment_report[peak_name]['set of predictions'] = numbers
            st.session_state.assignment_report[peak_name]['total set evidence'] = total_set_evidence
            st.session_state.assignment_report[peak_name]['atom set evidence'] = atom_set_evidence
            st.session_state.assignment_report[peak_name]['percentiles'] = percentile


            # Saving the predictions to the streamlit session
            st.session_state.possible_assignments[peak_name] = possibilities
            st.session_state.possible_assignments_atom_confidence[peak_name] = atom_score
        



    def plot_predicted_assignments(self):
        """
        Create the ability where a user can choose the peak and then the
        predicted assignments and their respective likelihoods are plotted
        in a bar chart
        """


        st.subheader("Predicted assignments")
        keys = list(st.session_state.possible_assignments.keys())
        order = {"higher likelihood": 0, "medium likelihood": 1, "small likelihood": 2, "out of distribution detection": 3}
        try:
            sorted_values = sorted(keys, key=lambda x: order[x.split('(')[1].split(')')[0]])
            keys = sorted_values
        except:
            pass
        self.peak_choice = st.selectbox("Select peak:", keys)
        names = []
        likelihoods = []
        
    

        set_of_predictions = st.session_state.assignment_report[self.peak_choice]['set of predictions']
        names = st.session_state.assignment_report[self.peak_choice]['names']
        likelihoods = st.session_state.assignment_report[self.peak_choice]['likelihoods']
        atom_set_evidence = st.session_state.assignment_report[self.peak_choice]['atom set evidence']
        total_set_evidence = st.session_state.assignment_report[self.peak_choice]['total set evidence']
        percentiles_per_atom = st.session_state.assignment_report[self.peak_choice]['percentiles']

          
        # Assigning a colour based on the Jeffreys scale for log[Bayes-factor]
        if(max(total_set_evidence.values()) < 0.05):
            color = "#5b1d2c"
        elif(max(total_set_evidence.values()) < 0.5):
            color = "#f1b291"
        elif(max(total_set_evidence.values()) < 0.75):
            color = "#82b3d4"
        else:
            color = "#1A3367"


        fig = go.Figure(data=[go.Bar(x=names,y=likelihoods, marker={'color': color})])
        fig.update_layout(xaxis = dict(title='Residue'))
        fig.update_layout(yaxis = dict(title='Posterior probability', range=[0,1]))
        if('out of distribution' in self.peak_choice):
            fig.update_layout(title='Out of distribution error (results not trustworthy)')
            
        st.plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"format": "svg","height": 600,"width": 800,"scale": 1}})
    
        st.markdown('*This set of predictions excludes predictions with a posterior probability less than 0.01.')


        st.subheader('Out of distribution tests')
        st.markdown('SpinForecast predictions should be used only as a guide to aid assignment and should be combined with other methods for validation. Peaks can sometimes be outside of the distribution of disordered residues in the BMRB.')
        
        st.markdown('Values in the table below represent the **percentile** of the chemical shift distributions (for each predicted assignment) for each atom that the experimental chemical shift appears at. Values below 0.05 and 0.95 indicate that the experimental chemical shift is out of the typical range of the predicted residue. If this is present for multiple atoms, the results should be carefully inspected and cross-validation using strip plots may be necessary to confirm the assignment. Predictions with chemical shifts for more than 1 atom out of the predicted distribution of chemical shifts results in a warning being generated. Note that it is common for amide proton atom (H) chemical shifts to be out of distribution given their significant pH and temperature dependence.')

        # Creating a table showing the per-atom evidence for the predictions made
        columns = ['Prediction']
        for atom in list(percentiles_per_atom.keys()):
            columns.append(atom)
        table_rows = []
        for prediction in set_of_predictions:
            row_set = [prediction]
            for atom in list(percentiles_per_atom.keys()):
                row_set.append(percentiles_per_atom[atom][prediction])
            table_rows.append(row_set)

        df = pd.DataFrame(table_rows, columns=columns)
        df[df.isna()] = 0
        def highlight(v):
            if v > 0.95:
                return "background-color: #f1b291"
            elif(v<0.05):
                return "background-color: #f1b291"
            return "background-color: #82b3d4"
        df = df.style.format("{:.2f}", subset=df.columns[1:]).map(highlight, subset=df.columns[1:])
        
        st.dataframe(df)

    

        st.subheader("Relative atom contribution to the predictions")
        st.markdown(r'Values in the table below, P(atom$_i$), represent the posterior probabilities for each prediction excluding the data for atom$_i$. This leave-one-out approach allows the comparison of which atoms (if any) are dominating the assignment predictions.')
        st.markdown(r'P(atom$_i$) = P(prediction|data[excluding-atom$_i$])-P(prediction|data[all-atoms]])')
   

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(colorscale='RdBu', showscale=True, cmin=-1, cmax=1, colorbar=dict(thickness=20, x=0.5, y=0.0, len=0.5, orientation='h', title=dict(text=r"P(atom)", side="top"),   # custom title
                tickvals=[-1, 0, 1],                 
                ticktext=["-1<br>Removing atom<br>decreases prediction<br>probability", "0<br>Removing atom<br>doesn't change prediction<br>probability","1<br>Removing atom<br>increases prediction<br>probability"]))))
        
        fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False), margin=dict(l=0, r=0, t=0, b=0), height=150)

    
        st.plotly_chart(fig, use_container_width=False)
        
        # Creating a table showing the per-atom evidence for the predictions made
        table_titles = ['Prediction']
        for atom in list(atom_set_evidence.keys()):
            table_titles.append('P({})'.format(atom))
        table_rows = []

        for prediction in set_of_predictions:
            row_set = [prediction+self.aa_sequence[int(prediction)-1]]
        
            for atom in list(atom_set_evidence.keys()):
                row_set.append(atom_set_evidence[atom][prediction])
            table_rows.append(row_set)

        df = pd.DataFrame(table_rows, columns=table_titles)
        df[df.isna()] = 0
        df = df.style.format("{:.2f}", subset=df.columns[1:]).background_gradient(cmap="RdBu",subset=df.columns[1:],vmin=-1,vmax=1)
        
        st.table(df) 


        st.markdown('*Note: values in each column do not always add to zero, removing an atom may increase the probabilities of residues not shown in the bar chart above.')


    

               
            
            

    def perform_checks(self):
        """
        Perform the following checks before running SpinForecast
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
page = SpinForecast()