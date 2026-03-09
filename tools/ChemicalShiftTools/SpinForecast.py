import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import gaussian_kde
from tools.ChemicalShiftTools.Potenci.potenci import potenci_backend
from tools.ChemicalShiftTools.SpinForecast.SpinForecast import SpinForecastBackend

# This page creates a front end tool where users can run BindDisAssign for sequence specific chemical shift distributions
# The page also produces plots of the predicted two-dimensional NMR spectra in each case (e.g. H-N, HA-CA, CA-N, C-N, CA-C etc)
# The backend code is in Pages/BindDisAssign


st.set_page_config(layout="wide")

residue_1_letter_codes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
residue_dict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W','TYR': 'Y','VAL': 'V', '': 'X'}
residue_dict_reversed = {(v,k) for k,v in residue_dict.items()}
colors = px.colors.qualitative.Alphabet
color_dict = {}
colormap_dict = {}
for i, r in enumerate(residue_1_letter_codes):
    color_dict[r] = colors[i]
    colormap_dict[r] = ['white', colors[i]]

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
        st.button('Plot predicted distributions', on_click=self.run_bindpredict)


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

    def run_bindpredict(self):
        """
        This code runs BindPredictDCS on the 1 letter amino acid sequence provided. 
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
        Update the table with predicted chemical shifts
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


        if(atom1 == 'H'):
            bin_width1 = 0.1
        else:
            bin_width1 = 0.5
        bin_width2 = 0.5

        

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

                nx = int((max(x)-min(x))/bin_width1)
                ny = int((max(y)-min(y))/bin_width2)

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

                # Build 2D independent grid
                Z = np.outer(y_smooth, x_smooth)
                Z = Z / np.sum(Z)

                max_x = np.argmax(Z, axis=1)
                max_y = np.argmax(Z, axis=0)
                

                label = residue + self.aa_sequence[int(residue)-1]
                color = color_dict[self.aa_sequence[int(residue)-1]]
                colormap = colormap_dict[self.aa_sequence[int(residue)-1]]

                # # Plot a 2D histogram based on the xvals and yvals distributions

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

        help_message = '''The peaklist should be in tabular format (.tab) or csv format. The residue column must contain the residue number to link it to the protein sequence.\n
        e.g., (for .tab):\n
        peak_name\tH\tN\tCA\n
        \t1H-N\t8.314\t125.242\t52.348\n
        \t2H-N\t8.450\t121.893\t56.297\n
        e.g., (for .csv)\n
        peak_name,H,N,CA\n
        1H-N,8.314,125.242,52.348\n
        2H-N,8.450,121.893,56.297\n
        \n
        '''
        self.uploaded_file = st.file_uploader('Load peaklist"', type=['tab','csv','list'], help=help_message)

        if "df_peaklist" not in st.session_state or self.uploaded_file is None:
            st.subheader('Loaded chemical shifts:')
            st.session_state.df_peaklist = pl.DataFrame(schema={"residue":str, "atom 1 (ppm)": float, "atom 2 (ppm)": float, "atom 3 (ppm)": float})       

        if self.uploaded_file is not None:
                from pathlib import Path
                import re
                from io import StringIO
                error = False
                filetype = Path(self.uploaded_file.name).suffix
                if(filetype == 'csv'):
                    self.dataframe_peaklist = pl.read_csv(self.uploaded_file)
                elif(filetype == '.tab' or filetype == '.list'):
                    text = self.uploaded_file.getvalue().decode()

                    # Replace arbitrary whitespace with a single tab
                    cleaned = re.sub(r"[ \t]+", "\t", text)

                    # Put cleaned text into a text buffer
                    buffer = StringIO(cleaned)

                    self.dataframe_peaklist = pl.read_csv(buffer, separator="\t")

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

            # Adding a row of tickboxes so the atoms being used for assessing probabilities can be adjusted
            columns = st.session_state.df_peaklist.columns
            number_of_cols = len(columns)-1
            cols = st.columns(number_of_cols)
            self.column_checkboxes = []
            for j, col in enumerate(cols):
                self.column_checkboxes.append(col.checkbox(label=columns[j+1], value=True))

            st.button("Assess probabilities", on_click=self.assess_probabilities)


    def assess_probabilities(self):
        
        dictionary = st.session_state.distribution_dictionary

        
        # atoms = ['N', 'CA', 'CA(i-1)', 'CB', 'CB(i-1)', 'C', 'C(i-1)']
        columns = self.dataframe_peaklist.columns[1:]
        atoms = []
        indexes = []

        weights = {'CA':1, 'CA(i-1)':1, 'CB':1, 'CB(i-1)':1, 'N':1, 'C':1,'C(i-1)':1,'CO':1,'CO(i-1)':1, 'H':0.5}

        for j, checkbox in enumerate(self.column_checkboxes):
            if(checkbox==True):
                atoms.append(columns[j])
                indexes.append(j+1)

        
        

        dictionaries = {}
        for atom in atoms:
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

        # peak_residue_names = self.dataframe_peaklist[self.dataframe_peaklist.columns[0]].to_list()
        lists = []
        for atom in atoms:
            lists.append(list(dictionaries[atom].keys()))
        
        list_of_residues = []
        for k,list_ in enumerate(lists):
            list_of_residues = set(list_+list(list_of_residues))


        for i, residue in enumerate(list_of_residues):
            kdes[residue] = {}
            for atom in atoms:
                try:
                    vals = dictionaries[atom][residue]
                    kde = gaussian_kde(vals)
                    kdes[residue][atom] = kde
                except:
                    pass
            

        
        classes = list(dictionaries[atoms[1]].keys())
        priors = {cls: 1 / len(classes) for cls in classes}
        

        def probability_calc(shifts):
            log_scores = {}
            counts = {}

            for cls in classes:
                try:
                    probability = np.log(priors[cls])
                    count=0
                    for i, atom in enumerate(atoms):
                        if(shifts[i]==999):
                            # No chemical shift inputted
                            continue
                        else:
                            p = kdes[cls][atom](shifts[i])[0]
                            probability = probability+np.log(p)*weights[atom]
                            count+=1


                    log_scores[cls] = probability
                    counts = count
                except:
                    pass

            # Convert to stable probabilities
            max_val = max(log_scores.values())
            exp_scores = {c: np.exp(v - max_val) for c, v in log_scores.items()}
            total = sum(exp_scores.values())

            posteriors = {c: v / total for c, v in exp_scores.items()}

            # Absolute evidence in original scale
            log_evidence = max_val + np.log(total)


            return posteriors, log_evidence, counts
        

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))


        peaks = []
        peak_names = []
        for row in st.session_state.df_peaklist.iter_rows():
            peak_names.append(row[0])
            peak_values = []
            for k in indexes:
                peak_values.append(row[k])
            peaks.append(peak_values)

        probabilities = {}
        confidence = {}
        counts = {}
        posterior_max = []
        total_log_evidence = []
        for i, peak in enumerate(peaks):
            probabilities[peak_names[i]], log_evidence, counts[peak_names[i]] = probability_calc(peak)
            total_log_evidence.append(log_evidence)
            posterior_max.append(max(list(probabilities[peak_names[i]].values())))
   


        mu = -16.156413504880273
        sigma = 43.512603134386026
        for i, peak in enumerate(peaks):
            # confidence[peak_names[i]] = sigmoid((total_log_evidence[i] - np.mean(total_log_evidence))/np.std(total_log_evidence))*posterior_max[i]
            confidence[peak_names[i]] = sigmoid((total_log_evidence[i] - mu)/sigma)*posterior_max[i]*counts[peak_names[i]]/(counts[peak_names[i]]+6)*2
        


        st.session_state.possible_assignments = {}
        st.session_state.assignment_confidence = {}
        for peak_name in peak_names:

            probs = probabilities[peak_name]
            possibilities = []
            for key, val in probs.items():
                if(val>0.01):
                    possibilities.append([key,val])
            
            if(confidence[peak_name] < 0.2):
                conf=' (low confidence)'    
            elif(confidence[peak_name] < 0.4):
                conf=' (medium confidence)'
            else:
                conf=' (high confidence)'
            
            st.session_state.possible_assignments[peak_name+conf] = possibilities
            st.session_state.assignment_confidence[peak_name+conf] = confidence[peak_name]



        



    def plot_predicted_assignments(self):
        """
        Create the ability where a user can choose the peak and then the
        predicted assignments and their respective likelihoods are plotted
        in a bar chart
        """

        # container = st.container()

        st.subheader("Predicted assignments")
        keys = list(st.session_state.possible_assignments.keys())
        order = {"high confidence": 0, "medium confidence": 1, "low confidence": 2}
        try:
            sorted_values = sorted(keys, key=lambda x: order[x.split('(')[1].split(')')[0]])
            keys = sorted_values
        except:
            pass
        self.peak_choice = st.selectbox("Select peak:", keys)
        names = []
        likelihoods = []
        for d in st.session_state.possible_assignments[self.peak_choice]:
            names.append(d[0] + self.aa_sequence[int(d[0])-1])
            likelihoods.append(d[1])


        confidence = st.session_state.assignment_confidence[self.peak_choice]
        if(confidence < 0.2):
            color = '#cb4f33'
        elif(confidence < 0.4):
            color = '#e1a558'
        else:
            color = '#74c1ae'


        fig = go.Figure(data=[go.Bar(x=names,y=likelihoods, marker={'color': color})])
        fig.update_layout(xaxis = dict(title='Residue'))
        fig.update_layout(yaxis = dict(title='Normalised Likelihood', range=[0,1]))
        fig.update_layout(title = dict(text='Confidence: {:.3f}'.format(confidence)))
            
        st.plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"format": "svg","height": 600,"width": 800,"scale": 1}})

        st.markdown('These results should be used only as a guide to aid assignment and should be combined with other methods for validation. Peaks can sometimes be outside of the distribution of disordered residues in the BMRB.')
        st.markdown('The confidence score (C) is a metric of the confidence in the likelihood values. Values far out of the predicted distributions or with low maximum posterior probabilities are given a low confidence score.')
        st.markdown('C>0.4 (higher confidence), 0.2<C<0.4 (medium condifence), C<0.2 (lower confidence)')
            
            

    def perform_checks(self):
        """
        Perform the following checks before running BindPredictDCS
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