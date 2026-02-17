import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Union, Tuple, Optional
from numpy.typing import NDArray
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import leastsq

st.set_page_config(layout="wide")


colors = px.colors.qualitative.Alphabet

residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP','TYR','VAL']
    

class bmrb_dashboard():
    """
    This class will produce the dashboard to show the data
    """
    def __init__(self, data_disordered, data_all, data_structured, tab):
        self.residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP','TYR','VAL']
        self.data_disordered = data_disordered
        self.data_all = data_all
        self.data_structured = data_structured
        self.tab = tab
        self.potenci_shifts = self.define_POTENCI_shifts()

        self.atom_names = self.get_atom_names(data_all)
        self.sample_state_options = self.get_sample_states(data_all)

        self.organism_superkingdoms = self.get_superkingdoms(data_all)
        self.organism_common_names = self.get_common_names(data_all)

        self.create_dashboard()
        dataframe, fig = self.plot_data()
        fig_aa = self.plot_amino_acid(dataframe)
        
        
        # Adding the figure plot
        with self.tab:   
            col1, col2 = st.columns([3,1], vertical_alignment="center")
            col1.plotly_chart(fig, use_container_width=True)        
            col2.plotly_chart(fig_aa, config = {'displayModeBar': False}, use_container_width=True)

            # Add referencing information to the BMRB
            self.add_download_button(dataframe)
            st.text("Chemical shift source: Biomolecular Magnetic Resonance Data Bank (BMRB) - September 2025")
            st.markdown("https://doi.org/10.1093/nar/gkac1050")
            st.markdown("https://bmrb.io")

    @st.cache_data
    def get_atom_names(_self, data_all):
        return sorted(data_all['atom'].unique().to_list())
    
    @st.cache_data
    def get_sample_states(_self, data_all):
        states = data_all['sample state'].unique().to_list()
        # Move solution and solid to the start of the list
        states.remove('solution')
        states.remove('solid')
        states = ['solution', 'solid'] + states
        return states
    
    @st.cache_data
    def get_superkingdoms(_self, data_all):
        superkingdoms = data_all['organism superkingdom'].unique().to_list()
        # Move All, Eukaryota, Prokaryota to the start of the list
        superkingdoms.remove('eukaryote')
        superkingdoms.remove('prokaryote')
        superkingdoms = ['All', 'eukaryote', 'prokaryote'] + superkingdoms
        return superkingdoms
    
    @st.cache_data
    def get_common_names(_self, data_all):
        common_names = data_all['organism common name'].unique().to_list()
        # Move All, Eukaryota, Prokaryota to the start of the list
        common_names.remove('human')
        common_names = ['All', 'human', 'non-human'] + common_names
        return common_names
        

        
    def add_download_button(self, dataframe):
        """
        Create a download button if the dataframe contains data.
        """
        if(type(dataframe)!=None):
            st.download_button(
                label="Download Plot Data",
                data=self.convert_for_download(dataframe),
                file_name="dashboard_data.csv",
                mime="text/csv",
                icon=":material/download:",
            )



    def define_POTENCI_shifts(self):
        """
        Returns a dictionary of POTENCI central shifts obtained from
        https://github.com/protein-nmr/POTENCI/blob/master/potenci1_3.py
        on 06/10/2025.
        """
        tablecent='''aa C CA CB N H HA HB
ALA 177.44069  52.53002  19.21113 125.40155   8.20964   4.25629   1.31544
CYS 174.33917  58.48976  28.06269 120.71212   8.29429   4.44261   2.85425
ASP 176.02114  54.23920  41.18408 121.75726   8.28460   4.54836   2.60054
GLU 176.19215  56.50755  30.30204 122.31578   8.35949   4.22124   1.92383
PHE 175.42280  57.64849  39.55984 121.30500   8.10906   4.57507   3.00036
GLY 173.83294  45.23929  None     110.09074   8.32746   3.91016   None
HIS 175.00142  56.20256  30.60335 120.69141   8.27133   4.55872   3.03080
ILE 175.88231  61.04925  38.68742 122.37586   8.06407   4.10574   1.78617
LYS 176.22644  56.29413  33.02478 122.71282   8.24902   4.25873   1.71982
LEU 177.06101  55.17464  42.29215 123.48611   8.14330   4.28545   1.54067
MET 175.90708  55.50643  32.83806 121.54592   8.24848   4.41483   1.97585
ASN 174.94152  53.22822  38.87465 119.92746   8.37189   4.64308   2.72756
PRO 176.67709  63.05232  32.03750 137.40612   None      4.36183   2.03318
GLN 175.63494  55.79861  29.44174 121.49225   8.30042   4.28006   1.97653
ARG 175.92194  56.06785  30.81298 122.40365   8.26453   4.28372   1.73437
SER 174.31005  58.36048  63.82367 117.11419   8.25730   4.40101   3.80956
THR 174.27772  61.86928  69.80612 115.48126   8.11378   4.28923   4.15465
VAL 175.80621  62.20156  32.77934 121.71912   8.06572   4.05841   1.99302
TRP 175.92744  57.23836  29.56502 122.10991   7.97816   4.61061   3.18540
TYR 175.49651  57.82427  38.76184 121.43652   8.05749   4.51123   2.91782'''

        potenci_dict = {}
        for i, line in enumerate(tablecent.split('\n')):
            if(i==0):
                continue
            line = line.split()
            potenci_dict[line[0]] = {}
            potenci_dict[line[0]]['C'] = float(line[1])
            potenci_dict[line[0]]['CA'] = float(line[2])
            try:
                potenci_dict[line[0]]['CB'] = float(line[3])
            except:
                potenci_dict[line[0]]['CB'] = None
            potenci_dict[line[0]]['N'] = float(line[4])
            try:
                potenci_dict[line[0]]['H'] = float(line[5])
            except:
                potenci_dict[line[0]]['H'] = None
            try:
                potenci_dict[line[0]]['HA'] = float(line[6])
            except:
                potenci_dict[line[0]]['HA'] = None
            try:
                potenci_dict[line[0]]['HB'] = float(line[7])
            except:
                potenci_dict[line[0]]['HB'] = None
            
        return potenci_dict


    

    def create_dashboard(self):
        """
        Create all the options that the user can interact with on the
        dashboard.
        """
        with self.tab:
            st.title("Chemical Shift Data Dashboard")

            # Add logo to the sidebar
            st.logo('./bind-logo-alpha.svg', size="large", link="https://bindresearch.org", icon_image='./bind-logo-alpha.svg')

            # Creating a subsection to select data to plot
            st.sidebar.subheader("Data selection:")
            options = ['Individual amino acid/atom', 'Individual amino acid/atom 2D', 'Individual amino acid/atom 3D', 'All amino acids for selected atom', 'All amino acids for selected atom 2D', 'All amino acids for selected atom 3D','All atoms for selected amino acid']
            self.overlays = st.sidebar.selectbox(label = '', options=options)
            
            if(self.overlays == 'Individual amino acid/atom'):
                self.residue = st.sidebar.selectbox("Select Amino Acid:", self.residues)
                atom_list_jumbled = self.data_disordered.filter(pl.col('residue')==self.residue)["atom"].unique().to_list()
                atom_list = []
                for name in self.atom_names:
                    if(name in atom_list_jumbled):
                        atom_list.append(name)
                self.atom = st.sidebar.selectbox("Select Atom:", atom_list)
            elif(self.overlays == "All atoms for selected amino acid"):
                self.residue = st.sidebar.selectbox("Select Amino Acid:", self.residues)
            elif(self.overlays == 'All amino acids for selected atom'):
                self.atom = st.sidebar.selectbox("Select Atom:", self.residues)
            elif(self.overlays == 'Individual amino acid/atom 2D'):
                self.residue = st.sidebar.selectbox("Select Amino Acid:", self.residues)
                self.atom_2D = st.sidebar.selectbox("Select Atom 2D:", ['H-N', 'HA-CA', 'C-N', 'CA-N', 'HB-CB', 'CA-CB', 'HA-N', 'HG-CG', 'HD-CD','HA-C', 'H-C', 'H-C (HA-CA + HB-CB + HG-CG + HD-CD)'])
            elif(self.overlays == 'Individual amino acid/atom 3D'):
                self.residue = st.sidebar.selectbox("Select Amino Acid:", self.residues)
                self.atom_3D = st.sidebar.selectbox("Select Atom 3D:", ['HA-CA-N', 'HA-CA-C', 'H-N-CA'])
            elif(self.overlays == 'All amino acids for selected atom 3D'):
                self.residue = st.sidebar.selectbox("Select Amino Acid:", self.residues)
                self.atom_3D = st.sidebar.selectbox("Select Atom 3D:", ['HA-CA-N', 'HA-CA-C', 'H-N-CA'])
            else:
                self.atom_2D = st.sidebar.selectbox("Select Atom 2D:", ['H-N', 'HA-CA', 'C-N', 'CA-N', 'HB-CB', 'CA-CB', 'HA-N', 'HG-CG', 'HD-CD', 'HA-C', 'H-C', 'H-C (HA-CA + HB-CB + HG-CG + HD-CD)'])

            # Creating a subsection to select which residues to plot (disordered or all residues)
            st.sidebar.subheader("Residues to plot:")
            help_disordered = """Disordered residues are those which are in regions of a protein with an\nAlphaFold2 pLDDT score < 70 for 30+ continuous residues."""
            self.show_disordered = st.sidebar.checkbox("Disordered residues", value=True, help=help_disordered)
            self.show_all = st.sidebar.checkbox("All residues")
            help_structured = """Structured residues are residues not classed as disordered by our definition and excludes residues from proteins described as disordered by the author in the BMRB entry (under physical state). Residues proteins classified as having a denatured physical state are also omitted."""
            self.show_structured = st.sidebar.checkbox("Structured residues", help=help_structured)


            # Creating a subsection to filter by following/preceding residue
            st.sidebar.subheader("Preceding/following residue:")
            self.filter_by_preceding = st.sidebar.checkbox("Filter by preceding residue")


            if(self.filter_by_preceding):
                self.preceding_residue = st.sidebar.selectbox("Preceding residue:", self.residues)

                if(self.overlays == 'Individual amino acid/atom' or self.overlays == 'Individual amino acid/atom 2D'):
                    show_array = [self.show_disordered, self.show_all, self.show_structured]
                    if(show_array.count(True)==1):
                        self.overlay_preceding = st.sidebar.checkbox("Overlay all preceding residues")

            
            self.filter_by_following = st.sidebar.checkbox("Filter by following residue")

            if(self.filter_by_following):
                self.preceding_residue = st.sidebar.selectbox("Following residue:", self.residues)

                if(self.overlays == 'Individual amino acid/atom' or self.overlays == 'Individual amino acid/atom 2D'):
                    show_array = [self.show_disordered, self.show_all, self.show_structured]
                    if(show_array.count(True)==1):
                        self.overlay_following = st.sidebar.checkbox("Overlay all following residues")

            # Creating a subsection to select which conditions to screen through
            st.sidebar.subheader("Experimental conditions:")

            self.filter_by_sample_state = st.sidebar.checkbox("Filter by sample state", help="e.g. solution, solid, gel etc.")
            if(self.filter_by_sample_state):
                self.sample_state = st.sidebar.selectbox(label='Sample state:', options=self.sample_state_options)
            
            self.filter_by_temperature = st.sidebar.checkbox("Filter by temperature")

            if(self.filter_by_temperature):
                st.sidebar.markdown("Temperature range (K):")
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    self.tmin = st.text_input("Min", value = 273, key='temp min')
                with col2:
                    self.tmax = st.text_input("Max", value = 373, key='temp max')

            self.filter_by_pH = st.sidebar.checkbox("Filter by pH")
            if(self.filter_by_pH):
                st.sidebar.markdown("pH range:")
                col1_pH, col2_pH = st.sidebar.columns(2)
                with col1_pH:
                    self.pH_min = st.text_input("Min", value = 0, key='pH min')
                with col2_pH:
                    self.pH_max = st.text_input("Max", value = 14, key='pH max')

            self.filter_by_pressure = st.sidebar.checkbox("Filter by pressure")
            if(self.filter_by_pressure):
                st.sidebar.markdown("Pressure range (atm):")
                col1_atm, col2_atm = st.sidebar.columns(2)
                with col1_atm:
                    self.pressure_min = st.text_input("Min", value = 0, key='pressure min')
                with col2_atm:
                    self.pressure_max = st.text_input("Max", value = 10, key='pressure max')

            self.filter_by_ion_strength = st.sidebar.checkbox("Filter by ion strength")
            if(self.filter_by_ion_strength):
                st.sidebar.markdown("Ionic strength range (M):")
                col1_M, col2_M = st.sidebar.columns(2)
                with col1_M:
                    self.ion_stength_min = st.text_input("Min", value = 0, key='ionic strength min')
                with col2_M:
                    self.ion_stength_max = st.text_input("Max", value = 5, key='ionic strength max')

            # Creating a section to filter by features of the organism such as organism superkingdom and common name
            st.sidebar.subheader("Organism information:")

            self.superkingdom = st.sidebar.selectbox("Superkingdom:", self.organism_superkingdoms)
            self.common_name = st.sidebar.selectbox("Common name:", self.organism_common_names)

            # Creating a subsection to update options for plotting such as histogram bin widths
            st.sidebar.subheader("Plot options:")
            if(self.overlays == 'Individual amino acid/atom 3D' or self.overlays == 'All amino acids for selected atom 3D'):
                self.bin_width = st.sidebar.text_input('Histogram x bin width (ppm)', value="0.1")
                self.bin_width2 = st.sidebar.text_input('Histogram y bin width (ppm)', value="1.0")
                self.bin_width3 = st.sidebar.text_input('Histogram z bin width (ppm)', value="1.0")
            elif(self.overlays == 'Individual amino acid/atom 2D' or self.overlays=='All amino acids for selected atom 2D'):
                if(self.atom_2D == 'H-N' or self.atom_2D == 'HA-CA' or self.atom_2D=='HB-CB' or self.atom_2D=='HG-CG' or self.atom_2D=='HD-CD' or self.atom_2D == 'H-C (HA-CA + HB-CB + HG-CG + HD-CD)'):
                    self.bin_width = st.sidebar.text_input('Histogram x bin width (ppm)', value="0.1")
                else:
                    self.bin_width = st.sidebar.text_input('Histogram x bin width (ppm)', value="1")
                
                self.bin_width2 = st.sidebar.text_input('Histogram y bin width (ppm)', value="1")
                
            else:
                self.bin_width = st.sidebar.text_input('Histogram bin width (ppm)', value="0.1")
            
            popup_text = """Available when Individual amino acid/atom is checked and one of Disordered residues/All residues/Structured residues is checked"""
            self.overlay_fit = st.sidebar.checkbox("Overlay fitted gaussian", help=popup_text)

            popup_text = """Available when Individual amino acid/atom is checked"""
            self.overlay_potenci = st.sidebar.checkbox("Overlay POTENCI random coil values", help=popup_text)




    
    def plot_amino_acid(self, dataframe):
        """
        Plot a general schematic for amino acid atoms and show in colour
        the current atoms being looked at.
        """

        fig_aa = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': False}]])

        atoms = dataframe['atom'].unique().to_list()

        try:
            atoms_to_include = self.data_all.filter(pl.col('residue'==self.residue))['atom'].unique()
        except:
            atoms_to_include = 'all'

        coordinate_dictionary_all = {'H':(-1,-1), 'N': (-1,0), 'HA': (0,-1), 'CA': (0,0), 'C': (1,0), 'CB': (0,1), 'HB': [(1,1), (-1,1)], 'CG': (0,2), 'HG': [(-1,2), (1,2)], 'CD': (0,3) , 'ND': (0,3), 'HD': [(1,3), (-1,3)], 'CE': (0,4), 'NE': (0,4), 'HE': [(1,4), (-1,4)], 'CZ': (0,5), 'NZ': (0,5), 'HZ': [(1,5), (-1,5)]}
        if('All amino acids' in self.overlays):
            coordinate_dictionary = {'H':(-1,-1), 'N': (-1,0), 'HA': (0,-1), 'CA': (0,0), 'C': (1,0), 'CB': (0,1), 'HB': [(1,1), (-1,1)], 'CG': (0,2), 'HG': [(-1,2), (1,2)], 'CD/ND': (0,3), 'HD': [(1,3), (-1,3)], 'CE/NE': (0,4), 'HE': [(1,4), (-1,4)], 'CZ/NZ': (0,5), 'HZ': [(1,5), (-1,5)]}
        else:
            atoms_to_include = self.data_all.filter(pl.col('residue')==self.residue)['atom'].unique()
            coordinate_dictionary = {}
            for atom in atoms_to_include:
                try:
                    coordinate_dictionary[atom] = coordinate_dictionary_all[atom]
                except:
                    pass

        
        # coordinate_dictionary = {'H':(-1,-1), 'N': (-1,0), 'HA': (0,-1), 'CA': (0,0), 'C': (1,0), 'CB': (0,1), 'HB': [(1,1), (-1,1)], 'CG': (0,2), 'HG': [(-1,2), (1,2)], 'CD': (0,3), 'HD': [(1,3), (-1,3)], 'CE': (0,4), 'HE': [(1,4), (-1,4)], 'CZ': (0,5), 'HZ': [(1,5), (-1,5)]}


        bonds_to_plot = []

        def parse_item(item):
            if(len(item)==1):
                return item, None

            else:
                prefix = item[0]
                suffix = item[-1]
            try:
                if suffix.isdigit():
                    number = int(suffix)
                else:
                    # assume it's letters a,b,c… and map to ordinal
                    number = ord(suffix) - ord('A') + 1

                return prefix, number
            except:
                return prefix, None
        
        a = list(coordinate_dictionary.keys())

        for i in range(len(a)):
            p1, n1 = parse_item(a[i])
            if(n1!=None):
                for j in range(i+1, len(a)):
                    p2, n2 = parse_item(a[j])
                    if(n2!=None):
                        if n1 == n2:
                            bonds_to_plot.append([a[i], a[j]])
                        # connect with +/-1
                        elif(p1[0] != 'H' and p2[0] != 'H' and abs(n1 - n2) == 1 and p1[0]):
                            if(a[i]!='C' and a[j]!='C' and a[i]!='N' and a[j]!='N'):
                                bonds_to_plot.append([a[i], a[j]])
                
        bonds_to_plot = [['H','N'], ['N','CA'], ['CA', 'C']] + bonds_to_plot


        for bond in bonds_to_plot:
            coords1 = coordinate_dictionary[bond[0]]
            coords2 = coordinate_dictionary[bond[1]]
            if(type(coords1)==list):
                for coord1 in coords1:
                    fig_aa.add_trace(go.Scatter(x=[coord1[0], coords2[0]], y=[coord1[1], coords2[1]], mode='lines',line_color='gray', text='', showlegend=False, hoverinfo="skip"), row=1, col=1)
            elif(type(coords2)==list):
                for coord2 in coords2:
                    fig_aa.add_trace(go.Scatter(x=[coords1[0], coord2[0]], y=[coords1[1], coord2[1]], mode='lines', line_color='gray', text='', showlegend=False, hoverinfo="skip"), row=1, col=1)
            else:
                fig_aa.add_trace(go.Scatter(x=[coords1[0], coords2[0]], y=[coords1[1], coords2[1]], mode='lines', line_color='gray', text='', showlegend=False, hoverinfo="skip"), row=1, col=1)
            
                

        # Add in wildcard bonds
        fig_aa.add_trace(go.Scatter(x=[-2,-1], y=[0, 0], mode='lines', line_color='gray', text='', showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig_aa.add_trace(go.Scatter(x=[1,2], y=[0, 0], mode='lines', line_color='gray', text='', showlegend=False, hoverinfo="skip"), row=1, col=1)



        for atom, coords in coordinate_dictionary.items():
            if(atom in atoms):
                color = "#F5979A"
            else:
                color = 'lightgray'
            if(atom=='HB' or atom=='HG' or atom == 'HD' or atom == 'HE' or atom == 'HZ'):
                for coord in coords:
                    fig_aa.add_trace(go.Scatter(
                    x=[coord[0]], y=[coord[1]],
                    mode='markers',
                    marker = dict(size=25,color=color),
                    showlegend=False,
                    text='',
                    hoverinfo="skip"
                        ))
            else:
                fig_aa.add_trace(go.Scatter(
                    x=[coords[0]], y=[coords[1]],
                    mode='markers',
                    marker = dict(size=25,color=color),
                    showlegend=False,
                    text='',
                    hoverinfo="skip"
                ))

        

                # Add the atom names to each coordinate
        x = []
        y = []
        text = []

        for atom, coords in coordinate_dictionary.items():
            if(atom=='HB' or atom=='HG' or atom == 'HD' or atom == 'HE' or atom == 'HZ'):
                for coord in coords:
                    x.append(coord[0])
                    y.append(coord[1])
                    text.append(atom)
            else:
                x.append(coords[0])
                y.append(coords[1])
                text.append(atom)

        
        fig_aa.add_trace(go.Scatter(
            x=x,
            y=y,
            text=text,
            mode='text',
            showlegend=False,
            zorder=1,
            hoverinfo="skip"
        ), row=1, col=1)


        fig_aa.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig_aa.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

        return fig_aa
    
    
    def get_filtered_data(self, structure: str, atom: str = '', residue: str = ''):
        """
        Filter the data in the dataframe by the current user selected options.
        
        Parameters
        ----------
        structure : str
                    Whether to select for disordered residues (disordered), all residues (all) or
                    structured residues (structured)
        atom : str
               Filter by atom type (e.g. H or CA)
        residue : str
                  Filter by residue type (e.g. ARG or LYS)

        Returns
        -------
        dataframe : pd.DataFrame
                    Dataframe of filtered data to plot
        """

        if(structure == 'disordered'):
            dataframe = self.data_disordered
        elif(structure == 'all'):
            dataframe = self.data_all
        else:
            dataframe = self.data_structured
        
        if(atom!=''):
            dataframe = dataframe.filter(pl.col('atom')==atom)

        if(residue!=''):
            dataframe = dataframe.filter(pl.col('residue')==residue)

        dataframe = self.filter_data_by_condition(dataframe)
        dataframe = self.filter_by_organism(dataframe)
        if(self.filter_by_preceding):
            if(self.overlays == 'Individual amino acid/atom' or self.overlays == 'Individual amino acid/atom 2D'):
                if(self.overlay_preceding!=True):
                    dataframe = self.filter_data_by_neighboring_residues(dataframe, preceding_residue=self.preceding_residue)
        if(self.filter_by_following):
            if(self.overlays == 'Individual amino acid/atom' or self.overlays == 'Individual amino acid/atom 2D'):
                if(self.overlay_following!=True):
                    dataframe = self.filter_data_by_neighboring_residues(dataframe, following_residue=self.preceding_residue)

        return dataframe
    

    def filter_data_by_neighboring_residues(self, dataframe, preceding_residue: str = '', following_residue: str = ''):
        """
        Filter the dataframe by preceding residue and following residue.

        e.g. show all ALA residues that are followed by PRO etc.
        """
        
        if(preceding_residue != ''):
            dataframe = dataframe.filter(pl.col('preceding residue type')==preceding_residue)
        if(following_residue != ''):
            dataframe = dataframe.filter(pl.col('following residue type')==following_residue)

        return dataframe
        

    def filter_data_by_condition(self, dataframe):
        """
        Filter the data by condition such as temperature, pH, ionic strength
        and pressure and return the updated dataframe.
        """
        if(self.filter_by_sample_state == True):
            dataframe = dataframe.filter(pl.col('sample state')==self.sample_state)
        if(self.filter_by_temperature):
            dataframe = dataframe.filter(pl.col('temperature (K)').cast(pl.Float64, strict=False)>=float(self.tmin))
            dataframe = dataframe.filter(pl.col('temperature (K)').cast(pl.Float64, strict=False)<=float(self.tmax))
        if(self.filter_by_pH):
            dataframe = dataframe.filter(pl.col('pH').cast(pl.Float64, strict=False)>=float(self.pH_min))
            dataframe = dataframe.filter(pl.col('pH').cast(pl.Float64, strict=False)<=float(self.pH_max))
        if(self.filter_by_pressure):
            dataframe = dataframe.filter(pl.col('pressure (atm)').cast(pl.Float64, strict=False)>=float(self.pressure_min))
            dataframe = dataframe.filter(pl.col('pressure (atm)').cast(pl.Float64, strict=False)<=float(self.pressure_max))
        if(self.filter_by_ion_strength):
            dataframe = dataframe.filter(pl.col('ionic strength (M)').cast(pl.Float64, strict=False)>=float(self.ion_stength_min))
            dataframe = dataframe.filter(pl.col('ionic strength (M)').cast(pl.Float64, strict=False)<=float(self.ion_stength_max))

        return dataframe
    
    def filter_by_organism(self, dataframe):
        """
        Filter the data by organism of origin features such as
        superkingdom (Eukaryota, Prokaryota), organism common name etc
        """
        if(self.superkingdom=='All' and self.common_name=='All'):
            return dataframe
        
        if(self.superkingdom == 'All'):
            if(self.common_name!='non-human'):
                dataframe = dataframe.filter(pl.col('organism common name')==self.common_name)
            else:
                dataframe = dataframe.filter(pl.col('organism common name')!='human')

        else:
            dataframe = dataframe.filter(pl.col('organism superkingdom')==self.superkingdom)

        return dataframe


    def plot_data(self):
        """
        Go through the dataframe and extract the relevent data
        for the current user-selected option.
        This dataframe is then plotted as a histogram or a set
        of histograms.
        """


        if(self.overlays == 'Individual amino acid/atom 2D' or self.overlays == 'All amino acids for selected atom 2D'):
            dataframe, fig = self.plot2D()
        elif(self.overlays == 'Individual amino acid/atom 3D' or self.overlays == 'All amino acids for selected atom 3D'):
            dataframe, fig = self.plot3D()
        else:
            dataframe, fig = self.plot1D()
            
        
        return dataframe, fig
    


    def plot3D(self):
        """
        Determine which 3D data to plot in the 3D histogram and
        then plot this.
        """

        if(self.show_disordered and not self.show_all and not self.show_structured):
            structure = 'disordered'
        elif(self.show_all and not self.show_disordered and not self.show_structured):
            structure = 'all'
        elif(self.show_structured and not self.show_disordered and not self.show_all):
            structure = 'structured'
        else:
            return None, None

        if(self.overlays == 'Individual amino acid/atom 3D'):
            identifier = 'residue'
            if(self.filter_by_preceding):
                if(self.overlay_preceding==True):
                    identifier = 'preceding residue type'
            if(self.filter_by_following):
                if(self.overlay_following==True):
                    identifier = 'following residue type'
            fig = go.Figure()
            atom1, atom2, atom3 = self.atom_3D.split('-')
            dataframe1 = self.get_filtered_data(structure = structure, atom = atom1, residue = self.residue)
            dataframe2 = self.get_filtered_data(structure = structure, atom = atom2, residue = self.residue)
            dataframe3 = self.get_filtered_data(structure = structure, atom = atom3, residue = self.residue)
            dataframe = pl.concat([dataframe1, dataframe2, dataframe3], how='vertical')
            if(identifier == 'residue'):
                values = [self.residue]
            else:
                values = residues
            fig = self.plot_3D_histogram(fig, dataframe, atom_pairs=[self.atom_3D], values = values, filter_type = identifier, plot_scatter=False)

        else:
            atom1, atom2, atom3 = self.atom_3D.split('-')
            dataframe1 = self.get_filtered_data(structure = structure, atom = atom1, residue = '')
            dataframe2 = self.get_filtered_data(structure = structure, atom = atom2, residue = '')
            dataframe3 = self.get_filtered_data(structure = structure, atom = atom3, residue = '')
            dataframe = pl.concat([dataframe1, dataframe2, dataframe3], how='vertical')
            values = self.residues
            fig = go.Figure()
            fig = self.plot_3D_histogram(fig, dataframe, atom_pairs=[self.atom_3D], values=values, filter_type = 'residue', plot_scatter=False)


        return dataframe, fig

    def plot2D(self):
        """
        Determine which 2D data to plot in the 2D histogram and
        then plot this.
        """

        if(self.show_disordered and not self.show_all and not self.show_structured):
            structure = 'disordered'
        elif(self.show_all and not self.show_disordered and not self.show_structured):
            structure = 'all'
        elif(self.show_structured and not self.show_disordered and not self.show_all):
            structure = 'structured'
        else:
            return None, None

        if(self.overlays == 'Individual amino acid/atom 2D'):
            identifier = 'residue'
            if(self.filter_by_preceding):
                if(self.overlay_preceding==True):
                    identifier = 'preceding residue type'
            if(self.filter_by_following):
                if(self.overlay_following==True):
                    identifier = 'following residue type'
            fig = go.Figure()
            if(self.atom_2D != 'H-C (HA-CA + HB-CB + HG-CG + HD-CD)'):
                atom1, atom2 = self.atom_2D.split('-')
                dataframe1 = self.get_filtered_data(structure = structure, atom = atom1, residue = self.residue)
                dataframe2 = self.get_filtered_data(structure = structure, atom = atom2, residue = self.residue)
                dataframe = pl.concat([dataframe1, dataframe2], how='vertical')
                if(identifier == 'residue'):
                    values = [self.residue]
                else:
                    values = residues
                fig = self.plot_2D_histogram(fig, dataframe, atom_pairs=[self.atom_2D], values = values, filter_type = identifier, plot_scatter=True)
            else:
                for k, pair in enumerate(['HA-CA', 'HB-CB', 'HG-CG', 'HD-CD']):
                    atom1, atom2 = pair.split('-')
                    dataframe1 = self.get_filtered_data(structure = structure, atom = atom1, residue = self.residue)
                    dataframe2 = self.get_filtered_data(structure = structure, atom = atom2, residue = self.residue)
                    dataframe = pl.concat([dataframe1, dataframe2], how='vertical')
                    if(k==0):
                        df_total = dataframe 
                    else:
                        df_total = pl.concat([df_total, dataframe])
                
                if(identifier == 'residue'):
                    values = [self.residue]
                else:
                    values = residues
                
                fig = self.plot_2D_histogram(fig, df_total, atom_pairs=['HA-CA', 'HB-CB', 'HG-CG', 'HD-CD'], values = values, filter_type = identifier, plot_scatter=True)

        else:
            if(self.atom_2D != 'H-C (HA-CA + HB-CB + HG-CG + HD-CD)'):
                atom1, atom2 = self.atom_2D.split('-')
                dataframe1 = self.get_filtered_data(structure = structure, atom = atom1, residue = '')
                dataframe2 = self.get_filtered_data(structure = structure, atom = atom2, residue = '')
                dataframe = pl.concat([dataframe1, dataframe2], how='vertical')
                values = self.residues
                fig = go.Figure()
                fig = self.plot_2D_histogram(fig, dataframe, atom_pairs=[self.atom_2D], values=values, filter_type = 'residue', plot_scatter=True)
            else:
                for k, pair in enumerate(['HA-CA', 'HB-CB', 'HG-CG', 'HD-CD']):
                    atom1, atom2 = pair.split('-')
                    dataframe1 = self.get_filtered_data(structure = structure, atom = atom1, residue = '')
                    dataframe2 = self.get_filtered_data(structure = structure, atom = atom2, residue = '')
                    dataframe = pl.concat([dataframe1, dataframe2], how='vertical')
                    if(k==0):
                        df_total = dataframe 
                    else:
                        df_total = pl.concat([df_total, dataframe])
                
                values = self.residues
                fig = go.Figure()
                fig = self.plot_2D_histogram(fig, df_total, atom_pairs=['HA-CA', 'HB-CB', 'HG-CG', 'HD-CD'], values = values, filter_type = 'residue', plot_scatter=True)


        return dataframe, fig
    
    def plot1D(self):
        """
        Determine which data to plot in the histogram and
        then plot this.
        """

        if(self.show_disordered and not self.show_all and not self.show_structured):
            structure = 'disordered'
        elif(self.show_all and not self.show_disordered and not self.show_structured):
            structure = 'all'
        elif(self.show_structured and not self.show_disordered and not self.show_all):
            structure = 'structured'
        else:
            dataframe, fig = self.plot_overlay()
            return dataframe, fig

        if(self.overlays == 'All amino acids for selected atom'):
            dataframe = self.get_filtered_data(structure=structure, atom=self.atom)
            title = f"Overlay Histogram across amino acids (atom={self.atom})"
            residues = dataframe["residue"].unique().to_numpy()
            identifier = "residue"
            values = residues
            color_map = {g: colors[i % len(colors)] for i, g in enumerate(residues)}
        elif(self.overlays == 'All atoms for selected amino acid'):
            dataframe = self.get_filtered_data(structure = structure, atom = '', residue = self.residue)
            title = f"Overlay Histogram across atoms (amino acid={self.residue})"
            atoms = dataframe["atom"].unique().to_numpy()
            identifier = "atom"
            values = atoms
            color_map = {g: colors[i % len(colors)] for i, g in enumerate(atoms)}
        else:
            dataframe = self.get_filtered_data(structure = structure, atom = self.atom, residue = self.residue)
            title=f"Normalized Histogram for {self.residue}, {self.atom}"
            identifier = 'residue'
            if(self.filter_by_preceding):
                if(self.overlay_preceding==True):
                    identifier = 'preceding residue type'
            if(self.filter_by_following):
                if(self.overlay_following==True):
                    identifier = 'following residue type'

            values = dataframe[identifier].unique().to_numpy()
            for i, value in enumerate(values):
                if(value not in self.residues):
                    np.delete(values,i)
            color_map = {g: colors[i % len(colors)] for i, g in enumerate(values)}
        
        fig = self.plot_histogram(dataframe, title, color=color_map, identifier = identifier, values=values)

        return dataframe, fig
    
    def plot_overlay(self):
        """
        Plot overlays of disordered/all-residues/structured-residues
        chemical shifts as a histogram.
        """
        dataframe = pl.DataFrame()
        if(self.overlays == 'All amino acids for selected atom'):
            dataframe_all = self.get_filtered_data(structure='all', atom=self.atom)
            dataframe_dis = self.get_filtered_data(structure='disordered', atom=self.atom)
            dataframe_struct = self.get_filtered_data(structure='structured', atom=self.atom)
            df_combined = pl.concat([dataframe_all, dataframe_dis, dataframe_struct], how='vertical')
            df_combined["Group"] = df_combined["Dataset"] + ": amino acid=" + df_combined["residue"]
            residues = df_combined["residue"].unique().to_numpy()
            color_map = {g: colors[i % len(colors)] for i, g in enumerate(residues)}
            flag='residue'
            
        elif(self.overlays == 'All atoms for selected amino acid'):
            dataframe_all = self.get_filtered_data(structure='all', residue=self.residue)
            dataframe_dis = self.get_filtered_data(structure='disordered', residue=self.residue)
            dataframe_struct = self.get_filtered_data(structure='structured', residue=self.residue)
            df_combined = pl.concat([dataframe_all, dataframe_dis, dataframe_struct], how='vertical')
            df_combined["Group"] = df_combined["Dataset"] + ": atom=" + df_combined["atom"]
            atoms = df_combined["atom"].unique().to_numpy()
            color_map = {g: colors[i % len(colors)] for i, g in enumerate(atoms)}
            flag = 'atom'
        
        elif(self.overlays == 'Individual amino acid/atom'):
            dataframe_all = self.get_filtered_data(structure='all', atom=self.atom, residue=self.residue)
            dataframe_dis = self.get_filtered_data(structure='disordered', atom=self.atom, residue=self.residue)
            dataframe_struct = self.get_filtered_data(structure='structured', atom=self.atom, residue=self.residue)
            df_combined = pl.concat([dataframe_all, dataframe_dis, dataframe_struct], how='vertical')
            df_combined = df_combined.with_columns(pl.lit(df_combined["Dataset"] + ": amino acid=" + df_combined["residue"]).alias("Group"))
            values = df_combined["residue"].unique().to_numpy()
            color_map = {g: colors[i % len(colors)] for i, g in enumerate(values)}
            flag='residue'

        residues_to_plot = []
        if(self.show_disordered):
            residues_to_plot.append('Disordered residues')
        if(self.show_all):
            residues_to_plot.append('All residues')
        if(self.show_structured):
            residues_to_plot.append('Structured residues')
        
        fig = self.plot_histogram_all(df_combined, residues_to_plot, colormap=color_map, flag=flag)

        return df_combined, fig
    

    def filter_atoms_2d(self, dataframe, atom1, atom2):
        """
        Filter the dataframe to 2D chemical shifts from atom1 and atom2 that come
        from the same residue in the same protein and bmrb id for the same conditions.

        Return the x and y values for chemical shifts of atom1 and atom2 
        """

        dataframe = (
                dataframe.group_by(["BMRB entry ID", "residue number", "entity id", "temperature (K)", 'physical state', 'pH', 'pressure (atm)', 'ionic strength (M)', 'preceding residue type', 'following residue type'])
                .agg([
                    pl.col('chemical shifts (ppm)').filter(pl.col('atom')==atom1).first().alias("xval"),
                    pl.col('chemical shifts (ppm)').filter(pl.col('atom')==atom2).first().alias("yval"),
                    pl.len().alias("group_size")
                ])
                .filter(pl.col("group_size") == 2)
            )

            
        x = dataframe["xval"].to_numpy()
        y = dataframe["yval"].to_numpy()

        return x, y
        
    
    def add_2d_contour_trace(self, fig, colormap, legendgroup, showlegend, x, y):
        """
        Add a 2D contour trace of the x,y chemical shift points
        """


        fig.add_trace(go.Histogram2dContour(
                x=x,
                y=y,
                contours=dict(coloring="lines"),
                colorscale=colormap,
                legendgroup=legendgroup,
                name=legendgroup,
                showscale=False,
                showlegend=showlegend,
                hoverinfo='skip',
                histnorm="probability density",
            ))

        
        fig.update_traces(xbins=dict(size=float(self.bin_width)))
        fig.update_traces(ybins=dict(size=float(self.bin_width2)))

        return fig
    

    def find_xmax_ymax(self, x, y):
        """
        For the x and y values which give the maximum of the histogram
        """
        x_edges = np.arange(x.min(), x.max() + float(self.bin_width), float(self.bin_width))
        y_edges = np.arange(y.min(), y.max() + float(self.bin_width2), float(self.bin_width2))
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=[x_edges, y_edges])
        max_idx = np.unravel_index(np.argmax(hist), hist.shape)
        # Get the center coordinates of that bin
        x_max = (x_edges[max_idx[0]] + x_edges[max_idx[0]+1]) / 2
        y_max = (y_edges[max_idx[1]] + y_edges[max_idx[1]+1]) / 2

        return x_max, y_max



    def plot_2D_histogram(self, fig: go.Figure, dataframe, atom_pairs: List, values: List, filter_type: str, plot_scatter: bool) -> go.Figure:
        """
        Plot a contour plot/2D histogram showing the data that is in each
        dataframe.

        Parameters
        ----------
        fig : go.Figure
              Plotly figure object
        dataframe : DataFrame
                    Dataframe of chemical shifts for both atoms in the atom
                    2D
        atom_pairs : List
                     Pairs of atoms to form 2D histograms together with
                     (e.g. [HA-CA] or [HA-CA, HB-CB, HG-CG, HD-CD])
        values : list
                list of residues to loop through to plot the overlaid 2D histogram
                chemical shift distributions for atom 1 and atom 2 in the atom 2D
        filter_type : str
                      An identifier for how to filter the data (i.e. by residue is default
                      but could also be preceding residue type or following residue type)
        plot_scatter : bool
                       A flag to toggle plotting a scatter plot over the 2D histograms or not.
        
        

        Returns
        -------
        fig : go.Figure
              Plotly figure object
        
        """

        colormaps = [['white', color] for color in colors]
        point_colors = []
        point_values = []
        scatter_plots = []
        
        
        for i, value in enumerate(values):
            df = dataframe.filter(pl.col(filter_type)==value)
            for k, pair in enumerate(atom_pairs):
                atom1, atom2 = pair.split('-')
                df1 = df.filter(pl.col('atom')==atom1)
                df2 = df.filter(pl.col('atom')==atom2)
                d = pl.concat([df1, df2], how='vertical')
                x, y = self.filter_atoms_2d(d, atom1, atom2)
                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]
                if(len(x)==0 or len(y)==0):
                    continue

                if(len(atom_pairs)>1):
                    legendgroup = value
                    if(k==0):
                        showlegend = True
                    else:
                        showlegend = False
                else:
                    legendgroup = value + ', {} shifts'.format(len(x))
                    showlegend= True

                fig = self.add_2d_contour_trace(fig, colormaps[i], legendgroup, showlegend, x, y)
                
                point_colors.append(colors[i])
                point_values.append(value)
                
                if(plot_scatter):
                    x_max, y_max = self.find_xmax_ymax(x, y)

                if(plot_scatter):
                    if(len(atom_pairs)==1):
                        text = legendgroup
                    else:
                        text = legendgroup + ' (' + pair + ')'
                    scatter_plots.append(go.Scatter(x=[x_max], y=[y_max], text=text, hoverinfo='text', mode='markers', name='', marker=dict(color=colors[i], size=6), showlegend=False, legendgroup=legendgroup))

        if(plot_scatter):
            for plot in scatter_plots:
                fig.add_trace(plot)

        if(len(atom_pairs)==1):
            xlabel = atom1 + ' Chemical Shift (ppm)'
            ylabel = atom2 + ' Chemical Shift (ppm)'
        else:
            xlabel = atom1[0] + ' Chemical Shift (ppm)'
            ylabel = atom2[0] + ' Chemical Shift (ppm)'

        fig.update_layout(xaxis = dict(title=xlabel, autorange='reversed'))
        fig.update_layout(yaxis = dict(title=ylabel, autorange='reversed'))

        return fig
    


    def filter_atoms_3d(self, dataframe, atom1, atom2, atom3):
        """
        Filter the dataframe to 2D chemical shifts from atom1, atom2 and atom3 that come
        from the same residue in the same protein and bmrb id for the same conditions.

        Return the x, y and z values for chemical shifts of atom1, atom2, atom3
        """

        dataframe = (
                dataframe.group_by(["BMRB entry ID", "residue number", "entity id", "temperature (K)", 'physical state', 'pH', 'pressure (atm)', 'ionic strength (M)', 'preceding residue type', 'following residue type'])
                .agg([
                    pl.col('chemical shifts (ppm)').filter(pl.col('atom')==atom1).first().alias("xval"),
                    pl.col('chemical shifts (ppm)').filter(pl.col('atom')==atom2).first().alias("yval"),
                    pl.col('chemical shifts (ppm)').filter(pl.col('atom')==atom3).first().alias("zval"),
                    pl.len().alias("group_size")
                ])
                .filter(pl.col("group_size") == 3)
            )

            
        x = dataframe["xval"].to_numpy()
        y = dataframe["yval"].to_numpy()
        z = dataframe["zval"].to_numpy()

        return x, y, z
    


    def add_3d_contour_trace(self, fig, colormap, legendgroup, showlegend, x, y, z):
        """
        Add a 2D contour trace of the x,y chemical shift points
        """

        if(self.atom_3D == 'HA-CA-N'):
            # Bin into a 3D histogram grid
            x_edges = np.arange(3.5, 5 + float(self.bin_width), float(self.bin_width))
            y_edges = np.arange(40, 65 + float(self.bin_width2), float(self.bin_width2))
            z_edges = np.arange(100, 135 + float(self.bin_width3), float(self.bin_width3))
        elif(self.atom_3D == 'H-N-CA'):
            # Bin into a 3D histogram grid
            x_edges = np.arange(5, 10 + float(self.bin_width), float(self.bin_width))
            y_edges = np.arange(100, 135 + float(self.bin_width2), float(self.bin_width2))
            z_edges = np.arange(40, 65 + float(self.bin_width3), float(self.bin_width3))
        elif(self.atom_3D == 'HA-CA-C'):
            # Bin into a 3D histogram grid
            x_edges = np.arange(3.5, 5 + float(self.bin_width), float(self.bin_width))
            y_edges = np.arange(40, 65 + float(self.bin_width2), float(self.bin_width2))
            z_edges = np.arange(170, 180 + float(self.bin_width3), float(self.bin_width3))


        hist, edges = np.histogramdd((x, y, z),  bins=[x_edges, y_edges, z_edges])

        # Compute voxel centers
        xc = (edges[0][:-1] + edges[0][1:]) / 2
        yc = (edges[1][:-1] + edges[1][1:]) / 2
        zc = (edges[2][:-1] + edges[2][1:]) / 2
        X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")

        # Flatten grid
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        values = hist.flatten()

        def hex_to_rgb(hex_color: str):
            """Convert hex color string (e.g. '#1f77b4') to an (R, G, B) tuple."""
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        rgb = hex_to_rgb(colormap[1])
        rgba = 'rgba('
        for val in rgb:
            rgba = rgba+str(val)+','
        rgba = rgba + '1)'
    

        # --- Base density (option 0) ---
        fig.add_trace(
            go.Volume(
                x=X,
                y=Y,
                z=Z,
                value=values,     # frequency density
                isomin=0,
                isomax=values.max(),
                opacity=1.0,      # low = more transparent
                surface_count=20, # smoother iso-surfaces
                colorscale=[[0, "rgba(255,255,255,0)"], [1, rgba]],
                showscale=False,
                name=legendgroup,
                showlegend=True,
            )
        )


        return fig
    

    def find_xmax_ymax_zmax(self, x, y, z):
        """
        For the x and y values which give the maximum of the histogram
        """
        """
        Add a 2D contour trace of the x,y chemical shift points
        """

        if(self.atom_3D == 'HA-CA-N'):
            # Bin into a 3D histogram grid
            x_edges = np.arange(3.5, 5 + float(self.bin_width), float(self.bin_width))
            y_edges = np.arange(40, 65 + float(self.bin_width2), float(self.bin_width2))
            z_edges = np.arange(100, 145 + float(self.bin_width3), float(self.bin_width3))
        elif(self.atom_3D == 'H-N-CA'):
            # Bin into a 3D histogram grid
            x_edges = np.arange(5, 10 + float(self.bin_width), float(self.bin_width))
            y_edges = np.arange(100, 130 + float(self.bin_width2), float(self.bin_width2))
            z_edges = np.arange(40, 65 + float(self.bin_width3), float(self.bin_width3))
        elif(self.atom_3D == 'HA-CA-C'):
            # Bin into a 3D histogram grid
            x_edges = np.arange(3.5, 5 + float(self.bin_width), float(self.bin_width))
            y_edges = np.arange(40, 65 + float(self.bin_width2), float(self.bin_width2))
            z_edges = np.arange(170, 180 + float(self.bin_width3), float(self.bin_width3))


        hist, edges = np.histogramdd((x, y, z),  bins=[x_edges, y_edges, z_edges])

        # Compute voxel centers
        xc = (edges[0][:-1] + edges[0][1:]) / 2
        yc = (edges[1][:-1] + edges[1][1:]) / 2
        zc = (edges[2][:-1] + edges[2][1:]) / 2

        max_idx = np.unravel_index(np.argmax(hist), hist.shape)
        # Get the center coordinates of that bin
        x_max = (xc[max_idx[0]] + xc[max_idx[0]+1]) / 2
        y_max = (yc[max_idx[1]] + yc[max_idx[1]+1]) / 2
        z_max = (zc[max_idx[2]] + zc[max_idx[2]+1]) / 2

        return x_max, y_max, z_max
    


    def plot_3D_histogram(self, fig: go.Figure, dataframe, atom_pairs: List, values: List, filter_type: str, plot_scatter: bool) -> go.Figure:
        """
        Plot a contour plot/2D histogram showing the data that is in each
        dataframe.

        Parameters
        ----------
        fig : go.Figure
              Plotly figure object
        dataframe : DataFrame
                    Dataframe of chemical shifts for both atoms in the atom
                    2D
        atom_pairs : List
                     Pairs of atoms to form 2D histograms together with
                     (e.g. [H-N-CA] or [HA-CA-C])
        values : list
                list of residues to loop through to plot the overlaid 2D histogram
                chemical shift distributions for atom 1 and atom 2 in the atom 2D
        filter_type : str
                      An identifier for how to filter the data (i.e. by residue is default
                      but could also be preceding residue type or following residue type)
        plot_scatter : bool
                       A flag to toggle plotting a scatter plot over the 2D histograms or not.
        
        

        Returns
        -------
        fig : go.Figure
              Plotly figure object
        
        """

        colormaps = [['white', color] for color in colors]
        point_colors = []
        point_values = []
        scatter_plots = []
        
        
        for i, value in enumerate(values):
            df = dataframe.filter(pl.col(filter_type)==value)
            for k, pair in enumerate(atom_pairs):
                atom1, atom2, atom3 = pair.split('-')
                df1 = df.filter(pl.col('atom')==atom1)
                df2 = df.filter(pl.col('atom')==atom2)
                df3 = df.filter(pl.col('atom')==atom3)
                d = pl.concat([df1, df2, df3], how='vertical')
                x, y, z = self.filter_atoms_3d(d, atom1, atom2, atom3)
                mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                x = x[mask]
                y = y[mask]
                z = z[mask]
                if(len(x)==0 or len(y)==0 or len(z)==0):
                    continue

                if(len(atom_pairs)>1):
                    legendgroup = value
                    if(k==0):
                        showlegend = True
                    else:
                        showlegend = False
                else:
                    legendgroup = value + ', {} shifts'.format(len(x))
                    showlegend= True

                fig = self.add_3d_contour_trace(fig, colormaps[i], legendgroup, showlegend, x, y, z)

                
                point_colors.append(colors[i])
                point_values.append(value)
                
                if(plot_scatter):
                    x_max, y_max, z_max = self.find_xmax_ymax_zmax(x, y, z)

                if(plot_scatter):
                    if(len(atom_pairs)==1):
                        text = legendgroup
                    else:
                        text = legendgroup + ' (' + pair + ')'
                    scatter_plots.append(go.Scatter3d(x=[x_max], y=[y_max], z=[z_max], text=text, hoverinfo='text', mode='markers', name=text, marker=dict(color=colors[i], size=6), showlegend=False, legendgroup=legendgroup))

        if(plot_scatter):
            for plot in scatter_plots:
                fig.add_trace(plot)

        if(len(atom_pairs)==1):
            xlabel = atom1 + ' Chemical Shift (ppm)'
            ylabel = atom2 + ' Chemical Shift (ppm)'
            zlabel = atom3 + ' Chemical Shift (ppm)'
        else:
            xlabel = atom1[0] + ' Chemical Shift (ppm)'
            ylabel = atom2[0] + ' Chemical Shift (ppm)'
            zlabel = atom3[0] + ' Chemical Shift (ppm)'

        fig.update_layout(
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel
            )
        )

        return fig


    def plot_histogram(self, dataframe, title: str, color: List, identifier: str, values: List) -> go.Figure:
        """
        Create a histogram of the data in the dataframe

        Parameters
        ----------
        dataframe : pd.DataFrame
                    Dataframe containing the chemical shifts to be plotted
        title : str
                Title to give the figure
        color : List
                A list of colours to be used when overlaying the data for different
                atoms or amino acids
        identifier : str
                     An identifier to further refine the dataframe ('residue' or 'atom')
        values : List
                 List of residues/atoms to loop through and plot the data for

        Returns
        -------
        fig : go.Figure
              The output Plotly figure object
        
        """
        fig = go.Figure()
      
        for value in values:
            df = dataframe.filter(pl.col(identifier)==value)
            x=df["chemical shifts (ppm)"].to_numpy()
            fig.add_trace(go.Histogram(
                x=x,
                opacity=0.6,
                name=str(value) + ', {} shifts'.format(len(x)),
                marker_color = color[value],
                histnorm="probability density",
                showlegend = True
            ))
        fig.update_traces(xbins=dict(size=float(self.bin_width)))
        fig.update_layout(barmode='overlay')
        if(len(values)==1):
            if (self.overlay_fit):
                n1 = 0
                if(self.show_disordered):
                    n1+=1
                if(self.show_all):
                    n1+=1
                if(n1==1):
                    fit_class = fitter(dataframe, float(self.bin_width))
                    A, mu, sigma = fit_class.fit()
                    x1 = np.arange(min(dataframe['chemical shifts (ppm)'].to_numpy()),max(dataframe['chemical shifts (ppm)'].to_numpy()), 0.005)
                    y1 = fit_class.normal_dist(A,mu,sigma,x1)
                    df_fit = pl.DataFrame({'x':x1,'y':y1})
                    name = 'mean={:.2f}'.format(mu) + ', standard deviation={:.2f}'.format(sigma)
                    fig.add_trace(go.Scatter(x=x1,y=y1, mode='lines', name=name,line=go.scatter.Line(color=color[values[0]], width=2)))
                    if(self.overlay_potenci):
                        fig.add_vline(x=self.potenci_shifts[self.residue][self.atom], line_width=3, line_dash="dash",line_color="black", name = 'POTENCI: '+str(self.potenci_shifts[self.residue][self.atom]))
        
            
            elif (self.overlay_potenci):
                fig.add_vline(x=self.potenci_shifts[self.residue][self.atom], line_width=3, line_dash="dash",line_color="black", name = 'POTENCI: '+str(self.potenci_shifts[self.residue][self.atom]))
    
        fig.update_layout(title=title)

        xlabel = 'Chemical Shift (ppm)'
        fig.update_layout(xaxis = dict(title=xlabel, autorange='reversed'))

        ylabel = 'Frequency Density'
        fig.update_layout(yaxis=dict(title=dict(text=ylabel)))

        return fig
    
    def plot_histogram_all(self, dataframe, residues_to_plot: List, colormap: List, flag: str) -> go.Figure:
        """
        Plot the data with different colours for each atom/residue type
        and different opacity for all/disordered residues

        Parameters
        ----------
        dataframe : pd.DataFrame
                    The dataframe containing the chemical shifts that need to be plotted.
        residues_to_plot : List
                           A list of the types of residues to plot (structured, disordered, all)
        colormap : List
                   A list of colours to be used for each atom or residue (atom/residue
                   dictated by flag)
        flag : str
               Either 'residue' or 'atom' and dictates whether to colour the plots by atom
               type or residue type.

        Returns
        -------
        fig : go.Figure
              The figure object generated by Plotly
        """
        fig = go.Figure()
        if(flag == 'atom'):
            for i, atom in enumerate(dataframe["atom"].unique().to_numpy()):
                for dataset_val in residues_to_plot:
                    filtered = dataframe.filter(pl.col("Dataset")==dataset_val)
                    if(dataset_val=='Disordered residues'):
                        opacity = 0.9
                    elif(dataset_val == 'Structured residues'):
                        opacity = 0.5
                    else:
                        opacity = 0.2
                    fig.add_trace(go.Histogram(
                        x=filtered["chemical shifts (ppm)"],
                        name=f"{dataset_val}: atom={atom}" + ', {} shifts'.format(len(filtered["chemical shifts (ppm)"].to_list())),
                        opacity=opacity,
                        marker_color = colormap[atom],
                        histnorm="probability density"
                    ))
        elif(flag == 'residue'):
            for i, residue in enumerate(dataframe["residue"].unique().to_numpy()):
                for dataset_val in residues_to_plot:
                    filtered = dataframe.filter(pl.col("Dataset")==dataset_val)
                    if(dataset_val=='Disordered residues'):
                        opacity = 0.9
                    elif(dataset_val == 'Structured residues'):
                        opacity = 0.5
                    else:
                        opacity = 0.2
                    fig.add_trace(go.Histogram(
                        x=filtered["chemical shifts (ppm)"].to_numpy(),
                        name=f"{dataset_val}: amino acid={residue}"+ ', {} shifts'.format(len(filtered["chemical shifts (ppm)"].to_list())),
                        opacity=opacity,
                        marker_color=colormap[residue],
                        histnorm="probability density"
                    ))

        fig.update_layout(barmode='overlay')
        fig.update_traces(xbins=dict(size=float(self.bin_width)))

        xlabel = 'Chemical Shift (ppm)'
        fig.update_layout(xaxis = dict(title=xlabel, autorange='reversed'))

        ylabel = 'Frequency Density'
        fig.update_layout(yaxis=dict(title=dict(text=ylabel)))
        return fig

    def convert_for_download(_self, _df):
        return _df.write_csv().encode("utf-8")
    

# A class to fit a normal distribution to the chemical shift statistics
class fitter():
    def __init__(self, dataframe, bin_width: Union[int,float]):
        """
        The class will fit the data in the dataframe to a gaussian (enabling
        the mean/standard deviation of the distributions to be determined).

        Parameters
        ----------
        dataframe : pd.DataFrame
                    A dataframe containing the data to fit a gaussian distribution
                    too.
        bin_width : int or float
                    The width of the histogram bins currently selected (in units of ppm)
        """
        self.shifts = dataframe['chemical shifts (ppm)'].to_numpy()
        number_of_bins = int((max(self.shifts)-min(self.shifts))/bin_width)
        self.hist, bin_edges = np.histogram(self.shifts, bins=number_of_bins, density=True)
        self.bin_centres = (bin_edges[:-1]+bin_edges[1:])/2

    def func(self,p0: List) -> NDArray:
        """
        The function to fit the histogram too
        y = A*exp(-(x-mu)^2/sigma^2)

        Parameters
        ----------
        p0 : List
             List of parameters to govern the gaussian function
        
        """
        A, mu, sigma = p0
        return A*np.exp(-((mu-self.bin_centres)**2)/(sigma**2))
    
    def chi(self, p0):
        """
        Calculate the chi (residuals) between the histogram values and 
        the fitted gaussian function

        Parameters
        ----------
        p0 : List
             List of parameters to govern the gaussian function        
        """
        y_real = self.hist
        y_calc = self.func(p0)
        return y_real - y_calc
    
    def fit(self) -> List[Union[int,float]]:
        """
        Function that performs the fit on the data.
        """
        A_initial = np.max(self.hist)
        mu_initial = np.mean(self.shifts)
        sigma_initial = np.std(self.shifts)
        p0 = [A_initial, mu_initial, sigma_initial]
        A, mu, sigma = leastsq(self.chi,p0)[0]
        return A, mu, sigma
    
    def normal_dist(self, A, mu, sigma, x):
        """
        Evaluating the normal distribution of parameters
        A, mu and sigma over a range of points x.
        """
        return A*np.exp(-((x-mu)**2)/(sigma**2))
