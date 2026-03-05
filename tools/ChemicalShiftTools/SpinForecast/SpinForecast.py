import polars as pl
from pathlib import Path
import streamlit as st


atoms = ['H', 'N', 'CA', 'C', 'CB', 'HA', 'HB']
residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP','TYR','VAL']
residue_dict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W','TYR': 'Y','VAL': 'V','': 'X'}
residue_dict_reverse = {v: k for k,v in residue_dict.items()}


class SpinForecastBackend():
    def __init__(self):
        # Temperature and pH corrected data to pH 7.0 and a temperature of 298K (terminal residues omitted)
        self.data_disordered = self.read_data('./Shifts_Disordered_Corrected_Referenced/')

    @st.cache_resource
    def read_data(_self, file_path: str):
        """
        Read the dataframes and then return
        """

        for i, residue in enumerate(residues):
            file = Path(file_path+residue+'.parquet')
            df = pl.scan_parquet(file)
            if(i==0):
                df_total = df
            else:
                df_total = pl.concat([df_total, df], how='vertical')

        df_total = df_total.with_columns((pl.col('chemical shifts (ppm)')-pl.col('referencing offset (ppm)')).alias('chemical shifts (ppm)'))

        return df_total
    

    @st.cache_data
    def read_terminal_shifts(_self, file_path: str):
        """
        Read the dataframes and then return (these have not been temperature and pH corrected by POTENCI)
        """

        file = Path(file_path+'.parquet')
        df = pl.scan_parquet(file)
        df_total = df
 
        return df_total



    
    def perform_calc(self, sequence, potenci_condition_corrections):
        """
        From the sequence, work out probability distributions of chemical shifts
        """

        # Convert the sequence to a dataframe showing the nearest neighbor residues (ignoring the terminal residues)
        df = self.calculate_nearest_neighbors(sequence)
        shift_distribution_dictionary = self.shifts_distributions(df, potenci_condition_corrections)
        return shift_distribution_dictionary
    

    def calculate_nearest_neighbors(self, sequence):
        """
        From the sequence of amino acids return a dataframe with the following columns:
        residue number, amino acid, following residue type, preceding residue type

        Ignore the terminal residues as the models have not been used to predict these
        """

        seq_list = list(sequence)
        dataframes = []


        for i, aa in enumerate(seq_list):
            
            index_iminus1 = i-1
            if(index_iminus1>=0):
                iminus1 = seq_list[i-1]
            else:
                iminus1 = 'X'
                # N terminal residues are not corrected for by potenci and must be read separately here
                self.Nterminal_shifts = self.read_terminal_shifts('./Shifts_Disordered/'+residue_dict_reverse[aa])

            index_iplus1 = i+1
            if(index_iplus1<=len(seq_list)-1):
                iplus1 = seq_list[i+1]
            else:
                iplus1 = 'X'
                # N terminal residues are not corrected for by potenci and must be read separately here
                self.Cterminal_shifts = self.read_terminal_shifts('./Shifts_Disordered/'+residue_dict_reverse[aa])
       
            
            
            row = [str(i+1), aa, iplus1, iminus1]
            dataframes.append(pl.DataFrame([row], schema={"residue number": str, "amino acid": str, "i+1 residue": str, "i-1 residue": str}, orient='row'))

        
        dataframe = pl.concat(dataframes)

        return dataframe
        
    
    def shifts_distributions(self, dataframe_for_sequence, potenci_condition_correction):
        residue_numbers = dataframe_for_sequence['residue number'].to_list()
        distribution_dictionary = {}
        for atom in atoms:
            distribution_dictionary[atom] = {}
            for number in residue_numbers:
                df_total_atom = self.data_disordered.filter(pl.col('atom')==atom)
                
                df = dataframe_for_sequence.filter(pl.col('residue number')==number)
                residue = df['amino acid'].to_list()[0]
                iminus1_residue = df['i-1 residue'].to_list()[0]
                if(iminus1_residue=='X'):
                    df_total_atom = self.Nterminal_shifts.filter(pl.col('atom')==atom)
                iplus1_residue = df['i+1 residue'].to_list()[0]
                if(iplus1_residue == 'X'):
                    df_total_atom = self.Cterminal_shifts.filter(pl.col('atom')==atom)
                if(atom == 'H' and residue == 'P'):
                    continue
                if(atom == 'CB' and residue == 'G'):
                    continue
                if(atom == 'HB' and residue == 'G'):
                    continue

                df1 = df_total_atom

                if(atom=='H'):
                    df1 = df1.filter(pl.col('chemical shifts (ppm)')<= 9.0)
                    df1 = df1.filter(pl.col('chemical shifts (ppm)')>= 7.5)
                elif(atom=='N'):
                    if(residue_dict_reverse[residue]!='PRO'):
                        df1 = df1.filter(pl.col('chemical shifts (ppm)')<= 130.0)
                        df1 = df1.filter(pl.col('chemical shifts (ppm)')>= 100.0)
                    else:
                        df1 = df1.filter(pl.col('chemical shifts (ppm)')<= 145.0)
                        df1 = df1.filter(pl.col('chemical shifts (ppm)')>= 130.0)
                elif(atom=='C'):
                    df1 = df1.filter(pl.col('chemical shifts (ppm)')<= 185.0)
                    df1 = df1.filter(pl.col('chemical shifts (ppm)')>= 165.0)


                if(iminus1_residue=='X'):
                    if(atom=='H' or atom=='N'):
                        continue
                    df_total = df1.filter(pl.col('residue')==residue_dict_reverse[residue])
                    df_total1 = df_total.filter(pl.col('preceding residue type')==residue_dict_reverse[iminus1_residue])
                    df_total2 = df_total1.filter(pl.col('following residue type')==residue_dict_reverse[iplus1_residue])
                elif(iplus1_residue=='X'):
                    df_total = df1.filter(pl.col('residue')==residue_dict_reverse[residue])
                    df_total1 = df_total.filter(pl.col('following residue type')==residue_dict_reverse[iplus1_residue])
                    df_total2 = df_total1.filter(pl.col('preceding residue type')==residue_dict_reverse[iminus1_residue])

                elif(atom=='H' or atom=='N'):
                    df_total = df1.filter(pl.col('residue')==residue_dict_reverse[residue])
                    df_total1 = df_total.filter(pl.col('preceding residue type')==residue_dict_reverse[iminus1_residue])
                    df_total2 = df_total1.filter(pl.col('following residue type')==residue_dict_reverse[iplus1_residue])
                    
                else:
                    df_total = df1.filter(pl.col('residue')==residue_dict_reverse[residue])
                    df_total1 = df_total.filter(pl.col('following residue type')==residue_dict_reverse[iplus1_residue])
                    df_total2 = df_total1.filter(pl.col('preceding residue type')==residue_dict_reverse[iminus1_residue])
                
                if(len(df_total2.select('chemical shifts (ppm)').collect().to_series().to_numpy())>=10):
                    df_final = df_total2
                elif(len(df_total1.select('chemical shifts (ppm)').collect().to_series().to_numpy())>=10):
                    df_final = df_total1
                else:
                    df_final = df_total

 
                try:
                    correction = potenci_condition_correction.filter(pl.col('residue number')==number)[atom].to_list()[0]
                    distribution_dictionary[atom][number] = df_final.select('chemical shifts (ppm)').collect().to_series().to_numpy() + correction
                except:
                    distribution_dictionary[atom][number] = df_final.select('chemical shifts (ppm)').collect().to_series().to_numpy()
        
                
                
        return distribution_dictionary
    


        