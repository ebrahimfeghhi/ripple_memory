import pandas as pd
import numpy as np
import sys

'''
In order to examine whether the relationship between HFA in other brain regions 
and clustring is mediated by HPC, we need to save the HPC data for the same brain sessions.
This script combines 
'''

non_ca1_regions = ['amy', 'entphc', 'ca3_dg']

savePATH = "/home1/efeghhi/ripple_memory/session_data/"
basePath = "/home1/efeghhi/ripple_memory/"

ca1_df = pd.read_csv(f"{basePath}session_data/ca1_all.csv")
ca1_ripple_sess = ca1_df[['sess', 'list_num', 'serial_pos', 'ripple_exists_short', 'hfa_short']]
ca1_ripple_sess.rename(columns={'ripple_exists_short': 'ripple_exists_short_ca1', 'hfa_short': 'hfa_short_ca1'}, inplace=True)

for non_ca1_region in non_ca1_regions:

    basePath = "/home1/efeghhi/ripple_memory/"

    non_ca1_df = pd.read_csv(f"{basePath}session_data/{non_ca1_region}_all.csv")
    
    # merge on session + list_num + serial_pos
    merged_df = pd.merge(non_ca1_df, ca1_ripple_sess, on=['sess', 'list_num', 'serial_pos'])
    
    merged_df.to_csv(f"{savePATH}{non_ca1_region}_all_ca1.csv")
