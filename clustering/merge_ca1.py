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

for non_ca1_region in non_ca1_regions:

    basePath = "/home1/efeghhi/ripple_memory/"

    non_ca1_df = pd.read_csv(f"{basePath}session_data/{non_ca1_region}.csv")
    ca1_df = pd.read_csv(f"{basePath}session_data/ca1.csv")
   
    non_ca1_df['session_list_sp'] = non_ca1_df['sess'] + "-L" + non_ca1_df['list_num'].astype(str) + "-S" + non_ca1_df['serial_pos'].astype(str)
    ca1_df['session_list_sp'] = ca1_df['sess'] + "-L" + ca1_df['list_num'].astype(str)  + "-S" + ca1_df['serial_pos'].astype(str)
    
    
    ca1_rows_with_nhr = ca1_df[ca1_df['session_list_sp'].isin(non_ca1_df['session_list_sp'])]
    nhr_rows_with_ca1 = non_ca1_df[non_ca1_df['session_list_sp'].isin(ca1_df['session_list_sp'])]
    sorted_nhr = nhr_rows_with_ca1.sort_values('session_list_sp').reset_index()
    sorted_ca1 = ca1_rows_with_nhr.sort_values('session_list_sp').reset_index()
    
    print(sorted_ca1.shape[0], sorted_nhr.shape[0])
    
    # do some checks to make sure we're not concat the wrong rows 
    are_equal_1 = (sorted_nhr['sess'].values == sorted_ca1['sess'].values)
    are_equal_2 = (sorted_nhr['clust'].values == sorted_ca1['clust'].values)
    
    if np.all(are_equal_1) and np.all(are_equal_2):
        pass
    else:
        print("DFs are misaligned")
        sys.exit()
        
    sorted_ca1 = sorted_ca1.add_suffix('_ca1')

    merged_df = pd.concat([sorted_nhr, sorted_ca1[['hfa_short_ca1', 'hfa_long_ca1', 'ripple_exists_short_ca1', 
                                        'ripple_exists_long_ca1']]], axis=1)

    merged_df.to_csv(f"{savePATH}{non_ca1_region}_ca1.csv")
