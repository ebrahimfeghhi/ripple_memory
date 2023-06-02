import pandas as pd 
pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)
from cmlreaders import get_data_index
import xarray as xarray
import matplotlib.pyplot as plt
from pylab import *
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42 # fix fonts for Illustrator
from general import *
from SWRmodule import *
from HFA_ripples_analysis import HFA_ripples_analysis

def convert_list_to_string(list1):
    
    list_str = ''
    
    if len(list1) > 0:
        for i, val in enumerate(list1):
            list_str += str(val)
            if i + 1 < len(list1):
                list_str += '_'
                
    return list_str

# init variables
sub_selection = 'first_half'
df = get_data_index("r1") # all RAM subjects
exp = 'catFR1' # 'FR1' 'catFR1' 'RepFR1'

# define region names of interest
region_names = ['HPC', 'AMY', 'nonHPC_MTL']
ripple_regions = ['ca1', 'dg', 'ca3'] 
hpc_regions = [['ca1'], ['ca3', 'dg']] # only applies w/ hpc

rs_str = ''
num_regions = len(ripple_regions)
rs_str = convert_list_to_string(ripple_regions)
hpc_ripple_types = ['single_elec', 'avg_elec']
select_subfield = True
data_folder = 'SWR_scratch'

ripple_bin_start_end = [100, 1100] 
ripple_str = convert_list_to_string(ripple_bin_start_end)

for region_name in region_names:
    for i, hpc_region in enumerate(hpc_regions):
        for j, hpc_ripple_type in enumerate(hpc_ripple_types):
        
            # ignore nested for loops unless HPC
            if (i > 0 or j > 0) and region_name != 'HPC':
                continue
            
            print(f"Generating SME plots for region {region_name}")

            RS = HFA_ripples_analysis(exp=exp, df=df, sub_selection=sub_selection, data_folder=data_folder, hpc_ripple_type=hpc_ripple_type,
                                        select_subfield=select_subfield, hpc_regions=hpc_region, ripple_regions=ripple_regions, 
                                        ripple_bin_start_end=ripple_bin_start_end)

            RS.remove_subject_sessions()
            RS.load_data_from_cluster('encoding', region_name=region_name)
            RS.getStartArray()
            RS.select_idxs_numpy()
            
            ripple_str = convert_list_to_string(ripple_bin_start_end)
            hpc_region_str = convert_list_to_string(hpc_region)
            
            title_r_nr = f'All events {region_name}'
            title_r = f'Ripple {region_name}'
            title_nr = f'No ripple {region_name}'
            
            if region_name == 'HPC':
                title_r_nr = f'{title_r_nr} {hpc_region_str} {hpc_ripple_type}'
                title_r = f'{title_r} {hpc_region_str} {hpc_ripple_type}' 
                title_nr = f'{title_nr} {hpc_region_str} {hpc_ripple_type}' 
            
            savePath_SME_all_events_fig = f'updates/figures/{title_r_nr}'
            savePath_SME_ripple_fig = f'updates/figures/{title_r}'
            savePath_SME_noripple_fig = f'updates/figures/{title_nr}'
            
            RS.plot_SME_HFA(0, title_r_nr, savePath_SME_all_events_fig)
            RS.plot_SME_HFA(1, title_r, savePath_SME_ripple_fig)
            RS.plot_SME_HFA(2, title_nr, savePath_SME_noripple_fig)
        
            mem_model, ols_model = RS.SME_ripple_interaction()
                    
            RS.save_model_info(mem_model, f'updates/stats_results/mem_model_SME_{title_r_nr}.csv')
            RS.save_model_info(ols_model, f'updates/stats_results/OLS_model_SME_{title_r_nr}.csv')
            
