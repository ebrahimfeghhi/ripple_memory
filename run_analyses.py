import pandas as pd 
pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)
from cmlreaders import get_data_index
import xarray as xarray
import matplotlib.pyplot as plt
from pylab import *
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42 # fix fonts for Illustrator
from general import *
from SWRmodule import *
from ripples_HFA_SME import ripple_analysis_SME
from ripples_HFA_SCE import ripple_analysis_SCE

def convert_list_to_string(list1):
    
    list_str = ''
    
    if len(list1) > 0:
        for i, val in enumerate(list1):
            list_str += str(val)
            if i + 1 < len(list1):
                list_str += '_'
                
    return list_str

# Set analysis mode, SME or SCE 
analyses_mode = 'SCE'

# init variables
sub_selection = 'first_half'
df = get_data_index("r1") # all RAM subjects
print(f"df {type(df)}")
exp = 'catFR1' # 'FR1' 'catFR1' 'RepFR1'
region_name = 'HPC'
regions_selected = ['ca1']
hpc_regions = ['ca1', 'dg']
rs_str = ''

num_regions = len(regions_selected)
rs_str = convert_list_to_string(regions_selected)
hpc_ripple_type = 'single_elec'
select_subfield = True

if analyses_mode == 'SCE':

    data_folder = 'SWR_semantic_scratch'   
    ripple_bin_start_end = [100, 1700] 
    ripple_str = convert_list_to_string(ripple_bin_start_end)
    ripple_delta = True
    HFA_delta = False
    
    RS = ripple_analysis_SCE(exp=exp, df=df, sub_selection=sub_selection, data_folder=data_folder, hpc_ripple_type=hpc_ripple_type,
                             select_subfield=select_subfield, hpc_regions=hpc_regions, regions_selected=regions_selected, 
                             ripple_bin_start_end=ripple_bin_start_end)
    
    RS.remove_subject_sessions()
    RS.load_data_from_cluster('encoding', region_name=region_name)
    RS.getStartArray()
    RS.select_idxs_numpy()
    RS.create_clustered_array()
    RS.mixed_effects_modeling(savebool=False)

    
if analyses_mode == 'SME':

    data_folder = 'SWR_scratch'
    ripple_bin_start_end = [100, 1100]
    ripple_str = convert_list_to_string(ripple_bin_start_end)
    hpc_regions_str = convert_list_to_string(hpc_regions)
    
    title_r = f'Ripple {rs_str} {region_name} {ripple_str} {hpc_regions_str}'
    title_nr = f'No ripple {rs_str} {region_name} {ripple_str} {hpc_regions_str}'
    
    savePath_SME_ripple_fig = f'updates/figures/{title_r}'
    savePath_SME_noripple_fig = f'updates/figures/{title_nr}'
        
    RS = ripple_analysis_SME(exp=exp, df=df, sub_selection=sub_selection, data_folder=data_folder, hpc_ripple_type=hpc_ripple_type,
                             select_subfield=select_subfield, hpc_regions=hpc_regions, regions_selected=regions_selected, 
                             ripple_bin_start_end=ripple_bin_start_end)
    
    RS.remove_subject_sessions()
    RS.load_data_from_cluster('encoding', region_name=region_name)
    RS.getStartArray()
    RS.select_idxs_numpy()
    
    RS.plot_SME_HFA(1, title_r, savePath_SME_ripple_fig)
    RS.plot_SME_HFA(2, title_nr, savePath_SME_noripple_fig)

    mem_model, ols_model = RS.SME_ripple_interaction()

    RS.save_model_info(mem_model, f'updates/stats_results/mem_model_SME_{rs_str}_{region_name}_{ripple_str}_{hpc_regions_str}.csv')
    RS.save_model_info(ols_model, f'updates/stats_results/OLS_model_SME_{rs_str}_{region_name}_{ripple_str}_{hpc_regions_str}.csv')
    
    RS.save_ripple_information(f'updates/ripple_info/{rs_str}_{region_name}_{ripple_str}.csv')

