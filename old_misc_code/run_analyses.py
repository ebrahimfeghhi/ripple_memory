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
sub_selection = 'whole'
df = get_data_index("r1") # all RAM subjects
exp = 'catFR1' # 'FR1' 'catFR1' 'RepFR1'

# define region names of interest
non_HPC_regions_all = ['AMY', 'ENTPHC']
non_HPC_regions = []
ripple_regions = ['ca1', 'dg', 'ca3'] 
hpc_regions_all = [['ca1'], ['ca3', 'dg']] 
hpc_regions = hpc_regions_all

rs_str = ''
num_regions = len(ripple_regions)
rs_str = convert_list_to_string(ripple_regions)
hpc_ripple_types_all = ['single_elec', 'any_ipsi_elec']
hpc_ripple_types = [hpc_ripple_types_all[0]]
select_subfield = True
data_folder = 'SWR_scratch'

ripple_bin_start_end = [100, 1100] 
ripple_str = convert_list_to_string(ripple_bin_start_end)

run_mixed_effect_models = False # if true, runs mixed effects models for the analyses specified in analysis_mode 

print("Creating save folder directories")
saveFolder_all = f'updates/figures_{exp}_all_data'
os.makedirs(saveFolder_all, exist_ok=True)
saveFolder_ripple = f'updates/figures_{exp}_ripple'
os.makedirs(saveFolder_ripple, exist_ok=True)
saveFolder_serial = f'updates/figures_{exp}_serial_position'
os.makedirs(saveFolder_serial, exist_ok=True)


print("Generating nonHPC plots")
for region_name in non_HPC_regions: 
    
        print(f"Generating SME plots for region {region_name}")

        RS = HFA_ripples_analysis(exp=exp, df=df, sub_selection=sub_selection, data_folder=data_folder, 
                                    select_subfield=[], hpc_regions=[], 
                                    ripple_regions=ripple_regions, 
                                    ripple_bin_start_end=ripple_bin_start_end)
        RS.remove_subject_sessions()
        
        ripple_str = convert_list_to_string(ripple_bin_start_end)
        
        title_all = f'All events {region_name}'
        title_r = f'Ripple {region_name}'
        title_nr = f'No ripple {region_name}'
        title_e = f'Early {region_name}'
        title_m = f'Middle {region_name}'
        
        # First only load data for which there sessions with ipsilateral HPC electrodes
        # that contain ripples
        RS.load_data_from_cluster('encoding', ripple_bool = True, region_name=region_name)
        RS.getStartArray()
        RS.select_idxs_numpy()
    
        print(f"Creating ripple/no ripple SME plots for {region_name}")
        savePath_SME_ripple_fig = f'{saveFolder_ripple}/{title_r}'
        savePath_SME_noripple_fig = f'{saveFolder_ripple}/{title_nr}'
        RS.plot_SME_HFA(analysis_type=0, mode=0, title_str=title_r, 
                        savePath=savePath_SME_ripple_fig)
        RS.plot_SME_HFA(analysis_type=0, mode=1, title_str=title_nr, 
                        savePath=savePath_SME_noripple_fig) 
        
        # Now load all data
        RS.load_data_from_cluster('encoding', ripple_bool = False, region_name=region_name, hpc_ripple_type='')
        RS.getStartArray()
        RS.select_idxs_numpy()
        
        print(f"Creating all data SME plots for {region_name}")
        savePath_SME_all_events_fig = f'{saveFolder_all}/{title_all}'
        RS.plot_SME_HFA(analysis_type=2, mode=None, title_str=title_all, 
                    savePath=savePath_SME_all_events_fig)
                        
        savePath_SME_early_fig = f'{saveFolder_serial}/{title_e}'
        savePath_SME_middle_fig = f'{saveFolder_serial}/{title_m}'
    
        print(f"Creating serial position SME plots for {region_name}")
        RS.plot_SME_HFA(analysis_type=1, mode=0, title_str=title_e, 
                        savePath=savePath_SME_early_fig)
        RS.plot_SME_HFA(analysis_type=1, mode=1, title_str=title_m, 
                        savePath=savePath_SME_middle_fig)
        
# now do HPC 
print("Generating HPC plots")
region_name = 'HPC'
for hpc_region in hpc_regions:
    
    ripple_str = convert_list_to_string(ripple_bin_start_end)
    hpc_region_str = convert_list_to_string(hpc_region)
    
    RS = HFA_ripples_analysis(exp=exp, df=df, sub_selection=sub_selection, data_folder=data_folder, 
                                select_subfield=select_subfield, hpc_regions=hpc_region, 
                                ripple_regions=ripple_regions, 
                                ripple_bin_start_end=ripple_bin_start_end)
    RS.remove_subject_sessions()
    
    title_all = f'All events {region_name} {hpc_region_str}'
    title_r = f'Ripple {region_name} {hpc_region_str}'
    title_nr = f'No ripple {region_name} {hpc_region_str}'
    title_e = f'Early {region_name} {hpc_region_str}'
    title_m = f'Middle {region_name} {hpc_region_str}'

    for hpc_ripple_type in hpc_ripple_types:

            title_r = f'Ripple {region_name} {hpc_region_str} {hpc_ripple_type}'
            title_nr = f'No ripple {region_name} {hpc_region_str} {hpc_ripple_type}'
           
            RS.load_data_from_cluster('encoding', ripple_bool = False, region_name=region_name, hpc_ripple_type=hpc_ripple_type)
            RS.getStartArray()
            RS.select_idxs_numpy()

            print(f"Creating ripple/no ripple SME plots for {hpc_region_str} {hpc_ripple_type}")
            savePath_SME_ripple_fig = f'{saveFolder_ripple}/{title_r}'
            savePath_SME_noripple_fig = f'{saveFolder_ripple}/{title_nr}'
            RS.plot_SME_HFA(analysis_type=0, mode=0, title_str=title_r, 
                            savePath=savePath_SME_ripple_fig)
            RS.plot_SME_HFA(analysis_type=0, mode=1, title_str=title_nr, 
                            savePath=savePath_SME_noripple_fig) 
                
    print(f"Creating all data SME plots for {hpc_region_str}")
    savePath_SME_all_events_fig = f'{saveFolder_all}/{title_all}'
    RS.plot_SME_HFA(analysis_type=2, mode=None, title_str=title_all, 
                savePath=savePath_SME_all_events_fig)
            
    print(f"Creating serial position SME plots for {hpc_region_str}")           
    savePath_SME_early_fig = f'{saveFolder_serial}/{title_e}'
    savePath_SME_middle_fig = f'{saveFolder_serial}/{title_m}'
    RS.plot_SME_HFA(analysis_type=1, mode=0, title_str=title_e, 
                    savePath=savePath_SME_early_fig)
    RS.plot_SME_HFA(analysis_type=1, mode=1, title_str=title_m, 
                    savePath=savePath_SME_middle_fig)
    
    if run_mixed_effect_models:

        dep_var_list = ['ripple_exists', 'HFA_mean']
        formula_list = ['ripple_exists~1+serial_pos+recalled', 'HFA_mean~1+serial_pos+recalled']
        re_formula_list = ['1+serial_pos+recalled', '1+serial_pos+recalled']
        params_list = [['Intercept', 'serial_pos', 'recalled'], 
                    ['Intercept', 'serial_pos', 'recalled']]
        
        RS.lmm_model(dep_var_list=dep_var_list, formula_list=formula_list, 
                    re_formula_list=re_formula_list, recalled_bool_list=[False, False], 
                    params_list=params_list, 
                    savePath=f'{saveFolder_all}/mem_model_{title_all}')
            
            
                
