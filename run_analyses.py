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
exp = 'FR1' # 'FR1' 'catFR1' 'RepFR1'
ripple_regions = ['ca1', 'dg', 'ca3'] 
if exp == 'FR1':
    data_folder = 'SWR_scratch'
else:
    data_folder = 'SWR_semantic_scratch'
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
saveFolder_clust = f'updates/figures_{exp}_clust'
os.makedirs(saveFolder_clust, exist_ok=True)
saveFolder_ripple_local = f'updates/figures_{exp}_ripple_local'
os.makedirs(saveFolder_ripple_local, exist_ok=True)

def run_analyses(analysis_mode, region_name, hpc_ripple_type='',
                 sub_region=[]):
    
    '''
    :param list analysis_mode: list containing all analyses to run 
    '''
    
    # this will be an empty str for non HPC regions
    sr_str= convert_list_to_string(sub_region) 
    
    load_ripple_data = False
    load_all_data = False
    
    # for HPC, ripple_data and all data is the same
    if region_name == 'HPC':
        load_all_data = True
        
    elif 'ripple' in analysis_mode or 'clustering_by_ripple' in analysis_mode:         
        load_ripple_data = True 
        
    elif 'all' in analysis_mode or 'serial_pos' in analysis_mode or 'local_ripple' in analysis_mode:
        load_all_data = True
            
    if load_ripple_data: 
        
        RS = HFA_ripples_analysis(exp=exp, df=df, sub_selection=sub_selection, data_folder=data_folder, 
                                        select_subfield=True, hpc_regions=sub_region, 
                                        ripple_regions=ripple_regions, 
                                        ripple_bin_start_end=ripple_bin_start_end)
        RS.remove_subject_sessions()
        RS.load_data_from_cluster('encoding', ripple_bool = True, region_name=region_name, hpc_ripple_type=hpc_ripple_type)
        RS.getStartArray()
        RS.select_idxs_numpy()  
        if exp=='catFR1':
            RS.create_semantic_clustered_array()
        if exp=='FR1':
            RS.create_temporal_clustered_array()

    # for HPC, just default to loading RS_all
    if load_all_data: 
        
        RS_all = HFA_ripples_analysis(exp=exp, df=df, sub_selection=sub_selection, data_folder=data_folder, 
                                        select_subfield=True, hpc_regions=sub_region, 
                                        ripple_regions=ripple_regions, 
                                        ripple_bin_start_end=ripple_bin_start_end)
        RS_all.remove_subject_sessions()
        RS_all.load_data_from_cluster('encoding', ripple_bool = False, region_name=region_name, hpc_ripple_type=hpc_ripple_type)
        RS_all.getStartArray()
        RS_all.select_idxs_numpy()  
        if RS_all.semantic_data:
            RS_all.create_clustered_array()
            
        if load_ripple_data and region_name == 'HPC':
            RS = RS_all 
            
    for am in analysis_mode: 
        
        # ripple/no ripple SME with HPC ripples
        if am == 'ripple_hpc':
            
            title_r = f'Ripple {region_name} {sr_str} {hpc_ripple_type}'
            title_nr = f'No ripple {region_name} {sr_str} {hpc_ripple_type}'
            
            savePath_SME_ripple_fig = f'{saveFolder_ripple}/{title_r}'
            savePath_SME_noripple_fig = f'{saveFolder_ripple}/{title_nr}'
            
            RS.plot_SME_or_SCE_HFA(analysis_type=0, mode=0, title_str=title_r, 
                        savePath=savePath_SME_ripple_fig)
            RS.plot_SME_or_SCE_HFA(analysis_type=0, mode=1, title_str=title_nr, 
                        savePath=savePath_SME_noripple_fig) 
            
        # ripple/no ripple SME with ripples from the same region
        if am == 'local_ripple':
            
            title_r = f'Ripple {region_name} {sr_str}'
            title_nr = f'No ripple {region_name} {sr_str}'
            
            savePath_SME_ripple_fig = f'{saveFolder_ripple_local}/{title_r}'
            savePath_SME_noripple_fig = f'{saveFolder_ripple_local}/{title_nr}'
            
            RS_all.plot_SME_or_SCE_HFA(analysis_type=0, mode=0, title_str=title_r, 
                        savePath=savePath_SME_ripple_fig)
            RS_all.plot_SME_or_SCE_HFA(analysis_type=0, mode=1, title_str=title_nr, 
                        savePath=savePath_SME_noripple_fig) 
            
        if am == 'serial_pos':
                            
            title_e = f'Early {region_name} {sr_str}'
            title_m = f'Middle {region_name} {sr_str}'
            savePath_SME_early_fig = f'{saveFolder_serial}/{title_e}'
            savePath_SME_middle_fig = f'{saveFolder_serial}/{title_m}'
            
            RS_all.plot_SME_or_SCE_HFA(analysis_type=1, mode=0, title_str=title_e, 
                        savePath=savePath_SME_early_fig)
            RS_all.plot_SME_or_SCE_HFA(analysis_type=1, mode=1, title_str=title_m, 
                        savePath=savePath_SME_middle_fig)
            
        if am == 'clustering':

            print(f"Clustering plots for {region_name}")
            
            title_clust = f'Recalled words {region_name} {sr_str}'
            
            savePath_SCE_clust_fig = f'{saveFolder_clust}/{title_clust}'
            
            RS_all.plot_SME_or_SCE_HFA(analysis_type=3, mode=None, title_str=title_clust, 
                    savePath=savePath_SCE_clust_fig)
            
        if am == 'clustering_by_ripple':
            
            print(f"Clustering by ripple plots for {region_name}")
            
            title_clust_r = f'Recalled + ripple {region_name} {sr_str}'
            title_clust_nr = f'Recalled + no ripple {region_name} {sr_str}'
            
            savePath_SCE_clust_fig_r = f'{saveFolder_clust}/{title_clust_r}'
            savePath_SCE_clust_fig_nr = f'{saveFolder_clust}/{title_clust_nr}'
            
            RS.plot_SME_or_SCE_HFA(analysis_type=4, mode=0, title_str = title_clust_r, 
                                   savePath=savePath_SCE_clust_fig_r)
            RS.plot_SME_or_SCE_HFA(analysis_type=4, mode=1, title_str = title_clust_nr, 
                                   savePath=savePath_SCE_clust_fig_nr)
            
        if am == 'all':
            
            title_all = f'All events {region_name}'
            savePath_SME_all_events_fig = f'{saveFolder_all}/{title_all}'
            RS_all.plot_SME_or_SCE_HFA(analysis_type=2, mode=None, title_str=title_all, 
                    savePath=savePath_SME_all_events_fig)
            
analysis_mode = []         
region_names = ['HPC']
hpc_subregions = [['ca1']]
hpc_ripple_types = ['single_elec', 'any_ipsi_elec']
for rn in region_names:
    if rn == 'HPC':
        for sr in hpc_subregions:
            for hpc_ripple_type in hpc_ripple_types:
                run_analyses(analysis_mode, rn, sub_region=sr, hpc_ripple_type=hpc_ripple_type) 
    else:
        run_analyses(analysis_mode, rn) 
        
            
    
            

        
        
        
            
    