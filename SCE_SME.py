from load_data import * 
from analyze_data import * 
import sys
import warnings
import matplotlib.pyplot as plt
from SWRmodule import *
sys.path.append('/home1/efeghhi/ripple_memory/')
from brain_labels import HPC_labels, ENT_labels, PHC_labels, temporal_lobe_labels,\
                        MFG_labels, IFG_labels, nonHPC_MTL_labels, ENTPHC_labels, AMY_labels

############### set parameters ###############
encoding_mode = 0

if encoding_mode: 
    start_time = -700
    end_time = 2300
    num_bins = 150
    catFR_dir = '/scratch/efeghhi/catFR1/ENCODING/'
    
else:
    start_time = -2000
    end_time = 2000
    num_bins = 200
    catFR_dir = '/scratch/efeghhi/catFR1/IRIonly/'
    
xr = np.linspace(start_time/1000, end_time/1000, num_bins)

ripple_start = 550
ripple_end = 900
##############################################

region_name = '' # if empty string, loads all data

# load all data
data_dict = load_data(catFR_dir, region_name=region_name, encoding_mode=encoding_mode)

if encoding_mode: 
    data_dict = remove_wrong_length_lists(data_dict)

# ca1
ca1_elecs = [x for x in HPC_labels if 'ca1' in x]
data_dict_ca1 = select_region(data_dict, ca1_elecs)
count_num_trials(data_dict_ca1, "ca1")

# dg
ca3_dg_elecs = [x for x in HPC_labels if 'dg' in x]
data_dict_ca3_dg = select_region(data_dict, ca3_dg_elecs)
count_num_trials(data_dict_ca3_dg, "ca3_dg")

# amy
data_dict_amy = select_region(data_dict, AMY_labels)
count_num_trials(data_dict_amy, "amy")

# entphc
data_dict_entphc = select_region(data_dict, ENTPHC_labels)
count_num_trials(data_dict_entphc, "entphc")

data_dicts = [data_dict_ca1, data_dict_ca3_dg, data_dict_amy, data_dict_entphc]
data_dict_labels = ['ca1', 'dg', 'amy', 'entphc']
modes = [[2,3], [2,3], [1], [1]]
skip_regions = ['dg', 'amy']

for data_dict_region, brain_region, mode in zip(data_dicts, data_dict_labels, modes):
        
    if brain_region in skip_regions:
        continue

    # create clustered int array
    clustered_int = create_semantic_clustered_array(data_dict_region, encoding_mode)
    data_dict_region['clust_int'] = clustered_int

    dd_trials = dict_to_numpy(data_dict_region, order='C')
    ripple_exists_idxs = np.argwhere(np.sum(dd_trials['ripple'][:, ripple_start:ripple_end],axis=1)>0)
    ripple_exists = np.zeros(dd_trials['ripple'].shape[0])
    ripple_exists[ripple_exists_idxs] = 1
    dd_trials['ripple_exists'] = ripple_exists

    for m in mode:
        
        print("Mode: ", m)
        print("Brain region: ", brain_region)
        plot_SCE(dd_trials, power='HFA', mode=m, xr=xr, region=brain_region, 
                ymin=None, ymax=None, smoothing_triangle=5, encoding_mode=encoding_mode, behav_key='correct')
        plot_SCE(dd_trials, power='theta', mode=m, xr=xr, region=brain_region, 
                ymin=None, ymax=None, smoothing_triangle=5, encoding_mode=encoding_mode, behav_key='correct')
        





