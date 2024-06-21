import numpy as np

import sys 
sys.path.append('/home1/efeghhi/ripple_memory/analysis_code/')
from load_data import *
from analyze_data import *
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import resample

from mne.time_frequency import tfr_array_morlet
from scipy.signal import hilbert


def load_data_np(encoding_mode, region_name=['HPC'], subregion=['ca1']):
    
    print("Loading data")


    condition_on_ca1_ripples = False
    
    if encoding_mode: 
        catFR_dir = '/scratch/efeghhi/catFR1/ENCODING/'
    else:
        catFR_dir = '/scratch/efeghhi/catFR1/NOIRI/'
        
    print("LOADING DATA FROM: ", region_name[0])
        
    data_dict, one_d_keys = load_data(directory=catFR_dir, region_name=region_name, 
                          encoding_mode=encoding_mode)

    if encoding_mode: 
        data_dict = remove_wrong_length_lists(data_dict, one_d_keys)
        
    selected_elecs = []
    if region_name == ['HPC']:
        for s in subregion:
            selected_elecs_s = [x for x in HPC_labels if s in x]
            selected_elecs.extend(selected_elecs_s)
    
        data_dict = select_region(data_dict, selected_elecs, one_d_keys)

    # create clustered int array
    clustered_int = create_semantic_clustered_array(data_dict, encoding_mode)
    data_dict['clust_int'] = clustered_int

    dd_trials = dict_to_numpy(data_dict, order='C')
    
    return dd_trials


