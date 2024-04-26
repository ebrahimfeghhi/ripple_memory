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


def load_data_np(encoding_mode):
    
    print("Loading data")
    
    region_name = ['HPC']

    condition_on_ca1_ripples = False
    
    if encoding_mode: 
        catFR_dir = '/scratch/efeghhi/catFR1/ENCODING/'
    else:
        catFR_dir = '/scratch/efeghhi/catFR1/NOIRI/'
        
    data_dict, one_d_keys = load_data(directory=catFR_dir, region_name=region_name, 
                          encoding_mode=encoding_mode)

    if encoding_mode: 
        data_dict = remove_wrong_length_lists(data_dict, one_d_keys)
        
    # ca1
    ca1_elecs = [x for x in HPC_labels if 'ca1' in x]
    
    data_dict_ca1 = select_region(data_dict, ca1_elecs, one_d_keys)
    count_num_trials(data_dict_ca1, "ca1")

    data_dict_region = data_dict_ca1
    

    # create clustered int array
    clustered_int = create_semantic_clustered_array(data_dict_region, encoding_mode)
    data_dict_region['clust_int'] = clustered_int

    dd_trials = dict_to_numpy(data_dict_region, order='C')
    
    return dd_trials


