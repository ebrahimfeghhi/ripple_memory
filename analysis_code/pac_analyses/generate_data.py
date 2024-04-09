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
        catFR_dir = '/scratch/efeghhi/catFR1/IRIonly/'
        
    data_dict = load_data(directory=catFR_dir, region_name=region_name, 
                          encoding_mode=encoding_mode)

    if encoding_mode: 
        data_dict = remove_wrong_length_lists(data_dict)
        
    # ca1
    ca1_elecs = [x for x in HPC_labels if 'ca1' in x]
    data_dict_ca1 = select_region(data_dict, ca1_elecs)
    count_num_trials(data_dict_ca1, "ca1")

    data_dict_region = data_dict_ca1

    # create clustered int array
    clustered_int = create_semantic_clustered_array(data_dict_region, encoding_mode)
    data_dict_region['clust_int'] = clustered_int

    dd_trials = dict_to_numpy(data_dict_region, order='C')
    
    return dd_trials


encoding_mode = 0 # 1 for encoding, 0 for recall


dd_trials = load_data_np(0)
breakpoint()

raw_data = dd_trials['raw']


dd_trials_updated = {'clust_int': dd_trials['clust_int'], 
                    'subj': dd_trials['subj'], 'elec_names': dd_trials['elec_names'], 
                    'elec_labels': dd_trials['elec_labels'], 
                     'elec_ripple_rate_array': dd_trials['elec_ripple_rate_array'], 
                    'elec_by_elec_correlation': dd_trials['elec_by_elec_correlation'], 
                     'trial_by_trial_correlation': dd_trials['trial_by_trial_correlation']}

if encoding_mode:
    
    dd_trials_updated['correct'] = dd_trials['correct']
    
del dd_trials

low_gamma_bp = mne.filter.filter_data(raw_data, sfreq=500, l_freq=30, h_freq=75)
high_gamma_bp = mne.filter.filter_data(raw_data, sfreq=500, l_freq=80, h_freq=178)
analytic_signal_lg = hilbert(low_gamma_bp)
analytic_signal_hg = hilbert(high_gamma_bp)
low_gamma_power = (np.abs(analytic_signal_lg)**2)[:, 500:-500]
high_gamma_power = (np.abs(analytic_signal_hg)**2)[:, 500:-500]

theta_bp = mne.filter.filter_data(raw_data, sfreq=500, l_freq=5, h_freq=8)
analytic_signal_theta = hilbert(theta_bp)
theta_phase = np.angle(analytic_signal_theta)[:, 500:-500]

dd_trials_updated['low_gamma'] = low_gamma_power
dd_trials_updated['high_gamma'] = high_gamma_power
dd_trials_updated['theta'] = theta_phase 

if encoding_mode:
    np.savez('updated_data/dd_trials_encoding', **dd_trials_updated)
else:
    np.savez('updated_data/dd_trials_recall', **dd_trials_updated)




