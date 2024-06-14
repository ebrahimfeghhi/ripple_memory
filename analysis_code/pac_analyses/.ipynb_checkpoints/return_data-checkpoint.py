import numpy as np
import sys 
sys.path.append('/home1/efeghhi/ripple_memory/analysis_code/')
from load_data import *
from analyze_data import *
sys.path.append('/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/')
from load_data_numpy import load_data_np
from comodulogram import remove_session_string, get_filtered_signal

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import decimate, resample

from mne.time_frequency import tfr_array_morlet
from scipy.signal import hilbert


run_mode = 1

if run_mode == 1:
    encoding_mode = 1
    saveName = 'encoding_'
    
if run_mode == 2:
    encoding_mode = 0
    saveName = 'recall_'
    
dd_trials = load_data_np(encoding_mode)

if encoding_mode:
    # relative to word onset 
    start_roi = 300
    end_roi = 1300
    
    # each trial consists of 5 sec of raw data, which starts
    # 1.7 before word onset and ends 3.3 sec after word onset 
    # the data is sampled at 500 Hz
    start_time = -1700
    end_time = 3300

else:
    
    start_roi = -1100
    end_roi = -100
    
    # for recall, 
    # each trial consists of 6 sec of data, centered around word recall 
    start_time = -3000
    end_time = 3000
    
sr_factor = 2

# convert to indices based on start time and sampling rate factor
start_idx = int((start_roi - start_time)/sr_factor)
end_idx = int((end_roi-start_time)/sr_factor)

print(start_idx, end_idx)

raw_data = dd_trials['raw']
clust = dd_trials['clust_int']
correct = dd_trials['correct']


subj_elec_labels = np.array([remove_session_string(x) for x in dd_trials['elec_labels']])

return raw_data, clust, correct, subj_elec_labels
