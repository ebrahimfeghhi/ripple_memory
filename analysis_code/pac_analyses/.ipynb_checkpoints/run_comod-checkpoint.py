import numpy as np
import sys 
sys.path.append('/home1/efeghhi/ripple_memory/analysis_code/')
from load_data import *
from analyze_data import *
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import decimate, resample

from comodulogram import save_MI_amplitude, remove_session_string
import numpy as np
from load_data_numpy import load_data_np


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
    start_roi = 200
    end_roi = 900
    
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

# convert to indices based on start time, sampling rate factor, and time region of interest
start_idx = int((start_roi - start_time)/sr_factor)
end_idx = int((end_roi-start_time)/sr_factor)

print(start_idx, end_idx)

raw_data = dd_trials['raw']
clust = dd_trials['clust_int']
correct = dd_trials['correct']

print("RAW DATA SHAPE: ", raw_data.shape)

nonclust_idxs = np.argwhere(clust==0)
clust_idxs = np.argwhere(clust==1)
incorrect_idxs = np.argwhere(correct==0)

subj_elec_labels = np.array([remove_session_string(x) for x in dd_trials['elec_labels']])

subj_elec_clust = subj_elec_labels[clust_idxs].squeeze()
subj_elec_nonclust = subj_elec_labels[nonclust_idxs].squeeze()
subj_elec_incorrect = subj_elec_labels[incorrect_idxs].squeeze()

subj_elec_min_trials_condition = dict(np.load("/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/subj_elec_min_trials_condition.npz"))

raw_data_clust = raw_data[clust_idxs].squeeze()
raw_data_not_clust = raw_data[nonclust_idxs].squeeze()
raw_data_incorrect = raw_data[incorrect_idxs].squeeze()
    
savePath = '/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/saved_results/comodulogram/'
low_fq_range = [[2,4],[7,9]]
high_fq_range = [[30,70], [80,120], [130,170]]
fs = 500

behavorial_mode = int(sys.argv[1])

if behavorial_mode == 0:
    print("RUNNING CLUSTERED")
    save_MI_amplitude(subj_elec_clust, raw_data_clust, 'clust', fs, savePath, 
                  low_fq_range, high_fq_range, start_idx, end_idx, subj_elec_min_trials_condition, 
                     match_trial_count=False)
if behavorial_mode == 1:
    print("RUNNING NOT CLUSTERED")
    save_MI_amplitude(subj_elec_nonclust, raw_data_not_clust, 'not_clust', fs, savePath, 
                      low_fq_range, high_fq_range, start_idx, end_idx, subj_elec_min_trials_condition, 
                     match_trial_count=False)
if behavorial_mode == 2:
    print("RUNNING NOT CORRECT")
    save_MI_amplitude(subj_elec_incorrect, raw_data_incorrect, 'not_recalled', fs, savePath, 
                       low_fq_range, high_fq_range, start_idx, end_idx, subj_elec_min_trials_condition)
if behavorial_mode == 3:
    print("RUNNING ALL")
    save_MI_amplitude(subj_elec_labels, raw_data, 'all_data', fs, savePath, 
                      low_fq_range, high_fq_range, start_idx, end_idx, subj_elec_min_trials_condition, 
                      match_trial_count=False)