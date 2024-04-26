import numpy as np
import sys 
sys.path.append('/home1/efeghhi/ripple_memory/analysis_code/')
from load_data import *
from analyze_data import *
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import decimate, resample

from mne.time_frequency import tfr_array_morlet
from scipy.signal import hilbert
from load_data_numpy import load_data_np

from pactools import Comodulogram, REFERENCES
from pactools.comodulogram import read_comodulogram
from pactools import simulate_pac

print("Generating comodulogram")

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


raw_data = dd_trials['raw']
clust = dd_trials['clust_int']

nonclust_idxs = np.argwhere(clust==0)
clust_idxs = np.argwhere(clust==1)
elec_clust_idxs = dd_trials['elec_labels'][clust_idxs]
elec_nonclust_idxs = dd_trials['elec_labels'][nonclust_idxs]
raw_data_clust = raw_data[clust_idxs].squeeze()
raw_data_not_clust = raw_data[nonclust_idxs].squeeze()

low_fq_range = np.linspace(2, 8, 7)
high_fq_range = np.arange(30, 140, 10)
fs = 500
low_fq_width = 1.0  # Hz
progress_bar = True
method = 'tort'

print("Creating comodulogram for clustered data")
# only compute PAC within ROI
mask = np.ones_like(raw_data_clust)
mask[:, start_idx:end_idx] = 0
mask = mask.astype(bool)
mask = mask[:100]
raw_data_clust = raw_data_clust[:100]

estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range, high_fq_range=high_fq_range, 
                         low_fq_width=low_fq_width, method=method,
                         progress_bar=progress_bar, n_surrogates=100, minimum_shift=0.1)
estimator.fit(raw_data_clust, mask=mask)