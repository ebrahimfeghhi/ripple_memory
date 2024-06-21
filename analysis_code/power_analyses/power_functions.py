import numpy as np
import sys 
base = '/home1/efeghhi/'
sys.path.append(f'{base}ripple_memory/analysis_code/pac_analyses/')
from load_data_numpy import load_data_np
from comodulogram import remove_session_string, get_filtered_signal

def z_score(power):
    
    '''
    :param ndarray power: 2d array of shape trials x timesteps
    '''
    # mean center by mean across time and trials 
    # then divide by standard deviation of the average across timesteps
    power = (power - np.mean(power)) / np.std(np.mean(power, axis=1),axis=0)
    return power

def process_power(power, zscore_by_idxs):
    
    from scipy.signal import decimate
    
    power_ds = decimate(np.log10(power), 10)
    
    power_ds_zscored = np.zeros_like(power_ds)
    
    for val in np.unique(zscore_by_idxs):
        
        val_idxs = np.argwhere(zscore_by_idxs==val).squeeze()
            
        for freq in range(power_ds.shape[0]):
            
            power_ds_zscored[freq, val_idxs] = z_score(power_ds[freq, val_idxs].squeeze())
        
    return power_ds_zscored

def load_z_scored_power(dd_trials, freq_range_str_arr, encoding_mode, high_fq_range=[[30,70], [80,120], [130, 170]], 
                       low_fq_range=[[2,4],[7,9]]):
    
    print("Generating figures for run_mode: ", encoding_mode)

    if encoding_mode:
        start_cutoff = 500
        end_cutoff = 2000

    else:
        start_cutoff = 500
        end_cutoff = 2500

    subjects = dd_trials['subj']

    fs = 500
    sr_factor = 1000/fs

    raw_data = dd_trials['raw']
    
    print(raw_data.shape)
    
    # z scoring is done for each unique subject, session, electrode combo
    subj_elec_sess_labels = dd_trials['elec_labels'] 

    for freq_range_str in freq_range_str_arr:
        
        if freq_range_str == 'high':
            freq_range = high_fq_range
            ylabel = 'Gamma power'
            bandwidth='auto'
        else:
            freq_range = low_fq_range
            ylabel = 'Theta power'
            bandwidth='auto'
        
        # filter signal using hilbert method
        filtered_sig = get_filtered_signal(raw_data, freq_range, start_cutoff, 
                                           end_cutoff, fs, bandwidth=bandwidth)
        
        # obtain power and amplitude
        filtered_sig_amp = np.real(np.abs(filtered_sig))
        filtered_sig_power = filtered_sig_amp**2
        
        # z-score data from each electrode 
        power_z = process_power(filtered_sig_power, subj_elec_sess_labels)
        
        power_z = power_z.squeeze()
        
        yield power_z, ylabel
        
    