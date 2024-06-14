%load_ext autoreload
%autoreload 2
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


region_name = ['AMY']
subregion = ['']
run_mode_arr = [2]
freq_range_str_arr = ['low', 'high']
savePath = '/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/saved_results/power_figures'


def z_score(power):
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
        
        if val_idxs.shape[0] < 10:
            continue
            
        for freq in range(power_ds.shape[0]):
            
            power_ds_zscored[freq, val_idxs] = z_score(power_ds[freq, val_idxs].squeeze())
        
    return power_ds_zscored
        
for run_mode in run_mode_arr:
    print("Generating figures for run_mode: ", run_mode)

    if run_mode == 1:
        encoding_mode = 1
        saveName = 'encoding_'

    if run_mode == 2:
        encoding_mode = 0
        saveName = 'recall_'

    dd_trials = load_data_np(encoding_mode, region_name=region_name, subregion=subregion)

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

    nonclust_idxs = np.argwhere(clust==0)
    clust_idxs = np.argwhere(clust==1)

    subj_elec_labels = np.array([remove_session_string(x) for x in dd_trials['elec_labels']])
    subj_elec_clust_idxs = subj_elec_labels[clust_idxs].squeeze()
    subj_elec_nonclust_idxs = subj_elec_labels[nonclust_idxs].squeeze()


    raw_data_clust = raw_data[clust_idxs].squeeze()
    raw_data_not_clust = raw_data[nonclust_idxs].squeeze()

    if run_mode == 1:
        correct = dd_trials['correct']
        incorrect_idxs = np.argwhere(correct==0)
        raw_data_incorrect = raw_data[incorrect_idxs].squeeze()
        subj_elec_incorrect_idxs = subj_elec_labels[incorrect_idxs].squeeze()


    savePath = '/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/saved_results/comodulogram/'
    low_fq_range = [[2,4],[7,9]]
    high_fq_range = [[30,70], [80,120], [130,170]]
    fs = 500

    subj_elec_sess_labels = dd_trials['elec_labels']
    sess_labels = dd_trials['sess']

    # encoding data is 3 seconds, but we save 5 seconds for bandpass filtering 
    if run_mode == 1:
        start_cutoff = 500
        end_cutoff = 2000
    # recall data is 4 seconds, but we save 6 seconds for bandpass filtering 
    if run_mode == 2:
        start_cutoff = 500
        end_cutoff = 2500
        
    for freq_range_str in freq_range:
        
        if freq_range_str == 'high':
            freq_range = high_fq_range
            ylabel = 'Gamma power'
            bandwidth=18.0
        else:
            freq_range = low_fq_range
            ylabel = 'Theta power'
            bandwidth=2.0

        filtered_sig = get_filtered_signal(raw_data, freq_range, start_cutoff, 
                                           end_cutoff, fs, bandwidth=bandwidth)

        filtered_sig_power = np.real(np.abs(filtered_sig))**2
        filtered_sig_amp = np.real(np.abs(filtered_sig))

        print(filtered_sig_power.shape)

        power_z = process_power(filtered_sig_power, subj_elec_labels)

        power_z_clust = power_z[:, clust_idxs].squeeze()
        power_z_not_clust = power_z[:, nonclust_idxs].squeeze()
        if run_mode == 1:
            power_z_incorrect = power_z[:, incorrect_idxs].squeeze()

        if run_mode == 2:
            recall_str = '_recall'
            time = np.linspace(-2, 2, 200)
        else:
            recall_str = ''
            time = np.linspace(-0.7, 2.3, 150)

        region_str = region_name[0]
        if len(subregion[0]) > 0:
            subregion_str = f'_{subregion[0]}'
        else:
            subregion_str = ''

        for i, f in enumerate(freq_range):

            pzc = power_z_clust[i]
            pznc = power_z_not_clust[i]

            plt.figure(figsize=(10,6))
            plt.errorbar(x=time, y=np.mean(pzc,axis=0), color='tab:blue', label="Clust")
            plt.errorbar(x=time, y=np.mean(pzc,axis=0), 
                        yerr=np.std(pzc,axis=0)/np.sqrt(pzc.shape[0]), color='tab:blue', alpha=0.4)


            plt.errorbar(x=time, y=np.mean(pznc,axis=0), color='tab:orange', label="Not clust")
            plt.errorbar(x=time, y=np.mean(pznc,axis=0), 
                         yerr=np.std(pznc,axis=0)/np.sqrt(pznc.shape[0]), alpha=0.4, color='tab:orange')

            if run_mode == 1:
                pzi = power_z_incorrect[i]
                plt.errorbar(x=time, y=np.mean(pzi,axis=0), color='tab:pink', label="Incorrect")
                plt.errorbar(x=time, y=np.mean(pzi,axis=0), 
                             yerr=np.std(pzi,axis=0)/np.sqrt(pzi.shape[0]), alpha=0.4, color='tab:pink')

            # Plot the 3 Hz wave
            #plt.plot(time, waveform, color='green', label="3 Hz Wave")

            plt.axvline(0, color='black')
            #plt.axvline(0.2, color='red')
            #plt.axvline(0.9, color='red')
            plt.ylabel(f"{ylabel} ({f[0]}-{f[1]} Hz)", fontsize=18)
            plt.xlabel("Time", fontsize=18)
            sns.despine()
            plt.legend(fontsize=16)
            plt.savefig(f"/{ylabel}_{f[0]}-{f[1]}_zscore_elec_{region_str}{subregion_str}{recall_str}", dpi=300, 
                        bbox_inches='tight')
            plt.show()
            
        if freq_range_str == 'low':
            from scipy.stats import ttest_rel

            start_idx = 450 # this corresponds to 200 ms after word onset 
            end_idx = 800 # this corresponds to 900 ms after word onset 
            end_base = 300  # at -100 ms
            start_base = 0 # at -700 ms
            print(start_base, end_base)

            subj_elec_unique, counts_se = np.unique(subj_elec_labels, return_counts=True)
            nonsig_elecs = 0
            total_elecs = 0
            nonsig_elecs_list = []
            plot_power_checks = False
            for seu in subj_elec_unique:
                total_elecs += 1
                se_idxs = np.argwhere(subj_elec_labels==seu)
                filtered_low_theta_se = filtered_sig_power[0, se_idxs].squeeze()
                low_pow_roi = np.max(np.log10(filtered_low_theta_se[:, start_idx:end_idx]), axis=1)
                low_pow_base = np.max(np.log10(filtered_low_theta_se[:, start_base:end_base]), axis=1)
                ttest_res = ttest_rel(low_pow_roi, low_pow_base, alternative='greater')
                if ttest_res.pvalue > 0.05:
                    if plot_power_checks:
                        plt.plot(np.linspace(-.7, 2.3, 1500), np.mean(10*np.log10(filtered_low_theta_se), axis=0))
                        #plt.plot(np.linspace(-.7, 2.3, 1500), 10*np.log10(np.mean(filtered_low_theta_se, axis=0)))
                        plt.axvline(-.7, color='red')
                        plt.axvline(-.1, color='red')
                        plt.axvline(0.2, color='black')
                        plt.axvline(0.9, color='black')
                        plt.show()
                    nonsig_elecs+=1
                    nonsig_elecs_list.append(seu)

                else:
                    if plot_power_checks:
                        plt.plot(np.linspace(-.7, 2.3, 1500), np.mean(10*np.log10(filtered_low_theta_se), axis=0))
                        #plt.plot(np.linspace(-.7, 2.3, 1500), 10*np.log10(np.mean(filtered_low_theta_se, axis=0)))
                        plt.axvline(-.7, color='red')
                        plt.axvline(-.1, color='red')
                        plt.axvline(0.2, color='black')
                        plt.axvline(0.9, color='black')
                        plt.show()


            print(nonsig_elecs, total_elecs)
            np.save(f'saved_results/comodulogram/elecs_with_no_low_theta_dg_{region_str}_{subregion_str}{recall_str}', nonsig_elecs_list)


