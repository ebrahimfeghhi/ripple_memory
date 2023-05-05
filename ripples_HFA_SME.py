import pandas as pd; pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)
import numpy as np
from cmlreaders import CMLReader, get_data_index
from ptsa.data.filters import ButterworthFilter, ResampleFilter, MorletWaveletFilter
import xarray as xarray
import sys
import os
import matplotlib.pyplot as plt
from pylab import *
from copy import copy
from scipy import stats
from scipy.stats import zscore
import seaborn as sns
import pickle
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42 # fix fonts for Illustrator
sys.path.append('/home1/john/johnModules')
from brain_labels import HPC_labels, ENT_labels, PHC_labels, temporal_lobe_labels,\
                         MFG_labels, IFG_labels, nonHPC_MTL_labels
from general import *
from SWRmodule import *
import statsmodels.formula.api as smf
from ripples_HFA_analysis import ripple_HFA_analysis

class ripple_analysis_SME(ripple_HFA_analysis):
    
    def __init__(self, exp, df, sub_selection, data_folder, hpc_ripple_type, select_subfield, hpc_regions,
                 ripple_bin_start_end=[100,1700], HFA_bins=[400,1100], regions_selected=['ca1']):
        
        super().__init__(exp=exp, df=df, sub_selection=sub_selection, data_folder=data_folder, hpc_ripple_type=hpc_ripple_type, 
                        select_subfield=select_subfield, hpc_regions=hpc_regions, ripple_bin_start_end=ripple_bin_start_end, 
                        HFA_bins=HFA_bins, regions_selected = regions_selected)
        
    def load_data_from_cluster(self, selected_period, region_name='HPC'):
        
        super().load_data_from_cluster(selected_period, region_name)
        
    def plot_SME_HFA(self, mode, title_str, savePath):

        '''
        :param int mode: 0 for all using all HFA activity, 1 for only ripples, and 2 for only non-ripples
        '''
        
        pad = int(np.floor(self.smoothing_triangle/2)) 
        
        HFA_array, word_correct_array, subject_name_array, session_name_array = super().separate_HFA_by_ripple(mode)
        
        plot_ME_mean = 1 # 0 for typical PSTH, 1 for ME mean, 2 for average across sub averages

        # set up the PVTH stats parameters here too (for encoding have 30 bins)
        psth_start = int(self.pre_encoding_time/self.bin_size)
        psth_end = int(self.encoding_time/self.bin_size)

        bin_centers = np.arange(psth_start+0.5,psth_end)
        xr = bin_centers #np.arange(psth_start,psth_end,binsize)

        # get vectors of encoding list identifier data for forgotten and recalled words
        # in encoded_word_key_array, 0 for not recalled, 1 for recalled, 2 for recalled but was an IRI<2 (don't care about that for encoding)
        start_array_enc_forgot = HFA_array[word_correct_array==0]
        start_array_enc_recalled = HFA_array[word_correct_array==1]

        # same for sub and sess
        sub_forgot = subject_name_array[word_correct_array==0]
        sess_forgot = session_name_array[word_correct_array==0]
        sub_recalled = subject_name_array[word_correct_array==1]
        sess_recalled = session_name_array[word_correct_array==1]

        # record min and max value before separating into recalled and not recalled in order to write text
        PSTH_all = triangleSmooth(np.mean(HFA_array,0),self.smoothing_triangle)
        min_val_PSTH = np.min(PSTH_all)
        max_val_PSTH = np.max(PSTH_all)

        # for recalled and then forgotten words
        for category in range(2):
            if category == 0:
                temp_start_array = start_array_enc_recalled
                sub_name_array = sub_recalled
                sess_name_array = sess_recalled

                # HFA 
                PSTH = triangleSmooth(np.mean(temp_start_array,0),self.smoothing_triangle)
                subplots(1,1,figsize=(5,4))
                plot_color = (0,0,1)
                num_words = f"Recalled: {temp_start_array.shape[0]}"

            else:       

                temp_start_array = start_array_enc_forgot
                sub_name_array = sub_forgot
                sess_name_array = sess_forgot
                
                PSTH = triangleSmooth(np.mean(temp_start_array,0),self.smoothing_triangle)

                plot_color = (0,0,0)
                num_words = f"Not recalled: {temp_start_array.shape[0]}"

            # note that output is the net ± distance from mean
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")    
                mean_plot,SE_plot = getMixedEffectMeanSEs(temp_start_array,sub_name_array,sess_name_array)

            if plot_ME_mean == 1:
                PSTH = triangleSmooth(mean_plot,self.smoothing_triangle) # replace PSTH with means from ME model (after smoothing as usual)  
            elif plot_ME_mean == 2:
                temp_means = []
                for sub in np.unique(sub_name_array):
                    temp_means = superVstack(temp_means,np.mean(temp_start_array[np.array(sub_name_array)==sub],0))
                PSTH = triangleSmooth(np.mean(temp_means,0),self.smoothing_triangle)
                SE_sub_level = np.std(temp_means,0)/np.sqrt(len(temp_means))     
            
            ## plot ##
            
            xr = bin_centers #np.arange(psth_start,psth_end,binsize)
            if pad > 0:
                xr = xr[pad:-pad]
                binned_start_array = temp_start_array[:,pad:-pad] # remove edge bins    
                PSTH = PSTH[pad:-pad]
                SE_plot = SE_plot[:,pad:-pad]        
            
            plot(xr,PSTH,color=plot_color, label=num_words)
            fill_between(xr, PSTH-SE_plot[0,:], PSTH+SE_plot[0,:], alpha = 0.3)
            xticks(np.arange(self.pre_encoding_time+pad*100,self.encoding_time-pad*100+1,500)/100,
                np.arange((self.pre_encoding_time+pad*100)/1000,(self.encoding_time-pad*100)/1000+1,500/1000))
            xlabel('Time from word presentation (s)')
            ylabel('HFA activity (z-scored)')
            title(title_str)
            tight_layout()
            ax = plt.gca()
            if min_val_PSTH < -1.0:
                lower_val = math.floor(min_val_PSTH)
            else:
                lower_val = -1.0
            if max_val_PSTH > 1.0:
                upper_val = math.ceil(max_val_PSTH)
            else:
                upper_val = 1.0
            ax.set_ylim(lower_val, upper_val)
            ax.set_xlim(self.pre_encoding_time/100,self.encoding_time/100)
            plot([0,0],[ax.get_ylim()[0],ax.get_ylim()[1]],linewidth=1,linestyle='-',color=(0,0,0))
            plot([1600,1600],[ax.get_ylim()[0],ax.get_ylim()[1]],linewidth=1,linestyle='--',color=(0.7,0.7,0.7))
            legend()

        plt.savefig(savePath, dpi=300)

    def ripple_recall(self):

        wca = self.word_correct_array
        sess_name_array = self.session_name_array
        subj_name_array = self.subject_name_array
        HFA = self.HFA_array 

        # ripple_exists array is a binary vector of shape num_events, with 1 indicating presence of a ripple
        ripple_idxs, _ = super().ripple_idxs_func()
        ripple_exists = np.zeros_like(wca)
        ripple_exists[ripple_idxs] = 1
        assert np.sum(ripple_exists) == ripple_idxs.shape[0], print("Ripple_exists and ripple_idxs not matching.")
        
         # run mixed effects model 
        vc = {'session':'0+session'}

        SE_df = pd.DataFrame(data={'session':sess_name_array,'subject':subj_name_array,'ripple_exists':ripple_exists, 
                                    'word_recalled': wca})

        get_bin_CI_model = smf.mixedlm("word_recalled ~ ripple_exists", SE_df, groups="subject", vc_formula=vc, 
                                        re_formula='ripple_exists')

        bin_model = get_bin_CI_model.fit(reml=True, method='nm',maxiter=2000)

         # run OLS model for verification 
        get_bin_CI_model_ols = smf.ols("word_recalled~ripple_exists", SE_df)
        bin_model_ols = get_bin_CI_model_ols.fit()

        return bin_model, bin_model_ols
        
    def SME_ripple_interaction(self):
        
        wca = self.word_correct_array
        sess_name_array = self.session_name_array
        subj_name_array = self.subject_name_array
        HFA = self.HFA_array 

        # ripple_exists array is a binary vector of shape num_events, with 1 indicating presence of a ripple
        ripple_idxs, _ = super().ripple_idxs_func()
        ripple_exists = np.zeros_like(wca)
        ripple_exists[ripple_idxs] = 1
        assert np.sum(ripple_exists) == ripple_idxs.shape[0], print("Something is not right with the ripple shapes.")

        HFA_bins = [(self.HFA_bins[0]-self.pre_encoding_time)/self.bin_size, (self.HFA_bins[1]-self.pre_encoding_time)/self.bin_size]
        HFA_time_range = np.arange(HFA_bins[0], HFA_bins[1]+1) # 11 to 18, corresponding to 400-1100 ms post word onset
        HFA_time_restricted = HFA[:, int(HFA_time_range[0]):int(HFA_time_range[-1])]
        HFA_mean = np.mean(HFA_time_restricted,axis=1)

        # run mixed effects model 
        vc = {'session':'0+session'}

        SE_df = pd.DataFrame(data={'session':sess_name_array,'subject':subj_name_array,'ripple_exists':ripple_exists, 
                                    'word_recalled': wca, 'HFA_mean': HFA_mean})
        get_bin_CI_model = smf.mixedlm("HFA_mean ~ ripple_exists*word_recalled", SE_df, groups="subject", vc_formula=vc, 
                                        re_formula='ripple_exists*word_recalled')
        bin_model = get_bin_CI_model.fit(reml=True, method='nm',maxiter=2000)

        # run OLS model for verification 
        get_bin_CI_model_ols = smf.ols("HFA_mean ~ ripple_exists*word_recalled", SE_df)
        bin_model_ols = get_bin_CI_model_ols.fit()

        return bin_model, bin_model_ols
        
    def save_ripple_information(self, savePath):
        
        pd.DataFrame(self.analysis_information, index=[0]).to_csv(savePath)