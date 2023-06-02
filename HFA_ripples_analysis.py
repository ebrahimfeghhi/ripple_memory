
import pandas as pd; pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)
import numpy as np
from cmlreaders import CMLReader, get_data_index
from ptsa.data.filters import ButterworthFilter, ResampleFilter, MorletWaveletFilter
import xarray as xarray
import sys
import os
import matplotlib.pyplot as plt
from pylab import *
from scipy.stats import zscore
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42 # fix fonts for Illustrator
sys.path.append('/home1/john/johnModules')
from brain_labels import HPC_labels, ENT_labels, PHC_labels, temporal_lobe_labels,\
                         MFG_labels, IFG_labels, nonHPC_MTL_labels
from general import *
from SWRmodule import *
import statsmodels.formula.api as smf
from ripples_HFA_general import HFA_ripples_prepare_data
from sklearn.metrics import r2_score


class HFA_ripples_analysis(HFA_ripples_prepare_data):
    
    def __init__(self, exp, df, sub_selection, data_folder, hpc_ripple_type, select_subfield, hpc_regions,
                 ripple_bin_start_end=[100,1700], HFA_bins=[400,1100], ripple_regions=['ca1']):

        super().__init__(exp=exp, df=df, sub_selection=sub_selection, data_folder=data_folder, hpc_ripple_type=hpc_ripple_type, 
                        select_subfield=select_subfield, hpc_regions=hpc_regions, ripple_bin_start_end=ripple_bin_start_end, 
                        HFA_bins=HFA_bins, ripple_regions=ripple_regions)
        
        self.psth_start = self.pre_encoding_time
        
    def remove_subject_sessions(self):
        
        super().remove_subject_sessions()
    
    def load_data_from_cluster(self, selected_period, region_name='HPC'):
        
        super().load_data_from_cluster(selected_period, region_name)
        
    def getStartArray(self):
        
        super().getStartArray()
        
    def semantic_clustering(self, ripple_delta, HFA_delta):
        
        '''
        (Based on my current understanding)
        
        Input:
        
            :param bool rippleDelta: if True, use start ripple times to compute sess_delta
            :param bool hfa_delta: if True, use hfa array to compute sess_delta
        
        Output: 
        
            numpy array sess_delta: mean difference between ripple rates or hfa 
            for clustered vs unclustered recalled
         
        '''
        
        self.ripple_delta = ripple_delta
        self.HFA_delta = HFA_delta

        assert ripple_delta != HFA_delta, print("Boolean arguments should not be equal")
        
        self.num_sessions = 0 
        
        self.counter_delta = 0 
        
        # select which serialpositions you're looking at (since curious if 1-6 show all the SCE)
        serialpos_select = np.arange(1,13) 

        # EF1208, what is this?
        remove_chaining = 0 # 2022-07-19 trying a control to see if SCE still exists after removing recalls that begin with SP 1+2 in a row
        
        # these values are all for subject-level SCE v. avg_recalls analysis
        if self.sub_selection == 'whole':
            min_SCE_trials = 20 # minimum SCE trials in session to include in SCE v. avg_recalls plot
        else:
            min_SCE_trials = 10
            
        stats_bin = self.ripple_bin_start_end[1]-self.ripple_bin_start_end[0] # only using 1 bin for encoding 

        self.adj_semantic_encoding_array = []
        self.rem_semantic_encoding_array = []
        self.rem_unclustered_encoding_array = []
        self.last_recall_encoding_array = [] # the last word remembered on each list (no transitions)...but make sure it's not an intrusion or repeat too!
        self.forgot_encoding_array = []
        self.sub_name_array0 = []; self.sess_name_array0 = []; self.elec_name_array0 = []
        self.sub_name_array1 = []; self.sess_name_array1 = []; self.elec_name_array1 = []
        self.sub_name_array2 = []; self.sess_name_array2 = []; self.elec_name_array2 = []
        self.sub_name_array3 = []; self.sess_name_array3 = []; self.elec_name_array3 = []
        self.sess_name_array4 = [] # forgot why I keep the others but leaving them 2022-06-10
        self.sub_name_array5 = []; self.sess_name_array5 = []; self.elec_name_array5 = []

        # for clustered v. unclustered subject-level analysis (need to record at session-level though for mixed model)
        self.sess_sessions = []
        self.sess_delta = []
        self.sess_subjects = []
        self.sess_recall_num = []
        self.sess_clust_num = []
        self.sess_prop_semantic = []
        
        session_names = np.unique(self.session_name_array)
        
        # EF1208, converting to numpy for parallel indexing 
        for sess in session_names:
            
            self.num_sessions += 1
            
            # for each session will get a clustered and unclustered a) ripple start array, or b) HFA_array
            clustered_data = []; unclustered_data = []
            
            # and also the proportion of semantically clustered recalls
            temp_corr = []; temp_sem_key = []
            
            # Number of lists for a given session 
            # EF1208, on AMY first half I'm seeing a session with 10 lists, is that normal?
            sess_list_nums = np.unique(self.list_num_key_np[self.session_names_np==sess]) 
            
            # loop through each list in the session 
            for ln in sess_list_nums:
                
                # obtain electrodes corresponding to the selected list
                list_elec_array = np.unique(self.electrode_array_np[(self.session_names_np==sess) & (self.list_num_key_np==ln)])
                
                for elec in list_elec_array:
    
                    # boolean array, with the number of True elements equal to the list length (12)
                    list_ch_idxs = (self.session_names_np==sess) & (self.list_num_key_np==ln) & (self.electrode_array_np==elec) 
                    
                    if ripple_delta:
                        list_ch_encoding_array = self.start_array_np[list_ch_idxs] # ripple start times for the selected list 
                        single_event_time = self.ripple_bin_duration
                    elif HFA_delta:
                        list_ch_encoding_array = self.HFA_array_np[list_ch_idxs]
                        single_event_time = self.HFA_bin_duration
                        
                    list_ch_cats = self.cat_array_np[list_ch_idxs] # semantic category of presented words
                    list_ch_corr = self.word_correct_array_np[list_ch_idxs] # binary array, whether or not the word was correctly recalled 
                    list_ch_semantic_key = self.semantic_array_np[list_ch_idxs] # list of lists containing recall (A, C, D, Z) for each recalled word
                    list_ch_recall_positions = self.recall_position_np[list_ch_idxs] # list containing encoded position of recalled words
                    
                    # remove ones starting with serialpos 1->2 as a control (or just 1 if it's len 1)
                    if remove_chaining == 1:
                        if len(list_ch_recall_positions[0])==1:
                            if list_ch_recall_positions[0][0]==1: # if 1st serialpos
                                continue # get out of this loop if only one recall and it's serialpos 1
                        elif len(list_ch_recall_positions[0])>0:
                            if ((list_ch_recall_positions[0][0]==1)&(list_ch_recall_positions[0][1]==2)):     
                                continue # get out of loop if recalls are serialpos 1->2 (no matter what)            

                    for i_recall_type, recall_type in enumerate(list_ch_semantic_key[0]): # all 12 lists have same values so just take 1st one
                        recall_position = list_ch_recall_positions[0][i_recall_type] # ditto re: taking 1st
                        if recall_position in serialpos_select: 
                            if recall_type == 'A': # adjacent semantic and adjacent in time 
        #                     if recall_type in ['A','C']: # adjacent AND remote semantic
                                # note the -1 since recall positions are on scale of 1-12
                                self.adj_semantic_encoding_array = superVstack(self.adj_semantic_encoding_array, list_ch_encoding_array[recall_position-1])
                                self.sub_name_array0.append(sess[0:6])
                                self.sess_name_array0.append(sess)
                                self.elec_name_array0.append(elec)
                            elif recall_type == 'C': # remote semantic, remote in time but from the same semantic category 
                                self.rem_semantic_encoding_array = superVstack(self.rem_semantic_encoding_array,list_ch_encoding_array[recall_position-1])
                                self.sub_name_array1.append(sess[0:6])
                                self.sess_name_array1.append(sess)
                                self.elec_name_array1.append(elec)
                            elif ( (recall_type == 'D') ): # & (recall_position>0) ): # remote unclustered
                                self.rem_unclustered_encoding_array = superVstack(self.rem_unclustered_encoding_array,list_ch_encoding_array[recall_position-1])
                                self.sub_name_array2.append(sess[0:6])
                                self.sess_name_array2.append(sess)  
                                self.elec_name_array2.append(elec)
                            elif ( (recall_type == 'Z') ): #& (recall_position>0) ): # last word of list & was actually a recalled word
                                self.last_recall_encoding_array = superVstack(self.last_recall_encoding_array,list_ch_encoding_array[recall_position-1])
                                self.sub_name_array3.append(sess[0:6])
                                self.sess_name_array3.append(sess)
                                self.elec_name_array3.append(elec)
                            else:
                                self.sess_name_array4.append(sess[0:6])
                                
                        # Creating clustered vs unclustered conditioned 
                        # start_arracyC will be of shape N x ripple_start:ripple_end, where N is the number of clustered recalls
                        # and ripple_start:ripple_end is the timestep range of interest for ripples
                        # same goes for start_arrayU, except N is the number of unclustered recalled 
                        if recall_position in serialpos_select: # so can select by serialpos (e.g. 1:6 or 7:12)
                            if recall_type in ['A','C']: # adjacent semantic or remote semantic
                                # note the -1 since recall positions are on scale of 1-12
                                clustered_data = superVstack(clustered_data,list_ch_encoding_array[recall_position-1])
                            elif ( (recall_type in ['D','Z']) & (recall_position>0) ): # remote unclustered or dead end (>0 means recalled word)
                                unclustered_data = superVstack(unclustered_data,list_ch_encoding_array[recall_position-1])
                                
                                
                    # unpack semantic clustering key to trial level (only need to do once for one electrode)
                    if elec == list_elec_array[0]:
                        for word in range(sum(list_ch_idxs)): 
                            if (word+1) in list_ch_recall_positions[0]: # serial positions are 1-indexed so add 1 to check in list_ch_recall_positions
                                temp_corr.append(1)
                                # use index from serialpos to get clustering classification
                                if ((sess== 'R1108J-2')&(ln==25)): # single mistake showss up
                                    if word == 8:
                                        temp_sem_key.append('A')
                                    elif word == 9:
                                        temp_sem_key.append('Z')
                                else: 
                                    temp_sem_key.append(list_ch_semantic_key[0][list_ch_recall_positions[0].index(word+1)])
                            else:
                                temp_corr.append(0)
                                temp_sem_key.append('')               
                                
                    # make forgotten array to plot along with SCE too which is easy enough 
                    forgotten_words = 1-np.array(list_ch_corr)
                    if sum(forgotten_words)>0: # R1065 a whiz
                        self.forgot_encoding_array = superVstack(self.forgot_encoding_array,np.array(list_ch_encoding_array)[findInd(forgotten_words),:])
                        self.sub_name_array5.extend(np.tile(sess[0:6],int(sum(forgotten_words))))
                        self.sess_name_array5.extend(np.tile(sess,int(sum(forgotten_words))))
                        self.elec_name_array5.extend(np.tile(elec,int(sum(forgotten_words))))
                        
                        
            if ((len(clustered_data)>min_SCE_trials) & (len(unclustered_data)>min_SCE_trials) & \
                (len(clustered_data)!=single_event_time) ): # last one in there for a len(1) start_arrayC

                # back at session-level record the delta, sub, sess, and avg_recall_num for *all* trials
                self.sess_sessions.append(sess)
                self.sess_subjects.append(sess[0:6])   

                # can just use list_elec_array to select only one electrode we know exists for this session (altho should be irrelevant when we average anyway)
                # This is a binary array, where a 1 indicates correct recall of a word 
                sess_word_correct_array = self.word_correct_array_np[((self.electrode_array_np==list_elec_array[0]) & (self.session_name_array_np==sess))]
                
                
                self.sess_recall_num.append(self.list_length*(sum(sess_word_correct_array)/len(sess_word_correct_array)))
                
                self.sess_prop_semantic.append(sum([trial in ['A','C'] for trial in temp_sem_key])/sum(temp_corr))
                
                
                # create histogram for ripples/HFA associated with clustered + unclustered words
                binned_clustered_data = binBinaryArray(clustered_data,stats_bin,self.sr_factor)
                binned_unclustered_data = binBinaryArray(unclustered_data ,stats_bin,self.sr_factor)
                
                # take difference between histogram means 
                self.sess_delta.append(np.mean(binned_clustered_data)-np.mean(binned_unclustered_data)) 
      
        trial_nums = [len(self.sub_name_array0),len(self.sub_name_array1),len(self.sub_name_array2),len(self.sub_name_array3),len(self.sess_name_array4)]
        
        
    def create_clustered_array(self, clustered=['A,C']):
        
        '''
        :param list clustered: indicates what recalls count as clustered. There are four possible recalls:
            1) 'A': adjacent semantic
            2) 'C': remote semantic
            3) 'D': remote unclustered
            4) 'Z': dead end 
        '''
        self.number_of_lists = int(self.num_selected_trials/self.list_length)
        
        self.clustered_array_np = np.zeros((self.number_of_lists, self.list_length))
        self.dead_ends = 0
        self.remote_semantic = 0
        self.adjacent_semantic = 0
        self.remote_nonsemantic = 0
        self.adjacent_nonsemantic = 0
        
        counter = 0
        
        for list_idx in range(0, self.num_selected_trials, self.list_length):
            
            recalled_idx = self.recall_position_np[list_idx]
            cluster = self.semantic_array_np[list_idx]
            
            cluster_trial_np = np.zeros(self.list_length)
            
            for r, c in zip(recalled_idx, cluster):
               
                if r > 0 and r < self.list_length:
                    if c in clustered:
                        cluster_trial_np[r-1] = 1 # change to 1 for clustered word
                    if c=='A':
                        self.adjacent_semantic += 1
                    if c=='C':
                        self.remote_semantic += 1
                    if c=='D':
                        self.remote_nonsemantic += 1
                    if c=='Z':
                        self.dead_ends += 1
                    
            self.clustered_array_np[counter] = cluster_trial_np
            
            counter += 1
            
        self.clustered_array_np = np.ravel(self.clustered_array_np)
        
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
        
        if mode != 0:
            # load both ripple and no ripple HFA to make sure y axis is the same
            HFA_array_ripple, _, _, _ = super().separate_HFA_by_ripple(1)
            HFA_array_noripple, _, _, _ = super().separate_HFA_by_ripple(2)
            PSTH_all_r = triangleSmooth(np.mean(HFA_array_ripple,0),self.smoothing_triangle)
            PSTH_all_nr = triangleSmooth(np.mean(HFA_array_noripple,0),self.smoothing_triangle)
            min_val_PSTH_r = np.min(PSTH_all_r)
            max_val_PSTH_r = np.max(PSTH_all_r)
            min_val_PSTH_nr = np.min(PSTH_all_nr)
            max_val_PSTH_nr = np.max(PSTH_all_nr)
            max_val_PSTH = max(max_val_PSTH_r, max_val_PSTH_nr)
            min_val_PSTH = min(min_val_PSTH_r, min_val_PSTH_nr)
        else:
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
        
    def SME_ripple_interaction(self):
        
        # ripple_exists array is a binary vector of shape num_events, with 1 indicating presence of a ripple
        ripple_idxs, _ = super().ripple_idxs_func()
        ripple_exists = np.zeros_like(self.word_correct_array_np)
        ripple_exists[ripple_idxs] = 1
        assert np.sum(ripple_exists) == ripple_idxs.shape[0], print("Something is not right with the ripple shapes.")
        
        # run mixed effects model 
        vc = {'session':'0+session'}

        SE_df = pd.DataFrame(data={'session':self.session_name_array_np,'subject':self.subject_name_array_np,'ripple_exists':ripple_exists, 
                                    'word_recalled': self.word_correct_array_np, 'HFA_mean': self.HFA_mean})
        get_bin_CI_model = smf.mixedlm("word_recalled ~ ripple_exists*HFA_mean", SE_df, groups="subject", vc_formula=vc, 
                                        re_formula='ripple_exists*HFA_mean')
        bin_model = get_bin_CI_model.fit(reml=True, method='nm',maxiter=2000)
        
        # run OLS model for verification 
        get_bin_CI_model_ols = smf.ols("word_recalled ~ ripple_exists*HFA_mean", SE_df)
        bin_model_ols = get_bin_CI_model_ols.fit()
            
        return bin_model, bin_model_ols
        
    def save_model_info(self, model, savePath):
        
        params=['Intercept', 'ripple_exists', 'HFA_mean', 
                                    'ripple_exists:HFA_mean']
        
        param_dict = {}
        param_dict['name'] = []
        param_dict['coef'] = []
        param_dict['stderr'] = []
        param_dict['pval'] = []
        param_dict['tval'] = []
        
        for param in params:

            param_dict['name'].append(param)
            param_dict['coef'].append(model.params[param])
            param_dict['stderr'].append(model.bse[param])
            param_dict['pval'].append(model.pvalues[param])
            param_dict['tval'].append(model.tvalues[param])

        pd.DataFrame(param_dict).to_csv(savePath, index=False)  
        
    ''' 
    def recalled_data(self):
        
        recalled = self.word_correct_array_np
        
        self.ripple_exists_re = self.ripple_exists_np[recalled==1]
        self.no_ripple_exists_re = 1 - self.ripple_exists_re
        self.clustered_re = self.clustered_array_np[recalled==1].squeeze()
        self.HFA_re = self.HFA_mean[recalled==1].squeeze()
        self.sess_re= self.session_name_array_np[recalled==1].squeeze()
        self.subj_re = self.subject_name_array_np[recalled==1].squeeze()
        
    def lmm_model(self, dep_var_list, formula_list, re_formula_list, recalled_bool_list):
        
        
        data = pd.DataFrame(data={'clustered':self.clustered_array_np, 'recalled':self.word_correct_array_np, 
                                        'ripple_exists':self.ripple_exists_np, 'HFA_mean':self.HFA_mean, 
                                        'high_HFA_mean': self.high_HFA_mean, 'recalled_high_HFA':self.high_HFA_recalled,
                                        'subject':self.subject_name_array_np, 
                                        'session':self.session_name_array_np})
        
        data_recalled = pd.DataFrame(data={'clustered':self.clustered_re, 'ripple_exists':self.ripple_exists_re, 
                                           'HFA_mean':self.HFA_re, 'subject':self.subj_re, 
                                        'session':self.sess_re})
        
        num_models = len(dep_var_list)
        
        for i in range(num_models):
        
        if recalled:
            data_lmm = data_recalled
        else:
            data_lmm = data
            
        y = data_lmm[dep_var]
            
        vc = {'session':'0+session'}
        model = smf.mixedlm(formula, data_lmm, groups="subject", vc_formula=vc, 
                                        re_formula=f'{re_formula}')
        model_fit = model.fit(method=["lbfgs"])
        
        y_hat = model_fit.predict()
        print(f"R squared: {r2_score(y, y_hat)}")
        
        print(model_fit.summary())
        
        return model_fit
    '''
        
        
        
        
            
