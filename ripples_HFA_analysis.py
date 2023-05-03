import pandas as pd; pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)
import numpy as np
import xarray as xarray
import sys
import os
import matplotlib.pyplot as plt
from pylab import *
from copy import copy
import pickle
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42 # fix fonts for Illustrator
sys.path.append('/home1/john/johnModules')
from general import *
from SWRmodule import *
import statsmodels.formula.api as smf

class ripple_HFA_analysis():
    
    '''
    General class for loading data + other helpful functions. 
    '''

    def __init__(self, exp, df, sub_selection, data_folder, hpc_ripple_type, regions_selected=['ca1', 'dg'],
                            select_subfield=True, hpc_regions=['ca1'], ripple_bin_start_end=[100,1700], HFA_bins=[400,1100], sr_factor=2.0
                            , pre_encoding_time=-700, encoding_time=2300, bin_size=100,
                            smoothing_triangle=5, samples=100, sr=500):
        
        '''
        :param str exp: catFR1 or FR1
        :param pandas DataFrame df: subject data
        :param str sub_selection: whole, first_half, or second_half
        :param str data_folder: SWR_scratch or SWR_semantic_scratch
        :param str hpc_ripple_type: only applies when region name from load data cluster is hpc
            single_elec loads ripples from each electrode in regions_selected individually
            avg_elec ripple value for an electrode will be 1 anytime a ripple occurs in an ipsilateral electrode
            in regions_selected
        :param list regions_selected: which regions in hpc to look for ripples, applies for any brain region
        :param bool selected_subfield: if true, only analyze data from hpc_regions when region name is set to hpc
        :param list hpc_regions: which regions in hpc to analyze, only applies when analyzing hpc
        :param list ripple_bin_start_end: timerange used for detecting presence of ripple
        :param list HFA_bins: timerange used for analyzing HFA 
        :param float sr_factor: sampling rate factor 
        :param int pre_encoding_time: msec before word presentation that recording was started
        :param int encoding_time: msec after word presentation that recording extends to 
        :param int bin_size: bin size used to create histograms to assess signficance for SME and SCE 
        :param int smoothing_triangle: parameter for smoothing signals 
        :param int samples: not sure what this is or why it's used
        :param int sr: number of samples taken per sec
        '''

        self.exp = exp
        self.df = df 
        self.sub_selection = sub_selection
        self.data_folder = data_folder 
        self.hpc_ripple_type = hpc_ripple_type
        self.ripple_bin_start_end = ripple_bin_start_end
        self.HFA_bins = HFA_bins 
        self.sr_factor = sr_factor
        self.pre_encoding_time = pre_encoding_time
        self.encoding_time = encoding_time
        self.bin_size = bin_size
        self.smoothing_triangle = smoothing_triangle
        self.samples = samples 
        self.regions_selected = regions_selected
        self.hpc_regions = hpc_regions
        self.select_subfield = select_subfield
        self.skipped_files = 0
        self.sr = sr 
        self.ripple_start_marker = int((self.ripple_bin_start_end[0] - self.pre_encoding_time) / self.sr_factor)
        self.ripple_end_marker = int((self.encoding_time - self.ripple_bin_start_end[1]) / self.sr_factor)
        self.analysis_information = {}
        self.ripple_bin_duration = (ripple_bin_start_end[1] - ripple_bin_start_end[0]) / self.sr_factor
        self.HFA_bin_duration = (HFA_bins[1] - HFA_bins[0]) / self.sr_factor
        
        # There are minor differences between loading the semantic and non semantic data 
        if 'semantic' in self.data_folder:
            self.semantic_data = True
        else:
            self.semantic_data = False

        if self.exp == 'catFR1':
            self.list_length = 12
        elif self.exp == 'FR1':
            self.list_length = 12

    def remove_subject_sessions(self):
        
        # 575 FR sessions. first 18 of don't load so skip those 
        exp_df = self.df[self.df.experiment==self.exp]
        if self.exp == 'FR1':
            exp_df = exp_df[
                            ((self.df.subject!='R1015J') | (self.df.session!=0)) & 
                            ((self.df.subject!='R1063C') | (self.df.session!=1)) & 
                            ((self.df.subject!='R1093J') | (~self.df.session.isin([1,2]))) &
                            ((self.df.subject!='R1100D') | (~self.df.session.isin([0,1,2]))) &
                            ((self.df.subject!='R1120E') | (self.df.session!=0)) &
                            ((self.df.subject!='R1122E') | (self.df.session!=2)) &
                            ((self.df.subject!='R1154D') | (self.df.session!=0)) &
                            ((self.df.subject!='R1186P') | (self.df.session!=0)) &
                            ((self.df.subject!='R1201P') | (~self.df.session.isin([0,1]))) &
                            ((self.df.subject!='R1216E') | (~self.df.session.isin([0,1,2]))) &
                            ((self.df.subject!='R1277J') | (self.df.session!=0)) &
                            ((self.df.subject!='R1413D') | (self.df.session!=0)) & 
                            ((self.df.subject!='R1123C') | (self.df.session!=2)) & # artifacts that bleed through channels (see SWR FR1 prob sessions ppt)
                            ((self.df.subject!='R1151E') | (~self.df.session.isin([1,2]))) & # more bleed-through artifacts (see same ppt)
                            ((self.df.subject!='R1275D') | (self.df.session!=3))  # 3rd session an actual repeat of 2nd session (Paul should have removed from database by now)
            #                 (self.df.subject!='R1065J') # sub with 9000 trials
                        ] 
        elif self.exp == 'catFR1': 
            exp_df = exp_df[
                            ((self.df.subject!='R1044J') | (self.df.session!=0)) & # too few trials to do pg pairwise corr
                            ((self.df.subject!='R1491T') | (~self.df.session.isin([1,3,5]))) & # too few trials to do pg pairwise corr
                            ((self.df.subject!='R1486J') | (~self.df.session.isin([4,5,6,7]))) & # repeated data...will be removed at some point... @@
                            ((self.df.subject!='R1501J') | (~self.df.session.isin([0,1,2,3,4,5]))) & # these weren't catFR1 (and they don't load right anyway)
                            ((self.df.subject!='R1235E') | (self.df.session!=0)) & # split EEG filenames error...documented on Asana
                            ((self.df.subject!='R1310J') | (self.df.session!=1)) & # session 1 is just a repeat of session 0
                            ((self.df.subject!='R1239E') | (self.df.session!=0)) # some correlated noise (can see in catFR1 problem sessions ppt)
            ]
        elif self.exp == 'RepFR1':
            exp_df = exp_df[
                            (self.df.subject!='R1564J') # clearly something wrong with these EEG when looking at ripple raster
                            ]
        self.exp_df = exp_df


    def load_data_from_cluster(self, selected_period, region_name='HPC', filter_type='hamming', remove_soz_ictal=0, recall_type_switch=0):

        '''
        :param str selected_period: input one of the following options
            'surrounding_recall': aligned to time of free recall 
            'whole_retrieval': aligned to beginning of retrieval period (beep_off)
            'encoding': aligned to word_on
            'whole_encoding': aligned to 1st word of each encoding period and ends 29.7 s later (average time for 12 words to be shown)
                       NOTE: this analysis is in SWRanalysis-encoding.ipynb now
            'math': aligned to math problem on
            'math_retrieval': aligned to math problem key-in time
        :param str region_name: input can be ENT, HPC, HPC_ENT ENT, HPC, PHC, TEMPORALLOBE, IFG, MFG, ENTPHC, AMY
        :param str remove_soz_ictal: input 0 for nothing, 1 for remove SOZ, 2 for keep ONLY SOZ ###
        :param str filter_type: input can be butter/hamming/hamming125200/tried hamming140250 for math
        :param bool hpc_ripples: if true, only load sessions which have both HPC data and region name data, and use ripples from HPC 
        '''

        self.filter_type = filter_type
        self.loaded_files = 0 
        self.attempted_loads = 0 
        self.region_name = region_name

        recall_minimum = 2000 # used if recall_type_switch = 3

        # get strings for path name for save and loading cluster data
        if recall_type_switch in [0,4,6,8]:
            # for these I'm using all trials, but selecting for which recall after the fact
            soz_label,recall_selection_name,subfolder = getSWRpathInfo(remove_soz_ictal,0,selected_period, recall_minimum)
        else: # these others I haven't set up indexing (see >line 100 in this cell)
            soz_label,recall_selection_name,subfolder = getSWRpathInfo(remove_soz_ictal,recall_type_switch,selected_period, recall_minimum)
            
        ripple_array = [];  HFA_array = []
        trial_nums = [];  encoded_word_key_array = []
        HPC_names = []; sub_sess_names = []; 
        region_electrode_ct = []; sub_names = []
        trial_by_trial_correlation = []; elec_ripple_rate_array = []
        elec_by_elec_correlation = []; fr_array = []
        list_num_key = []

        serialpos_array = []; list_recall_num_array = []; 
        rectime_array = []; recall_before_intrusion_array = []
        recall_position_array = []; session_events = pd.DataFrame()

        electrode_labels = []; channel_coords = []; channel_nums = []

        analysis_df = getSplitDF(self.exp_df, self.sub_selection, self.exp)
        
        semantic_clustering_key = []; temporal_clustering_key = []

        category_array = []
        
        self.skipped_files_no_hpc_ripples = 0
        self.skipped_files_no_trials = 0
        self.skipped_files_unequal_trials = 0
        self.skipped_files_ipsilateral = 0
        self.missing_data = []

        for p, row in enumerate(analysis_df.itertuples()): 
            
            self.attempted_loads += 1
            try:
                
                sub = row.subject; session = row.session; exp = row.experiment
                path_name = f'/scratch/john/{self.data_folder}/{subfolder}'

                # load data from HPC so that we can condition on HPC ripples from CA1 + DG from the ipsilateral hemisphere
                fn_HPC = os.path.join(path_name,
                'SWR_'+exp+'_'+sub+'_'+str(session)+'_HPC_'+selected_period+recall_selection_name+
                            '_'+soz_label+'_'+filter_type+'.p')

                with open(fn_HPC, 'rb') as fh:
                    
                    dat_HPC = pickle.load(fh)
                    ripple_array_hpc = dat_HPC['ripple_array']
                    trial_nums_hpc =  dat_HPC['trial_nums']
                    HPC_names_single_file = dat_HPC['HPC_names'] 
                    
                # skip rest of loop if there are no hpc ripples
                if np.array(ripple_array_hpc).shape[0] == 0 or np.sum(ripple_array_hpc)==0:
                    self.skipped_files_no_hpc_ripples  += 1
                    continue
                
                fn = os.path.join(path_name,
                    'SWR_'+exp+'_'+sub+'_'+str(session)+'_'+region_name+'_'+selected_period+recall_selection_name+
                                '_'+soz_label+'_'+filter_type+'.p') 
                
                self.loaded_files += 1
                
                with open(fn,'rb') as f:
                    
                    dat = pickle.load(f)
                    trial_nums_single_file = dat['trial_nums']
                    area_2_region_names_single_file = dat['HPC_names']
                
                    if len(trial_nums_hpc) == 0 or len(trial_nums_single_file) == 0:
                        self.skipped_files_no_trials += 1
                        continue
                    
                    if trial_nums_hpc[0] != trial_nums_single_file[0]:
                        self.skipped_files_unequal_trials  += 1
                        continue
                    
                    hpc_ripples_ipsilateral = self.check_hpc_ripples(area_2_region_names_single_file, HPC_names_single_file)
                    
                    if hpc_ripples_ipsilateral==False:
                        self.skipped_files_ipsilateral += 1
                        continue 
                    
                    trial_nums = np.append(trial_nums,dat['trial_nums'])
                    HFA_array = superVstack(HFA_array,dat['HFA_array'])
                    region_electrode_ct.append(dat['region_electrode_ct'])
                    encoded_word_key_array.extend(dat['encoded_word_key_array'])
                    HPC_names.extend(dat['HPC_names']) # EF1208, too lazy to change this rn but it shouldn't be HPC names (something like region names)
                    sub_sess_names.extend(dat['sub_sess_names'])
                    sub_names.extend(dat['sub_names'])

                    # check to see the number of trials
                    trial_by_trial_correlation.extend(dat['trial_by_trial_correlation']) # one value for each electrode for this session
                    elec_by_elec_correlation = np.append(elec_by_elec_correlation,dat['elec_by_elec_correlation'])
                    elec_ripple_rate_array.extend(dat['elec_ripple_rate_array']) # ripple rate by electrode so append
                
                    serialpos_array.extend(dat['serialpos_array'])
                    recall_position_array.extend(dat['recall_position_array']) # 1-indexed
                    list_recall_num_array.extend(dat['list_recall_num_array'])
                    session_events = session_events.append(dat['session_events']) # doesn't append in place 
                        
                    electrode_labels.extend(dat['electrode_labels'])
                    channel_coords.extend(dat['channel_coords'])
                    
                    if not self.semantic_data: 
                        channel_nums.extend(dat['channel_nums'])
                    else:
                        if self.exp == 'catFR1':
                            category_array.extend(dat['category_array'])
                            semantic_clustering_key.extend(dat['semantic_clustering_key'])
                        elif self.exp == 'FR1':
                            temporal_clustering_key.extend(dat['temporal_clustering_key'])
                            
                    list_num_key.extend(dat['list_num_key'])
             
                    
                # load ripples from HPC 
                ripples_single_file = self.ripples_hpc(trial_nums_single_file, area_2_region_names_single_file, trial_nums_hpc, 
                                                HPC_names_single_file, ripple_array_hpc)
                                   
                ripple_array = superVstack(ripple_array, ripples_single_file)
                
            except Exception as e:

                LogDFExceptionLine(row, e, 'ClusterLoadSWR_log.txt')  

        print('**Done reading data**')

        subject_name_array,session_name_array,electrode_array,channel_coords_array,channel_nums_array = getSubSessPredictorsWithChannelNums(
                sub_names,sub_sess_names,trial_nums,electrode_labels,channel_coords,channel_nums)
    
        word_correct_array = []
        for sess_elec in encoded_word_key_array:
            if self.semantic_data:
                word_correct_array.extend(sess_elec)
            else:
                word_correct_array.append(sess_elec)
        word_correct_array = np.array(word_correct_array)
        word_correct_array[word_correct_array>0] = 1 # 1s and 2s are corrects
        
        sp_array = []
        for sp in serialpos_array:
            sp_array.extend(sp)
        serialpos_array = sp_array
        
        temp = []
        for enc in encoded_word_key_array:
            temp.extend(enc)
        encoded_word_key_array = copy(temp)

        if not self.semantic_data:
            self.channel_nums_array = np.array(channel_nums_array)
        
        region_electrode_ct = np.array(region_electrode_ct)
        
        print('From '+str(sum(region_electrode_ct>0))+'/'+str(len(region_electrode_ct))+' sessions with >0 '+region_name+' electrodes')
        print('Total trials: '+str(int(np.sum(trial_nums))))
        print('Unique sessions: '+str(len(np.unique(sub_sess_names))))
        print('...from '+str(len(np.unique(subject_name_array)))+' patients')

        # some info about regions 
        sub_elec = [subject_name_array[i]+electrode_array[i] for i in range(len(electrode_array))]
        print('Number of electrodes: '+str(len(np.unique(sub_elec))))

        print('Electrode regions X sessions:')
        unique_names = np.unique(HPC_names)
        for name in unique_names:
            num_elecs = sum(np.array([names.find(name) for names in HPC_names])>=0)
            print(str(num_elecs)+' for '+name)

        # save data to class 
        self.HFA_array = HFA_array
        self.ripple_array = ripple_array
        self.trial_nums = trial_nums
        self.sub_sess_names = sub_sess_names
        self.HPC_names = HPC_names
        self.recall_before_intrusion_array = recall_before_intrusion_array
        self.rectime_array = rectime_array
        self.list_recall_num_array = list_recall_num_array
        self.selected_period = selected_period
        self.sub_names = sub_names
        self.serialpos_array = serialpos_array
        self.region_name = region_name
        self.subject_name_array = subject_name_array
        self.session_name_array = session_name_array
        self.electrode_array = electrode_array
        self.channel_coords_array = channel_coords_array
        self.word_correct_array = word_correct_array
        self.category_array = category_array
        self.temporal_clustering_key = temporal_clustering_key
        self.semantic_clustering_key = semantic_clustering_key
        self.list_num_key = list_num_key
        self.encoded_word_key_array = encoded_word_key_array
        self.recall_position_array = recall_position_array
        self.session_events = session_events
        
        self.clean_up_ripples()
        
    def select_idxs_numpy(self):
            
        if self.select_subfield and self.region_name == 'HPC':
            
            selected_recalls = np.zeros(len(self.start_array), dtype=bool)
            
            location_names = []
            
            # location names will be a list of length num trials
            # where every entry is the HPC region the trial is recorded from 
            for s in range(len(self.HPC_names)):
                selected_trials = int(self.trial_nums[s])
                location_names.extend(np.tile(self.HPC_names[s],selected_trials))  
                
            location_names_np = np.asarray(location_names)
            
            for s in self.hpc_regions:
                left_match_idxs = np.argwhere(location_names_np==f'left {s}')
                right_match_idxs = np.argwhere(location_names_np==f'right {s}')
                matching_idxs = np.concatenate((left_match_idxs, right_match_idxs), axis=0)
                selected_recalls[matching_idxs] = 1
                
        else:
            # otherwise, select all regions 
            selected_recalls = np.ones(len(self.start_array), dtype=bool)
            
        if self.semantic_data:
            # from john 
            selected_recalls[(np.array(self.session_name_array)=='R1180C-2') & (np.array(self.list_num_key)==24)] = 0
            selected_recalls[(np.array(self.session_name_array)=='R1278E-10') & (np.array(self.list_num_key)==25)] = 0
            
        print(f"Selecting {np.sum(selected_recalls)} trials out of {selected_recalls.shape[0]} trials")
        self.num_selected_trials = np.sum(selected_recalls)
            
        self.session_names_np = np.array(self.session_name_array)[selected_recalls]
        self.electrode_array_np = np.array(self.electrode_array)[selected_recalls]
        self.recall_position_np = np.array(self.recall_position_array)[selected_recalls]
        self.start_array_np = self.start_array[selected_recalls]
        self.encoded_word_key_array_np = np.array(self.encoded_word_key_array)[selected_recalls]
        self.serialpos_array_np = np.array(self.serialpos_array)[selected_recalls]
        self.list_num_key_np = np.array(self.list_num_key)[selected_recalls]
        self.word_correct_array_np = self.word_correct_array[selected_recalls]
        self.session_events_np = self.session_events[selected_recalls]
        self.HFA_array_np = self.HFA_array[selected_recalls]
        self.session_name_array_np = np.array(self.session_name_array)[selected_recalls]
        self.subject_name_array_np = np.array(self.subject_name_array)[selected_recalls]
        self.ripple_array_np = np.array(self.ripple_array)[selected_recalls]
        
        if self.semantic_data:
            if self.exp == 'FR1':
                self.temporal_array_np = np.array(self.temporal_clustering_key)[selected_recalls]
            if self.exp == 'catFR1':
                self.semantic_array_np = np.array(self.semantic_clustering_key)[selected_recalls]
                self.cat_array_np = np.array(self.category_array)[selected_recalls]
            
        
    def check_hpc_ripples(self, area2_names, hpc_names):
        
        # check to make sure there are electrodes on the same hemisphere for HPC and area 2
        elec_left_area2 = False
        elec_right_area2 = False
        elec_left_hpc = False
        elec_right_hpc = False
        for region in area2_names:
            if 'left' in region:
                elec_left_area2 = True
            if 'right' in region:
                elec_right_area2 = True
        for region in hpc_names:
            for rs in self.regions_selected:
                if f'left {rs}' in region:
                    elec_left_hpc = True
                if f'right {rs}' in region:
                    elec_right_hpc = True
                
        if (elec_left_area2==True and elec_left_hpc==True) or (elec_right_area2==True and elec_right_hpc==True):
            hpc_ripples_ipsilateral = True
        else:
            hpc_ripples_ipsilateral = False
            
        return hpc_ripples_ipsilateral
        

    def ripples_hpc(self, area2_trialnums, area2_regions, hpc_trialnums, hpc_regions, ripple_array_hpc):

        '''
        Note: area 2 can be any brain region, including HPC 

        Returns a ripple array that is of shape (num area2 trials x timesteps). At each timestep, 
        there will be a 1 if a ripple occurred in selected_regions on the ipsilateral side 
        as the electrode that the trial corresponds to, and a 0 otherwise. 
        
        If region_name is hpc and hpc_ripple_type is set to 'single_elec', then simply return ripple_array_hpc

        :param list area2_trialnums: number of trials collected from each electrode in area 2
        :param list area2_regions: region within area2 that the electrode is placed in 
        :param list hpc_trialnums: number of trials collected from each electrode in hpc 
        :param list hpc_regions: regions of hpc to search for ripples
        :param numpy array ripple_array_hpc: contains ripples from hpc, of shape num_trials x timesteps  
        '''
        
        # don't do any averaging across electrodes 
        if self.region_name == 'HPC' and self.hpc_ripple_type == 'single_elec':
            return ripple_array_hpc

        timesteps = ripple_array_hpc.shape[1] # timesteps over which recording is taken 
        num_trials = np.sum(area2_trialnums) # number of trials collected from area2 
        
        # Step 1: Separate hpc right and left ripples
        # only select ripples from self.regions_selected 
        left_hemi_data = []
        right_hemi_data = []
        trialStart = 0
        for region, trials in zip(hpc_regions, hpc_trialnums):
            trials = int(trials)
            for rs in self.regions_selected:
                if f'right {rs}' in region: 
                    right_hemi_data.append(ripple_array_hpc[trialStart:trialStart+trials])
                if f'left {rs}' in region: 
                    left_hemi_data.append(ripple_array_hpc[trialStart:trialStart+trials])
            trialStart += trials 
            
        # describing procedure for left_hemi_data, but it is the exact same for right_hemi_data
        
        # left_hemi_data is a list where each entry is of size trials x timesteps
        # each entry contains ripples from that hemisphere for a hpc region specified in 
        # self.regions_selected
        
        # np.stack(left_hemi_data) results in an array of shape N x trials x timesteps
        # where N is the number of electrodes in regions_selected in that hemisphere
        # summing across the first axis then gives us a trials x timesteps array. 
        # where each value in the array is the number of ripples that occurred at that time 
        # across the N electrodes. Because we don't care if multiple ripples occurred at the same 
        # time, we clip this array to have a max value of 1. 
        # If there are no ripples in a hemi, then fill an array of that shape with all zeros. 
        
        if len(right_hemi_data) > 0:
            rhd_np = np.clip(np.sum(np.stack(right_hemi_data),axis=0), a_min=0, a_max=1)
        else:
            rhd_np = np.zeros((trials, timesteps))
        if len(left_hemi_data) > 0:
            lhd_np = np.clip(np.sum(np.stack(left_hemi_data),axis=0), a_min=0, a_max=1)
        else:
            lhd_np = np.zeros((trials, timesteps))

        # fill an array of the same shape as the nonhpc ripples with the averaged hpc ripples
        # from the correct hemisphere. 
        ripples_hpc_area2shape = np.zeros((int(num_trials), timesteps))

        trialStart = 0
        for region, trials in zip(area2_regions, area2_trialnums):
            
            trials = int(trials)
            if 'right' in region:
                ripples_hpc_area2shape[trialStart:trialStart+trials] = rhd_np
            if 'left' in region:
                ripples_hpc_area2shape[trialStart:trialStart+trials] = lhd_np
            
            trialStart += trials 
            
        return ripples_hpc_area2shape
    
    def clean_up_ripples(self):
        
        # Not sure what to do with 0.5 values right now, so going to remove them.
        self.ripple_array= np.where(self.ripple_array==0.5, 0, self.ripple_array)
        
        # select ripples only in time range of interest
        self.ripple_array = self.ripple_array[:, self.ripple_start_marker:-self.ripple_end_marker]
        self.analysis_information['ripple_timesteps'] = self.ripple_array.shape[0]

    def getStartArray(self):
        
        start_array,end_array = getStartEndArrays(self.ripple_array)
        self.start_array = start_array
        self.end_array = end_array  
        
        self.analysis_information['num_ripples'] = sum(self.start_array)
        self.analysis_information['num_trials'] = self.start_array.shape[0]

    def ripple_idxs_func(self):

        '''
        Returns idxs of ripples and no ripples from ripple_array
        '''

        # sum across rows to find rows where a ripple occurs
        ripple_array_rowsum = np.sum(self.ripple_array_np,axis=1)
        # if the row consists of all zeros, no ripple occurred, so set value to False
        # otherwise a ripple occurred, so set the value to True 
        ripple_bool = np.where(ripple_array_rowsum==0, False, True) 
        ripple_idxs = np.squeeze(np.argwhere(ripple_bool==True))
        non_ripple_idxs = np.squeeze(np.argwhere(ripple_bool==False))
        
        self.analysis_information['num_trials_with_ripple'] = ripple_idxs.shape[0]
        self.analysis_information['num_trials_without_ripple'] = non_ripple_idxs.shape[0]
        
        return ripple_idxs, non_ripple_idxs
    
    def separate_HFA_by_ripple(self, mode):

        if mode == 0:
            selected_idxs = np.arange(self.HFA_array_np.shape[0])
        if mode == 1: 
            selected_idxs, _ = self.ripple_idxs_func()
        if mode == 2: 
            _, selected_idxs = self.ripple_idxs_func()
            
        HFA_array = self.HFA_array_np[selected_idxs]
        word_correct_array = self.word_correct_array_np[selected_idxs]
        subject_name_array = self.subject_name_array_np[selected_idxs]
        session_name_array = self.session_name_array_np[selected_idxs]

        return HFA_array, word_correct_array, subject_name_array, session_name_array
