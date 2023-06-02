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
import pickle
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42 # fix fonts for Illustrator
sys.path.append('/home1/john/johnModules')
from brain_labels import HPC_labels, ENT_labels, PHC_labels, temporal_lobe_labels,\
                         MFG_labels, IFG_labels, nonHPC_MTL_labels
from general import *
from SWRmodule import *  

df = get_data_index("r1") # all RAM subjects
exp = 'catFR1' # 'FR1' 'catFR1' 'RepFR1'

# sub lists I've been using to explore FR1 

# subs = ['R1010J'] #'R1385E'] # ['R1065J'] #
# subs = ['R1002P','R1003P','R1006P','R1010J','R1112M','R1163T'] # initial 6 for hippocampus

# subs = subs+['R1001P','R1092J','R1151E','R1320D'] # additional subs with issues that I resolved with various RR and corr thresholds
# subs = ['R1112M','R1018P','R1020J','R1022J','R1023J','R1026D','R1027J'] # 7 with entorhinal
# subs = ['R1010J','R1112M'] # original 2 # R1108J beatiful catFR1 data; R1065J magical subject (like with FR1)
# subs = ['R1112M','R1163T'] # subjects with FR1 and catFR1. SRs of 1000 and 500 tho
# subs = ['R1151E'] # all channels X sessions get removed but 3 after 60/180 hz line removal
# these should all be worked out now. See SWR FR1 problem sessions PPT on Box for details of each
# subs = ['R1299T','R1332M','R1397D','R1349T','R1339D','R1337E','R1275D','R1151E','R1123C','R1120E','R1045E'] # final surrounding_recall problem subs after looking at huge raster!
# subs = ['R1308T','R1338T','R1358T'] # these guys had weird SRs and some loading problems when I went to whole_retrieval analysis...fixed with rounding
# subs = ['R1023J','R1101T','R1148P','R1368T','R1323T','R1334T'] # single session subs with memory allocation errors in ClusterRun
# subs = ['R1006P','R1010J','R1030J','R1032D','R1049J','R1051J','R1052E','R1054J','R1093J','R1098D','R1104D','R1108J','R1111M',
#         'R1115T','R1118N','R1124J','R1207J','R1230J','R1316T','R1329T','R1323T','R1337E','R1367D'] # subs that didn't load on 2020-07-04 encoding run
# subs = ['R1368T','R1461T','R1463E'] # subs with particularly low localization.pair matches in pairs...see Excel in loading info on Box for details
# subs = ['R1433E','R1355T','R1367D','R1368T'] # subs with "ca3" or "dg"
# subs = ['R1001P'] # subject with over 8000 FR1 trials X electrodes
# bad_subs = ['R1433E','R1051J'] # no electrode cats
# subs = ['R1379E','R1385E','R1387E','R1394E','R1402E'] # partial beep subs
# subs = ['R1379E','R1402E','R1396T','R1395M','R1415T','R1389J','R1404E']  # test subs for eeg offset correction
# subs = ['R1054J','R1345D','R1048E','R1328E','R1308T', # first 2 are sr ≥ 1000. 3rd is 500 Hz.
#         'R1137E','R1136N','R1094T','R1122E','R1385E', # nice example FR1 subs used in Fig. 2
#         'R1405E','R1486J','R1501J'] # adding in 3 catFR only patients that will go into Fig. 4
# subs = ['R1120E','R1349T','R1397D','R1332M','R1299T'] # FR1 patients with electrode search ranges limited per SWR problem sessions doc on Box
# subs = ['R1269E','R1328E','R1367D','R1397D','R1405E','R1405E','R1447M','R1469D'] # catFR1 patients with electrode search changes limited
# subs = ['R1065J','R1492J','R1525J'] # killer subs
# subs = ['R1030J','R1035M'] # MFG subs
subs = ['R1086M'] #'R1035M'] #['R1108J'] # R1065J # 'R1035M'
# subs = ['R1111M','R1108J','R1191J','R1229M','R1247P','R1264P','R1261P','R1016M','R1065J','R1191J',
#         'R1190P','R1254E','R1337E','R1118N','R1153T','R1156D'] # FR1 patients that are getting encoding memory errors 
# subs = ['R1051J','R1118N','R1154D','R1207J','R1308T','R1311T','R1329T','R1334T','R1336T','R1334T',
#         'R1342M','R1212P','R1346T','R1311T','R1323T','R1329T','R1342M','R1346T','R1367D','R1569T'] # catFR1 memory error patients

# subs = ['R1001P','R1002P','R1003P','R1006P','R1010J','R1018P','R1020J','R1022J', # mimicking test session
#         'R1023J','R1026D','R1027J','R1030J','R1031M','R1032D','R1033D','R1034D']
# subs = ['R1367D'] #['R1525J'] #['R1367D'] #R1065J'

sub_df = df[(df.subject.isin(subs))  & (df.experiment == exp)] # all sessions for subs
# sub_df = df[(df.subject.isin(subs))  & (df.experiment == exp) & (df.session==1)] # single session
# sub_df = df[(df.experiment == exp) & (df.session == 0)] # all FR subs 
# sub_df = sub_df[0:1]
sub_df

# 575 FR sessions. first 18 of don't load so skip those 
exp_df = df[df.experiment==exp]
if exp == 'FR1':
    exp_df = exp_df[
                    ((df.subject!='R1015J') | (df.session!=0)) & 
                    ((df.subject!='R1063C') | (df.session!=1)) & 
                    ((df.subject!='R1093J') | (~df.session.isin([1,2]))) &
                    ((df.subject!='R1100D') | (~df.session.isin([0,1,2]))) &
                    ((df.subject!='R1120E') | (df.session!=0)) &
                    ((df.subject!='R1122E') | (df.session!=2)) &
                    ((df.subject!='R1154D') | (df.session!=0)) &
                    ((df.subject!='R1186P') | (df.session!=0)) &
                    ((df.subject!='R1201P') | (~df.session.isin([0,1]))) &
                    ((df.subject!='R1216E') | (~df.session.isin([0,1,2]))) &
                    ((df.subject!='R1277J') | (df.session!=0)) &
                    ((df.subject!='R1413D') | (df.session!=0)) & 
                    ((df.subject!='R1123C') | (df.session!=2)) & # artifacts that bleed through channels (see SWR FR1 prob sessions ppt)
                    ((df.subject!='R1151E') | (~df.session.isin([1,2]))) & # more bleed-through artifacts (see same ppt)
                    ((df.subject!='R1275D') | (df.session!=3))  # 3rd session an actual repeat of 2nd session (Paul should have removed from database by now)
    #                 (df.subject!='R1065J') # sub with 9000 trials
                   ] 
elif exp == 'catFR1': 
    exp_df = exp_df[
                    ((df.subject!='R1044J') | (df.session!=0)) & # too few trials to do pg pairwise corr
                    ((df.subject!='R1491T') | (~df.session.isin([1,3,5]))) & # too few trials to do pg pairwise corr
                    ((df.subject!='R1486J') | (~df.session.isin([4,5,6,7]))) & # repeated data...will be removed at some point... @@
                    ((df.subject!='R1501J') | (~df.session.isin([0,1,2,3,4,5]))) & # these weren't catFR1 (and they don't load right anyway)
                    ((df.subject!='R1235E') | (df.session!=0)) & # split EEG filenames error...documented on Asana
                    ((df.subject!='R1310J') | (df.session!=1)) & # session 1 is just a repeat of session 0
                    ((df.subject!='R1239E') | (df.session!=0)) # some correlated noise (can see in catFR1 problem sessions ppt)
    ]
elif exp == 'RepFR1':
    exp_df = exp_df[
                    (df.subject!='R1564J') # clearly something wrong with these EEG when looking at ripple raster
                    ]
print(f"Experiment data frame shape: {exp_df.shape}")
# exp_df = exp_df[257:] # for catFR1 this is R1385E-onwwards
# exp_df = exp_df[472:] # for FR1 this is R1385E-onwwards
# exp_df = exp_df[468:] # for catFR1 this is R1525J-onwards
exp_df.head()

# Want to try and find those files that haven't been made yet (due to memory limits) and re-run only those
# **only an issue with encoding or whole_retrieval due to their large EEG matrices**

### params that clusterRun used
selected_period = 'encoding' # surrounding_recall # whole_retrieval # encoding 
recall_type_switch = 0 # 0 for original, 1 for only those with subsequent, 2 for second recalls only, 3 for isolated recalls
region_name = 'HPC' #'ENT' #'HPC' #HPC_ENT #ENT #HPC #ENTPHC #AMY
remove_soz_ictal = 0
recall_minimum = 2000
filter_type = 'hamming'
extra = '' #'_no_param_removal' #-intrusions #'-ZERO_IRI'
###

# get strings for path name for save and loading cluster data
soz_label,recall_selection_name,subfolder = getSWRpathInfo(remove_soz_ictal,recall_type_switch,selected_period,recall_minimum)

rerun_mask = []

for i,row in enumerate(exp_df.itertuples()):
    sub = row.subject; session = row.session; exp = row.experiment

    path_name = '/scratch/john/SWR_scratch/'+subfolder
    fn = os.path.join(path_name,
        'SWR_'+exp+'_'+sub+'_'+str(session)+'_'+region_name+'_'+selected_period+recall_selection_name+
                      '_'+soz_label+'_'+filter_type+extra+'.p') 
    try:
        with open(fn,'rb') as f:
            dat = pickle.load(f)
    except:
        rerun_mask.append(i)
        
# view the rerun_mask
len(rerun_mask)
rerun_df = exp_df.iloc[rerun_mask]
rerun_df.head()

## Now to load data from cluster, have to go through and append across sessions

### params that clusterRun used (note that exp is defined in first cell so can get exp_df above)

selected_period = 'surrounding_recall'
# 'surrounding_recall': aligned to time of free recall 
# 'whole_retrieval': aligned to beginning of retrieval period (beep_off)
# 'encoding': aligned to word_on
# 'whole_encoding': aligned to 1st word of each encoding period and ends 29.7 s later (average time for 12 words to be shown)
#               NOTE: this analysis is in SWRanalysis-encoding.ipynb now
# 'math': aligned to math problem on
# 'math_retrieval': aligned to math problem key-in time

recall_type_switch = 0
    # (1 and 3 are basically opposites...those with a subsequent recall and those without)
    # 0: Original analysis taking only recalls without a recall in 2 s IRI before them
    # 1: Take these same recalls, but keep only those WITH a recall within 2 s after they occur 
    # 2: test condition where we look at second recalls within IRI ONLY (there is an initial recall in 2 s before current recall)
    # 3: isolatead recalls with no other recalls +/- RECALL_MINIMUM s
    # 4: only first recall of every retrieval period
    # 5: take only those recalls that come second in retrieval period within 2 s of first retrieval
    # 6: take only NOT first recall of every retrieval period (opposite of 4)
    # 7: take only NOT first recall AND ISOLATED trials (this should REALLY maximize SWR bump)
    # 8: take only 2nd recalls
    # 10: same as 0 but with no IRI (mostly just to see number of recalls)
recall_minimum = 2000 # used if recall_type_switch = 3
region_name = 'HPC' #'ENT' #'HPC' #HPC_ENT #ENT #HPC # PHC # TEMPORALLOBE # IFG # MFG
remove_soz_ictal = 0 # 0 for nothing, 1 for remove SOZ, 2 for keep ONLY SOZ ###
filter_type = 'hamming' # butter/hamming/hamming125200/tried hamming140250 for math
sub_selection = 'whole' # 'second_half', 'whole' ,'first_half'
                              # analyze first 40%, remaining 60% of data, or whole? 
                              # works for FR1 and catFR1
extra = '' # _no_param_removal # -ORIGNORMAN # -intrusions # '-ZERO_IRI'
###

# get strings for path name for save and loading cluster data
if recall_type_switch in [0,4,6,8]:
    # for these I'm using all trials, but selecting for which recall after the fact
    soz_label,recall_selection_name,subfolder = getSWRpathInfo(remove_soz_ictal,0,selected_period,recall_minimum)
else: # these others I haven't set up indexing (see >line 100 in this cell)
    soz_label,recall_selection_name,subfolder = getSWRpathInfo(remove_soz_ictal,recall_type_switch,selected_period,recall_minimum)
    
ripple_array = []; HFA_array = []
trial_nums = []; encoded_word_key_array = []
HPC_names = []; sub_sess_names = []
region_electrode_ct = []; sub_names = []
trial_by_trial_correlation = []; elec_ripple_rate_array = []
elec_by_elec_correlation = []; fr_array = []
list_num_key = []

serialpos_array = []; list_recall_num_array = []; # ~~~
rectime_array = []; recall_before_intrusion_array = []
recall_position_array = []; session_events = pd.DataFrame()

electrode_labels = []; channel_coords = []; channel_nums = []

analysis_df = getSplitDF(exp_df,sub_selection,exp)

for row in analysis_df.itertuples(): #analysis_df.itertuples(): #sub_df.itertuples():  
    try:
        sub = row.subject; session = row.session; exp = row.experiment

        path_name = '/scratch/john/SWR_scratch/'+subfolder
        if filter_type == 'butter':
            subfolder = 'IRIonly 2022-03-04 zscore events only HFA (and all other old files)' # this has the 'butter'
        fn = os.path.join(path_name,
            'SWR_'+exp+'_'+sub+'_'+str(session)+'_'+region_name+'_'+selected_period+recall_selection_name+
                          '_'+soz_label+'_'+filter_type+extra+'.p') #'-NOCUTOFFS.p') #'_no_param_removal.p')   #'.p') #+'.intrusions.p') # +'.-wrong.p') (for wrong math)
                        # -NOCUTOFFS for Vaz filter for Norman/Staresina comparison
        with open(fn,'rb') as f:
            dat = pickle.load(f)

            ripple_array = superVstack(ripple_array,dat['ripple_array'])
#             HFA_array = superVstack(HFA_array,dat['HFA_array'])
            region_electrode_ct.append(dat['region_electrode_ct'])
            encoded_word_key_array.extend(dat['encoded_word_key_array'])
            HPC_names.extend(dat['HPC_names'])
            sub_sess_names.extend(dat['sub_sess_names'])
            sub_names.extend(dat['sub_names'])
            trial_nums = np.append(trial_nums,dat['trial_nums'])
            trial_by_trial_correlation.extend(dat['trial_by_trial_correlation']) # one value for each electrode for this session
            elec_by_elec_correlation = np.append(elec_by_elec_correlation,dat['elec_by_elec_correlation'])
            elec_ripple_rate_array.extend(dat['elec_ripple_rate_array']) # ripple rate by electrode so append
            #,'total_recalls':total_recalls, 'kept_recalls':kept_recalls}, f)
            if selected_period == 'whole_retrieval':
                if np.shape(dat['fr_array'])[0]!=np.shape(dat['ripple_array'])[0]:
                    print(sub+str(session))
                fr_array = superVstack(fr_array,dat['fr_array'])
            elif selected_period == 'encoding':
                serialpos_array.extend(dat['serialpos_array'])
                session_events = session_events.append(dat['session_events']) # doesn't append in place 
            elif selected_period == 'surrounding_recall': # ~~~
                serialpos_array.extend(dat['serialpos_array']); list_recall_num_array.extend(dat['list_recall_num_array']); # ~~
                rectime_array.extend(dat['rectime_array']); recall_before_intrusion_array.extend(dat['recall_before_intrusion_array'])
                recall_position_array.extend(dat['recall_position_array'])
                               
            elif (selected_period == 'math') | (selected_period == 'math_retrieval'):
                rectime_array.extend(dat['rectime_array'])
                recall_position_array.extend(dat['recall_position_array']); list_recall_num_array.extend(dat['list_recall_num_array'])
            elif selected_period == 'whole_encoding':
                serialpos_array.extend(dat['serialpos_array']); recall_position_array.extend(dat['recall_position_array'])
                list_recall_num_array.extend(dat['list_recall_num_array']); 

            electrode_labels.extend(dat['electrode_labels'])
            channel_coords.extend(dat['channel_coords'])
            channel_nums.extend(dat['channel_nums'])
            list_num_key.extend(dat['list_num_key'])
            
    except Exception as e:
        LogDFExceptionLine(row, e, 'ClusterLoadSWR_log.txt')  
print('**Done reading data**')
        
## loading *all* the recalls with 0, but if it's 4 or 6 load just those trials

# trying new method of loading...translate these to ripple_array length dependent on recall_type_switch 
# (this way I can always load from recall_type_switch = 0)

subject_name_array,session_name_array,electrode_array,channel_coords_array,channel_nums_array = getSubSessPredictorsWithChannelNums(
        sub_names,sub_sess_names,trial_nums,electrode_labels,channel_coords,channel_nums)

if selected_period == 'surrounding_recall':
    if recall_type_switch == 4:
        temp_recall_idxs = np.array(recall_position_array)==1
    elif recall_type_switch == 6:
        temp_recall_idxs = np.array(recall_position_array)>1
    elif recall_type_switch == 8:
        temp_recall_idxs = np.array(recall_position_array)==2
    else:
        temp_recall_idxs = np.array(recall_position_array)>=0
    serialpos_array = np.array(serialpos_array)[temp_recall_idxs]
    recall_before_intrusion_array = np.array(recall_before_intrusion_array)[temp_recall_idxs]
    list_num_key = np.array(list_num_key)[temp_recall_idxs]
elif (selected_period == 'math') | (selected_period == 'math_retrieval'):
    temp_recall_idxs = np.array(recall_position_array)>=0 # just keep them all for math
    encoded_word_key_array.extend(dat['encoded_word_key_array'])
elif selected_period == 'whole_encoding':
    temp_recall_idxs = np.array(list_recall_num_array)>=0 # just keep them all
elif selected_period == 'encoding':
    print('There is a separate program for loading encoding dumb guy!')
    session_events = session_events[temp_recall_idxs]
    
subject_name_array = np.array(subject_name_array)[temp_recall_idxs]
session_name_array = np.array(session_name_array)[temp_recall_idxs]
electrode_array = np.array(electrode_array)[temp_recall_idxs]
channel_coords_array = np.array(channel_coords_array)[temp_recall_idxs]
channel_nums_array = np.array(channel_nums_array)[temp_recall_idxs]
ripple_array = np.array(ripple_array)[temp_recall_idxs]
# HFA_array = np.array(HFA_array)[temp_recall_idxs]

rectime_array = np.array(rectime_array)[temp_recall_idxs]
list_recall_num_array = np.array(list_recall_num_array)[temp_recall_idxs]
recall_position_array = np.array(recall_position_array)[temp_recall_idxs]
    
print('**Done translating to ripple_array frame**!!')
print('...')
print('% of all HPC subjects for '+exp)
if exp == 'catFR1':
    len(np.unique(sub_names))/136*100 # % HPC subs for catFR1
    print('% of HPC recalls for '+exp)
    ripple_array.shape[0]/50053*100 # % recalls for catFR1
if exp == 'FR1':
    len(np.unique(sub_names))/167*100 # % HPC subs for FR1
    print('% of HPC recalls for '+exp)
    ripple_array.shape[0]/60417*100 # % recalls for FR1
    
## some info on data loaded from cluster runs ##
ripple_array.shape

region_electrode_ct = np.array(region_electrode_ct)
# print('Number of electrodes in each session: '); region_electrode_ct
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
import mne
from scipy.signal import firwin,filtfilt,kaiserord
import pingouin as pg

### PARAMS ###

save_values = 0

selected_period = 'encoding' 
# 'surrounding_recall': aligned to time of free recall 
# 'whole_retrieval': aligned to beginning of retrieval period (beep_off)
# 'encoding': aligned to word_on 
# 'whole_encoding': aligned to 1st word of each encoding period and ends 29.7 s later (average time for 12 words to be shown)
# 'math': aligned to math problem on
# 'math_retrieval': aligned to math problem key-in time

# there are three periods this code is set up to look at: periods aligned to recall, the entire retrieval period, and the encoding period
recall_type_switch = 0 # how do we select recalls?? Numbers 0:3
# 0: Original analysis taking only recalls without a recall in 2 s IRI before them
# 1: Take these same recalls, but keep only those WITH a recall within 2 s after they occur
# 2: test condition where we look at second recalls within IRI ONLY
# 3: ISOLATED only!
# 4: only first recall of every retrieval period
# 5: take only those recalls that come second in retrieval period within 2 s of first retrieval
# 6: take only NOT first recall of every retrieval period
# 7: take only NOT first recall AND ISOLATED trials (this should REALLY maximize SWR bump)
# 10: same as 0 but with no IRI (mostly just to see number of recalls)

selected_region = HPC_labels #HPC_labels # ENT_labels+HPC_labels

remove_soz_ictal = 0 # 0 for nothing, 1 for remove SOZ, 2 for keep ONLY SOZ ###

min_ripple_rate = 0.1 # Hz. # 0.1 for hamming
max_ripple_rate = 1.5 # Hz. # 1.5 for hamming
max_trial_by_trial_correlation = 0.05 # if ripples correlated more than this remove them # 0.05 for hamming
max_electrode_by_electrode_correlation = 0.2 #??? # 0.2 for hamming

filter_type = 'hamming' # see local version below for details 
# butter (Vaz algorithm)
# hamming (Norman algorithm)
# hamming125200 (Norman algorithm meant to capture "true" ripple frequency per Sullivan...Buzsaki 2011
# hamming140250 (Same idea, but higher bands)
# staresina (Staresina et al 2015 NatNeuro)

# Additional details: 
# •Vaz used Butterworth from 80-120 Hz; Norman used Hamming from 70-180 Hz
# •Butterworth from Vaz et al: 2nd order from 80-120 ms, hilbert, select events >2 STD above mean of filtered traces.
#      Events >=25 ms long w/ max amp >3 SD were retained as ripples. Join adjacent ripples separated by <15 ms 
# •Hilbert from Norman et al: "70-180 Hz zero-lag linear-phase Hamming FIR filter w/ transition bandwidth of 5 Hz"
#      Then Hilbert, clip extreme to 4 SD, square this clipped, smooth w/ Kaiser FIR low-pass filter with 40 Hz cutoff,
#      mean and SD computed across entire experimental duration to define the threshold for event detection
#      Events from original (squared but unclipped) signal >4 SD above baseline were selected as candidate SWR events. 
#      Duration expanded until ripple power <2 SD. Events <20 ms or >200 ms excluded. Adjacent events <30 ms separation (peak-to-peak) merged.
# •Using IED detection from Vaz for 250 Hz highpass events (really 250-499). Norman uses 20-60 Hz events which is...odd.
#       See detectRipples code in module for this algorithm


# recall params
recall_minimum = 2000
IRI = 2000 # inter-ripple interval...remove ripples within this range (keep only first one and remove those after it)
retrieval_whole_time = 30000
# encoding params
encoding_whole_time = 1000*np.round(12*(1.6+0.875),1) # 0.875 is average of 0.75-1.0 s so 29.7 is average encoding length
encoding_time = 2300 # actual preentation is 1.6 s + 0.75-1.0 s so keep +700 ms so can plot +500 ms
pre_encoding_time = -700 # since minimum ISI is 0.75 s let's only plot the 500 ms before word on with a 200 ms buffer
# these aren't likely to be changed:
desired_sample_rate = 500. # in Hz. This seems like lowerst common denominator recording freq.
eeg_buffer = 300 # buffer to add to either end of IRI when processing eeg #**

# soz_keep = [0,1] # 0 are good elecs and 1 are SOZ elecs. Never keep 2 (bad leads) ###
# if remove_soz_ictal == 1:
#     soz_keep = [0]
# elif remove_soz_ictal == 2:
#     soz_keep = [1]

### END PARAMS ###

# get region label
if selected_region == HPC_labels:
    region_name = 'HPC'
elif selected_region == ENT_labels:
    region_name = 'ENT'
elif selected_region == PHC_labels:
    region_name = 'PHC'
elif selected_region == temporal_lobe_labels:
    region_name = 'TEMPORALLOBE'
elif selected_region == MFG_labels:
    region_name = 'MFG'
elif selected_region == IFG_labels:
    region_name = 'IFG'
elif selected_region == nonHPC_MTL_labels:
    region_name = 'nonHPC_MTL'    

# analysis period
if selected_period == 'surrounding_recall':
    psth_start = -IRI # only makes sense to look at period <= IRI
    psth_end = IRI # how long to grab data after recall
elif selected_period == 'whole_retrieval':
    psth_start = -IRI # doesn't have to be IRI just 2000 ms is convenient
    psth_end = IRI+retrieval_whole_time
elif selected_period == 'whole_encoding':
    psth_start = -2000
    psth_end = encoding_whole_time    
elif selected_period == 'encoding':
    psth_start = pre_encoding_time
    psth_end = encoding_time
elif selected_period == 'whole_encoding':
    psth_start = -2000
    psth_end = 2000
elif (selected_period == 'math') | (selected_period == 'math_retrieval'): #$$
    psth_start = -2000 # just use 2000 since math problems are actually like 5 s apart typically
    psth_end = 2000   

ripple_array = []; fr_array = []; HFA_array = []
trial_nums = []; 
session_ct = 0; channel_ct = 0; total_channel_ct = 0
HPC_names = []; sub_names = []; sub_sess_names = []
electrodes_per_session = []
total_lists = 0; total_recalls = 0; kept_recalls = 0
align_adjust = 0
ent_elec_ct = []; sd_regions = []; not_sd_regions = []
ripple_ied_accum_ct = []
time_add_save = [];             
encoded_word_key_array = []
list_num_key = []

list_recall_num_array = []; rectime_array = []; recall_before_intrusion_array = [] # new ones added 2020-11-24
serialpos_array = [] # used to be encoding info but commandeered for surrounding_recalls ~~~
recall_position_array = []; recall_index_array = []
session_events = pd.DataFrame()

trial_by_trial_correlation = []; elec_by_elec_correlation = []
elec_ripple_rate_array = []

channel_coords = []; electrode_labels = []; channel_nums = []

for row in sub_df.itertuples(): #sub_df.itertuples():
    
    sub = row.subject; session = row.session; exp = row.experiment
    print(f"Working on data from subject {sub}")
    mont = int(row.montage); loc = int(row.localization)
    reader = CMLReadDFRow(row)
    breakpoint()
    evs = reader.load('task_events')
    # 1) change evs.recalled to False 2) comment out nonrepeat_indicator>0 for good_recalls = (or it removes all the -1s) 
    # 3) set getOutputPositions to [] 4) change save name at bottom to '_intrusions.p'
    if exp == 'RepFR1':
        evs_free_recall = evs[(evs.type=='REC_WORD') & (evs.intrusion==0)]
    else:
        evs_free_recall = evs[(evs.type=='REC_WORD') & (evs.recalled==True)] # recalled word AND correct (from this list...False for instrusions).        
    word_evs = evs[evs['type']=='WORD'] # get words 

    # select which recalls??         
    [recall_selection_name,selected_recalls_idxs] = selectRecallType(recall_type_switch,evs_free_recall,IRI,recall_minimum)
    
    _,nonrepeat_indicator = removeRepeatedRecalls(evs_free_recall,word_evs) # remove free recalls that happened more than once
    
    # let's make sure remaining recalls are not repeated, have eeg, and are not from non-practice lists (practice is -1 in FR1/catFR1 and 0 in RepFR1)
    good_recalls = selected_recalls_idxs & np.array(evs_free_recall.eegoffset>-1) & np.array(evs_free_recall.list > 0) & (nonrepeat_indicator>0) ##^
    evs_free_recall = evs_free_recall[good_recalls]

    # get temp key of recalls that lead to intrusions ~~~
    pre_intrusion_recalls = getRecallsBeforeIntrusions(evs,evs_free_recall)

    if len(evs_free_recall)==0: #...and that any are left!
        continue

    # get output position in each list for this session's recalls
    session_corrected_list_ops = getOutputPositions(evs,evs_free_recall)
        
    pairs = reader.load('pairs')
    try:
        localizations = reader.load('localization')
    except:
        localizations = []
    tal_struct, bipolar_pairs, mpchans = get_bp_tal_struct(sub, montage=mont, localization=loc)
    elec_regions,atlas_type,pair_number,has_stein_das = get_elec_regions(localizations,pairs) 

#         # count elec regions with specific string...mostly here to comment out below and test for good sessions
#         if len(findAinB(ENT_labels,elec_regions))>0:
#             ent_ct = len(findAinB(ENT_labels,elec_regions))
#             ent_elec_ct = np.append(ent_elec_ct,sub+'_'+str(session)+'_ct-'+str(ent_ct))

    # load eeg
    if selected_period == 'surrounding_recall':
        total_recalls = total_recalls + len(evs_free_recall) # get total recalls from lists
        total_lists = total_lists + len(evs[evs.type=='WORD'].list.unique()) # get total lists
        kept_recalls = kept_recalls + len(evs_free_recall)
        eeg_events = evs_free_recall
        
        # fix EEG offset due to Unity implementation error @@
#         init_time = eeg_events.iloc[0].eegoffset
        eeg_events = correctEEGoffset(sub,session,exp,reader,eeg_events)
#         print(sub+'-'+str(session)+'-'+exp+': '+str(eeg_events.iloc[0].eegoffset-init_time))
        
    elif selected_period == 'whole_retrieval':
        # grab whole retrieval periods for a better baseline of SWRs
        evs_rets = evs[evs.type=='REC_START']
        evs_rets = evs_rets[evs_rets.list>-1] # remove practice lists
        evs_rets = evs_rets[evs_rets.eegoffset>-1] # any trial with no eeg gets removed by cmlreaders so it's not in ripple_array 
        eeg_events = evs_rets
        
        # get alignmnet of end of beep time to EEG so can align retrieval to end of beep across all sessions
        align_adjust = getRetrievalStartAlignmentCorrection(sub,session,exp) # in ms
        
    elif selected_period == 'whole_encoding':
        # grab whole encoding periods to assess lists with more ripples
        evs_enc = evs[evs.list > -1] # remove practice lists **            
        evs_enc = evs_enc[evs_enc.eegoffset > -1] # any trial with no eeg gets removed by cmlreaders so it's not in ripple_array 

        # beginning of encoding doesn't have a consistent code, and the last consistent one before it is COUNTDOWN_END, so search for 1st WORD after each
        
        # gotta do this for each individually since it's not a consistent offset across lists (e.g. sometimes COUNTDOWN_END shows up after ORIENT_START)
        countdown_idxs = findInd(evs_enc.type=='COUNTDOWN_END')
        first_word_idxs = []
        for countdown_idx in countdown_idxs:
            got_words = 0
            next_type = 1
            while got_words == 0:
                if (countdown_idx+next_type)<len(evs_enc):
                    # if you find a WORD, align all the events to this index to get the encoding_starts
                    if evs_enc.iloc[countdown_idx+next_type].type == 'WORD':
                        first_word_idxs.append(countdown_idx+next_type)
                        # reset values
                        got_words = 1
                        next_type = 1                    
                    else:
                        next_type+=1
                else: # sometimes patients stop working after countdown, so if that happens just get out of while loop and you're done!
                    got_words = 1
                    next_type = 1
        
        eeg_events = evs_enc.iloc[first_word_idxs]           
        
    elif selected_period == 'encoding':
        ## apparently there are repeated words?? should add program to check and remove
        # I'm going to save encoding word events too, but need a mask to keep track of:
        # 0) words not recalled 1) words recalled from this list 2) words later recalled BUT IRI<2 s so removed
        evs_encoding_words = evs[evs.type=='WORD']
        evs_encoding_words = evs_encoding_words[evs_encoding_words.list>-1]            
        evs_encoding_words = evs_encoding_words[evs_encoding_words.eegoffset>-1]
        encoded_word_key = np.zeros(len(evs_encoding_words)) # 0 for not recalled
        encoded_word_key[evs_encoding_words.recalled==True] = 2 # 2 for recalled but removed bc IRI<2 s
        encoded_word_key[evs_encoding_words.item_name.isin(evs_free_recall.item_name.unique())] = 1 # recalled words
        # since finding all encoding words IN the list of correctly free recalled words won't have any intrusions
        eeg_events = evs_encoding_words

    elif (selected_period == 'math') | (selected_period == 'math_retrieval'): #$$
        
        evs_math = reader.load('all_events')
        evs_math = evs_math[(evs_math.type=='PROB')]

        # select which recalls??         
        [recall_selection_name,selected_recalls_idxs] = selectRecallType(recall_type_switch,evs_math,IRI,recall_minimum)

        # let's make sure remaining recalls are not repeated, have eeg, and are not from non-practice lists
        good_recalls = selected_recalls_idxs & np.array(evs_math.eegoffset>-1) & np.array(evs_math.list > -1) & np.array(evs_math.iscorrect==1)
        evs_math = evs_math[good_recalls]
        eeg_events = copy(evs_math) #$$
        
        if selected_period == 'math_retrieval': #$$
            temp_eeg = reader.load_eeg(events=eeg_events, rel_start=0, rel_stop=100) # just to get sampling rate
            sr = temp_eeg.samplerate
            sr_factor = 1000/sr
            eeg_events.mstime = eeg_events.mstime+eeg_events.rectime # align to retrieval key-in times...although I don't think this is actually used again?
            eeg_events.eegoffset = eeg_events.eegoffset+[int(v) for v in np.round(eeg_events.rectime/sr_factor).values] # this is only one that matters for aligning to EEG 
            # positive means looking at EEG into the future (when rectime happens)

    # fixing bad trials
    if sub == 'R1045E' and exp=='FR1': # this one session has issues in eeg trials past these points so remove events
        if selected_period == 'surrounding_recall':
            eeg_events = eeg_events.iloc[:65,:] # only the first 66 recalls have good eeg
        elif selected_period == 'whole_retrieval':
            eeg_events = eeg_events.iloc[:20,:] # only the first 20 retrieval periods have good eeg
        elif selected_period == 'encoding':
            eeg_events = eeg_events.iloc[:263,:] # same idea
            encoded_word_key = encoded_word_key[:263]
        elif (selected_period == 'math') | (selected_period == 'math_retrieval'):
            eeg_events = [] #$$

    eeg = reader.load_eeg(events=eeg_events, rel_start=psth_start-eeg_buffer+align_adjust, 
                            rel_stop=psth_end+eeg_buffer+align_adjust, clean=True, scheme=pairs) #**
    # event X channel X time
#         import ipdb; ipdb.set_trace()

    sr = eeg.samplerate
    
    # If using Vaz algo can't do >250 Hz IED detection so don't use this sub
    if (sr<990) and filter_type=='butter': #^^^
        print('Cannot use '+sub+' since using Vaz algo and sr ≤ 500!')
        continue        

    # if weird samplerate, add a few ms to make the load work
    if (499<sr<500) | (998<sr<1000):
        time_add = 1
        if (499<sr<500):
            sr = 500
        elif (998<sr<1000):
            sr = 1000
        while eeg.shape[2] < (psth_end-psth_start+2*eeg_buffer)/(1000/sr):
            eeg = reader.load_eeg(events=eeg_events, rel_start=psth_start-eeg_buffer+align_adjust, 
                                    rel_stop=psth_end+eeg_buffer+time_add+align_adjust, clean=True, scheme=pairs)
            if time_add>50: #**
                continue
            time_add+=1
        time_add_save.append(time_add)
        eeg.samplerate = sr # need to overwrite those that were just fixed

    eeg_ptsa = eeg.to_ptsa()
    eeg = None # clear variable
#         break;break;break # to look at eeg_ptsa (plots at bottom of notebook) before filtering

    # if we're doing big period like encoding, split in half by electrodes
    # so don't run out of memory
    if (selected_period == 'whole_encoding') | (selected_period == 'whole_retrieval'):
        half_elec = int(np.shape(eeg_ptsa)[1]/2)
        eeg_ptsa1 = ButterworthFilter(timeseries=eeg_ptsa[:,:half_elec,:], freq_range=[58.,62.], filt_type='stop', order=4).filter()
        eeg_ptsa1 = ButterworthFilter(timeseries=eeg_ptsa1, freq_range=[178.,182.], filt_type='stop', order=4).filter()
        eeg_ptsa2 = ButterworthFilter(timeseries=eeg_ptsa[:,half_elec:,:], freq_range=[58.,62.], filt_type='stop', order=4).filter()
        eeg_ptsa2 = ButterworthFilter(timeseries=eeg_ptsa2, freq_range=[178.,182.], filt_type='stop', order=4).filter()
        eeg_ptsa = eeg_ptsa1.append(eeg_ptsa2,'channel')
    else:
        # line removal...don't do 120 for now (I never see any line noise there for whatever reason)
        eeg_ptsa = ButterworthFilter(timeseries=eeg_ptsa, freq_range=[58.,62.], filt_type='stop', order=4).filter()
        eeg_ptsa = ButterworthFilter(timeseries=eeg_ptsa, freq_range=[178.,182.], filt_type='stop', order=4).filter()
        
    # let's save HFA too
    HFA_freqs = np.logspace(np.log10(64),np.log10(178),10)
    HFA_eeg = ButterworthFilter(timeseries=eeg_ptsa, freq_range=[118.,122.], filt_type='stop', order=4).filter()
    HFA_eeg = ButterworthFilter(timeseries=HFA_eeg, freq_range=0.5, filt_type='highpass',order=4).filter() 
    HFA_morlet = MorletWaveletFilter(timeseries=HFA_eeg, freqs=HFA_freqs, output='power', width=5, verbose=True).filter()

    # now can remove buffers
    sr_factor = 1000/sr
    HFA_morlet = HFA_morlet[:,:,:,int(eeg_buffer/sr_factor):int(np.shape(HFA_morlet)[3]-(eeg_buffer/sr_factor))]
    HFA_morlet = xarray.ufuncs.log10(HFA_morlet, out=HFA_morlet.values)
    # resample down to 10 Hz (100 ms bins)

    HFA_morlet = ResampleFilter(timeseries=HFA_morlet,resamplerate=10).filter() # axes are freqs (10) X words X pairs X bins after downsample
#         # zscore across events & time bins # doing it differently now after talking to Mike 2022-03-08
#         HFA_morlet = (HFA_morlet - np.mean(HFA_morlet, axis=(1,3))) / np.std(HFA_morlet, axis=(1,3)) 
    # z-score using std of time bin averaged instead (mean is same either way)
    HFA_morlet = (HFA_morlet - np.mean(HFA_morlet, axis=(1,3))) / np.std(np.mean(HFA_morlet, axis=3),axis=1)
    HFA_morlet = np.mean(HFA_morlet,0) # mean over the 10 frequencies (now down to events X pairs X 100 ms bins)
            
    ## FILTERS ##
    trans_width = 5. # Width of transition region, normalized so that 1 corresponds to pi radians/sample. 
    # That is, the frequency is expressed as a fraction of the Nyquist frequency.
    ntaps = (2/3)*np.log10(1/(10*(1e-3*1e-4)))*(sr/trans_width) # gives 400 with sr=500, trans=5
    # formula from Belanger's Digital Processing of Signals
    # see https://dsp.stackexchange.com/questions/31066/how-many-taps-does-an-fir-filter-need for how to use
    
    if sr == 512 or sr == 1024 or sr == 1023.999: # last one fixes R1221P @@
        ntaps = np.ceil(ntaps)
        
    nyquist = sr/2        

    print(f"Performing filtering")
    
    # filter for ripples using filter selected above
    if filter_type == 'hamming':
        # need to subtract out to get the filtered signal since default is bandstop but want to keep it as PTSA  
        FIR_bandstop = firwin(int(ntaps+1), [70.,178.], fs=sr, window='hamming',pass_zero='bandstop')               
        #         eeg_rip_band = filtfilt(FIR_bandpass,1.,eeg_ptsa) # can't use ptsa_to_mne this way so use eeg minus bandstopped signal            
        eeg_rip_band = eeg_ptsa-filtfilt(FIR_bandstop,1.,eeg_ptsa) 
        bandstop_25_60 = firwin(int(ntaps+1), [20.,58.], fs=sr, window='hamming',pass_zero='bandstop') # Norman 2019 IED            
        eeg_ied_band = eeg_ptsa-filtfilt(bandstop_25_60,1.,eeg_ptsa)
        ntaps40, beta40 = kaiserord(40, trans_width/nyquist)
        kaiser_40lp_filter = firwin(ntaps40, cutoff=40, window=('kaiser', beta40), scale=False, nyq=nyquist, pass_zero='lowpass')  
                    
    elif filter_type == 'butter':
        eeg_rip_band = ButterworthFilter(timeseries=eeg_ptsa, freq_range=[80.,120.], filt_type='bandpass',order=2).filter()
#             if sr == 500: # dropped below 250 Hz because too close Nyquist
#                 eeg_ied_band = ButterworthFilter(timeseries=eeg_ptsa, freq_range=250., filt_type='highpass',order=2).filter() 
#             elif sr >= 1000:    
#                 #this seems okay since large range far from Nyquist...problem is it likely misses key events
        eeg_ied_band = ButterworthFilter(timeseries=eeg_ptsa, freq_range=[250.,490.], filt_type='bandpass',order=2).filter() #^^^
        eeg_raw = ptsa_to_mne(eeg_ptsa,[0,psth_end-psth_start+2*eeg_buffer])    #**  
#         eeg_ptsa = None # clear variable # no reason to do this in local version...really for cluster

    elif filter_type == 'staresina':
        FIR_bandstop_star = firwin(241, [80.,100.], fs=sr, window='hamming',pass_zero='bandstop') # order = 3*80+1               
        eeg_rip_band = eeg_ptsa-filtfilt(FIR_bandstop_star,1.,eeg_ptsa)
        
    break