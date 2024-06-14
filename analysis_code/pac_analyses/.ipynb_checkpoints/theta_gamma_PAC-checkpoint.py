import numpy as np 
import sys 
sys.path.append('/home1/efeghhi/ripple_memory/analysis_code/')
from load_data import *
from analyze_data import *
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

def permute_amplitude_time_series(amplitude_time_series):
    
    
    '''
    ndarray amplitude_time_series: array containing amplitude values over a given trial 
    
    Circular permutes the input vector. 
    '''
    
    # Get the length of the time series
    series_length = len(amplitude_time_series)
    
    min_cut_index = int(0.1 * series_length)
    max_cut_index = int(0.9 * series_length )

    # Choose a random index to cut the time series
    cut_index = np.random.randint(min_cut_index, max_cut_index)

    # Create the permuted time series by reversing the order of both parts
    permuted_series = np.concatenate((amplitude_time_series[cut_index:], amplitude_time_series[:cut_index]))

    return permuted_series

def compute_gamma_theta_dist(pd):
    
    '''
    dataframe pd: contains theta phase, low gamma, and high gamma values for a single trial
    
    This function groups high and low gamma values by theta phase to create a histogram. 
    It then normalizes this histogram such that the bins sum to 1.
    '''
    

    low_gammas_by_theta = []
    high_gammas_by_theta = []
    
    for theta_phase in np.unique(pd.theta):
        
        tp = pd.loc[pd.theta==theta_phase]
        low_gammas_by_theta.append(np.mean(tp.lg))
        high_gammas_by_theta.append(np.mean(tp.hg))
        
    lg_pdist = np.array(low_gammas_by_theta)/np.sum(low_gammas_by_theta)
    hg_pdist = np.array(high_gammas_by_theta)/np.sum(high_gammas_by_theta)
    
    return lg_pdist, hg_pdist

def compute_entropy(probability_arr):
    
    '''
    ndarray probability_arr: 1 or 2d probability array
    
    If 1d array, returns entropy of the probability distribution.
    If 2d array, returns entr
    '''
    
    import math

    if len(probability_arr.shape)==1:
        entropy = -np.sum(probability_arr*np.log(probability_arr))
    else:
        # sum across bins (which are along the second axis) for permuted data
        entropy = -np.sum(probability_arr*np.log(probability_arr), axis=1)
    return entropy

def compute_MI(dist, n_bins):
    
    '''
    ndarray dist: probability distribution of gamma amplitudes grouped by theta phase.
    For permutated data dist is of shape num_permutations x n_bins
    int n_bins: number of theta phase bins used
    
    Computes modulation index, which is the entropy of the provided 
    '''
    
    
    uniform_entropy = np.log(n_bins)
    entropy_vals = compute_entropy(dist)
    
    MI = (uniform_entropy - entropy_vals)/uniform_entropy
    return MI


def compute_p_value(dat, dat_permuted):
    
    '''
    :param float dat: values from real data
    :param ndarray dat_permuted: permuted data
    
    Returns one-side p-value testing the hypothesis that the real
    data value is larger than the permuted distribution.
    '''
    
    # 1) find the number of permuted values the real data is larger than 
    # 2) divide this by the number of permuted values 
    # 3) Subtract by 1 to get p value
    p_value = 1 - np.argwhere(dat > dat_permuted).shape[0]/dat_permuted.shape[0]
    return p_value


# load data 
def load_pac_pd(encoding_mode, roi_mode):
    
    '''
    :param int encoding_mode: 0 for recall, 1 for encoding 

    :param int roi_mode: 0 for short, 1 for medium, 2 for long.
    '''
    
    if roi_mode == 0:
        enc_start_roi = 400
        enc_end_roi = 1100
        rec_start_roi = -600
        rec_end_roi = -100
        
    if roi_mode == 1:
        print("Loading 1 second data")
        enc_start_roi = 300
        enc_end_roi = 1300
        rec_start_roi = -1100
        rec_end_roi = -100
    
    if roi_mode == 2: 
        enc_start_roi = 100
        enc_end_roi = 1700
        rec_start_roi = -1600
        rec_end = -100

    sr = 500 # sampling rate in Hz
    sr_factor = 1000/sr 
    
    if encoding_mode:
        # for encoding trials, neural recording 
        # starts from -700 ms before word onset 
        # and finishes 2300 ms post word onset 
        start_time = -700
        end_time = 2300

        # timepoints of interest
        start_roi = enc_start_roi
        end_roi = enc_end_roi

    else:
        # for recall trials, neural recording 
        # starts from -1000 ms before word onset 
        # and finishes 1000 ms post word onset 
        start_time = -2000
        end_time = 2000
        
        # this is where we are interested in analyzing 
        # -600 to -100 ms (0 ms is word onset)
        start_roi = rec_start_roi
        end_roi = rec_end_roi

    # convert to indices based on start time and sampling rate factor
    start_idx = int((start_roi - start_time)/sr_factor)
    end_idx = int((end_roi-start_time)/sr_factor)

    
    if encoding_mode:
        dd_trials = np.load('/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/updated_data/dd_trials_encoding.npz')
    else:
        dd_trials = np.load('/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/updated_data/dd_trials_recall.npz')

    # corresponds to 400 - 1100 ms post word onset if encoding
    # or -600 to -100 ms before word recall if recalled 
    start_time = start_roi
    end_time = end_roi
    num_time_points = end_time - start_time
    num_trials = len(dd_trials['theta'])
    n_bins = 18 # number of bins for theta phase 
        
    selected_idxs = np.ones(num_trials,  dtype=bool)
    
    # discretize theta phase to take on integer values from 0 to 18
    theta_binned = np.ravel(np.floor(((dd_trials['theta'][:, start_time:end_time] + np.pi)/(2*np.pi))*n_bins))
    high_gamma_raveled = np.ravel(dd_trials['high_gamma'][:, start_time:end_time])
    low_gamma_raveled = np.ravel(dd_trials['low_gamma'][:, start_time:end_time])
    trial_number = np.repeat(np.arange(num_trials), num_time_points)
    clustered = np.repeat(dd_trials['clust_int'][selected_idxs], num_time_points)
    subj_ravel = np.repeat(dd_trials['subj'][selected_idxs], num_time_points)
    elec_ravel = np.repeat(dd_trials['elec_labels'][selected_idxs], num_time_points)
    
    if encoding_mode: 
        correct = np.repeat(dd_trials['correct'][selected_idxs], num_time_points)
    else:
        # if using recalled data, just fill all trials with correct set to -1
        # since there is no correct key
        correct = -1*np.ones_like(clustered)
        
   
    regression_pd = pd.DataFrame({'theta':theta_binned, 'high_gamma': high_gamma_raveled,'low_gamma':low_gamma_raveled, 
                        'correct': correct, 'clust': clustered, 'subj': subj_ravel, 'trial_num': trial_number, 
                                 'elec_labels': elec_ravel})

    return regression_pd, n_bins, num_time_points

def compute_gamma_theta_dist(pd):
    
    low_gammas_by_theta = []
    high_gammas_by_theta = []
    
    for theta_phase in np.unique(pd.theta):
        
        tp = pd.loc[pd.theta==theta_phase]
        low_gammas_by_theta.append(np.mean(tp.lg))
        high_gammas_by_theta.append(np.mean(tp.hg))
        
    # don't make into a distribution until after you sum across trials
    lg_pdist = np.array(low_gammas_by_theta)
    hg_pdist = np.array(high_gammas_by_theta)
    
    return lg_pdist, hg_pdist
        
def compute_theta_gamma_PAC(single_subject, num_time_points, n_bins,
                            num_permutations=100):
    
    num_trials_subj = np.unique(single_subject.trial_num).shape[0]
    stored_dist_lg = np.zeros((num_trials_subj, n_bins))
    stored_dist_permutations_lg = np.zeros((num_trials_subj, num_permutations, n_bins))
    stored_dist_hg = np.zeros((num_trials_subj, n_bins))
    stored_dist_permutations_hg = np.zeros((num_trials_subj, num_permutations, n_bins))

    for idx, trial_num in enumerate(np.unique(single_subject.trial_num)):
        
        if idx % 20 == 0:
            print(f"Trial {idx} out {np.unique(single_subject.trial_num).shape[0]}")

        trial_pd = single_subject.loc[single_subject.trial_num == trial_num]

        original_pd = pd.DataFrame({'lg': trial_pd.low_gamma, 'hg': trial_pd.high_gamma, 
                                    'theta': trial_pd.theta})

        # returns nbins shape array,
        # where each entry is the gamma (low or high) "probability"
        # at a given theta phase bin
        lg_dist, hg_dist = compute_gamma_theta_dist(original_pd)

        # store gamma distributions by theta phase for each trial 
        stored_dist_lg[idx] = lg_dist
        stored_dist_hg[idx] = hg_dist

        # for each trial, also compute a distribution when permuting the gamma 
        # arrays
        for n in range(num_permutations):

            low_gamma_permuted = permute_amplitude_time_series(trial_pd.low_gamma)
            high_gamma_permuted = permute_amplitude_time_series(trial_pd.high_gamma)


            permuted_pd = pd.DataFrame({'lg':low_gamma_permuted, 'hg':high_gamma_permuted, 
                                        'theta':trial_pd.theta})

            lg_dist_p, hg_dist_p = compute_gamma_theta_dist(permuted_pd)

            stored_dist_permutations_lg[idx, n] = lg_dist_p
            stored_dist_permutations_hg[idx, n] = hg_dist_p

    return stored_dist_lg, stored_dist_hg, stored_dist_permutations_lg, stored_dist_permutations_hg

def finished_subjects(mode, metric, 
                      savePath = '/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/saved_results/'):
    
    '''
    :param int mode: not correct (0), correct (1), unclustered (2), clustered (3) 
    :param str: MOVI or MI
    :param str savePath: folder where previous results are stored to 
    '''
    
    # just load hg, because the subjects are the same
    results= dict(np.load(f"{savePath}hg_{metric}_by_subj_{mode}.npz"))
    
    subjects_ran = sorted(results.keys())
    
    return subjects_ran         
            
def save_JSD(theta_gamma_hist_encoding, theta_gamma_hist_recalled):
         
        stored_dist_lg_encoding = np.mean(theta_gamma_hist_encoding[0],axis=0)
        stored_dist_hg_encoding = np.mean(theta_gamma_hist_encoding[1],axis=0)
        stored_dist_permutations_lg_encoding = np.mean(theta_gamma_hist_encoding[2],axis=0)
        stored_dist_permutations_hg_encoding = np.mean(theta_gamma_hist_encoding[3], axis=0)
        
        stored_dist_lg_recalled = np.mean(theta_gamma_hist_recalled[0],axis=0)
        stored_dist_hg_recalled = np.mean(theta_gamma_hist_recalled[1],axis=0)
        stored_dist_permutations_lg_recalled = np.mean(theta_gamma_hist_recalled[2],axis=0)
        stored_dist_permutations_hg_recalled = np.mean(theta_gamma_hist_recalled[3],axis=0)
        
        print(stored_dist_lg_encoding.shape, stored_dist_permutations_lg_encoding.shape)
        
        jsd_lg = jensenshannon(stored_dist_lg_encoding, stored_dist_lg_recalled)
        jsd_hg = jensenshannon(stored_dist_hg_encoding, stored_dist_hg_recalled)
       
        jsd_lg_permuted = jensenshannon(stored_dist_permutations_lg_encoding, 
                                        stored_dist_permutations_lg_recalled, axis=1)
        
        jsd_hg_permuted = jensenshannon(stored_dist_permutations_hg_encoding, 
                                        stored_dist_permutations_hg_recalled, axis=1)
        
        p_val_lg = compute_p_value(jsd_lg, jsd_lg_permuted)
        p_val_hg = compute_p_value(jsd_hg, jsd_hg_permuted)
        
          
        jsd_lg_z = (jsd_lg - np.mean(jsd_lg_permuted))/np.std(jsd_lg_permuted)
        jsd_hg_z = (jsd_hg - np.mean(jsd_hg_permuted))/np.std(jsd_hg_permuted)
        
    
        return jsd_lg, jsd_hg, p_val_lg, p_val_hg, jsd_lg_z, jsd_hg_z, jsd_lg_permuted, jsd_hg_permuted
        
        
def save_MI(theta_gamma_hist, n_bins):
    
        stored_dist_lg = theta_gamma_hist[0]
        stored_dist_hg = theta_gamma_hist[1]
        stored_dist_permutations_lg = theta_gamma_hist[2]
        stored_dist_permutations_hg = theta_gamma_hist[3]

        # average aross distributions generated across trials for a given participant 
        # and compute MI
        dist_lg = np.mean(stored_dist_lg, axis=0)
        dist_hg = np.mean(stored_dist_hg, axis=0)

        # average aross distributions generated across trials for a given participant 
        # and compute MI
        MI_lg = compute_MI(np.mean(stored_dist_lg, axis=0), n_bins)
        MI_hg = compute_MI(np.mean(stored_dist_hg, axis=0), n_bins)

        # also do the same for permuted data. For permuted data, when we average across the trials 
        # the phase amplitude coupling should cancel out because each trial should be coupled to 
        # a distinct theta phase value because of the coupling. The resulting output will be of shape
        # num_permutations. 
        MI_lg_p = compute_MI(np.mean(stored_dist_permutations_lg, axis=0), n_bins)
        MI_hg_p = compute_MI(np.mean(stored_dist_permutations_hg, axis=0), n_bins)

        # compute p-value as fraction of permuted MI values that are higher
        # than or equal to the MI value obtained with the real data 
        p_val_lg = compute_p_value(MI_lg, MI_lg_p)
        p_val_hg = compute_p_value(MI_hg, MI_hg_p)
        
        MI_lg_z = (MI_lg - np.mean(MI_lg_p))/np.std(MI_lg_p)
        MI_hg_z = (MI_hg - np.mean(MI_hg_p))/np.std(MI_hg_p)
        
        return MI_lg, MI_hg, MI_lg_z, MI_hg_z, p_val_lg, p_val_hg, dist_lg, dist_hg
    
def save_MI_JSD(behav_mode, roi_mode,
            savePath='/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/saved_results/'):
    
    '''
    :param int behav_mode: whether to analyze clustered (3) or unclustered (2) recalls
    :param int roi_mode: 0 (short), 1 (1 sec), or 2 (1.5 sec) region of interest 
    :param str savePath: where to save MI and JSD values
    '''
    
    bad_sessions = ['R1108J-2']
    
    # load neural and behavorial data for encoding and recall phases 
    dd_trials_encode = np.load('/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/updated_data/dd_trials_encoding.npz')
    dd_trials_recalled = np.load('/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/updated_data/dd_trials_recall.npz')
    
    # process into a pd dataframe
    pac_pd_encoding, n_bins, num_time_points_encoding = load_pac_pd(1, roi_mode)
    pac_pd_recalled, _, num_time_points_recalled = load_pac_pd(0, roi_mode)
    
    # only interested in clust/unclust recalls for now, so analyzing
    # only subsuquently recalled trials
    pac_pd_encoding = pac_pd_encoding.loc[pac_pd_encoding.correct==1]
    
    # unclustered recalls
    if behav_mode == 2:
        pac_pd_encoding = pac_pd_encoding.loc[(pac_pd_encoding.clust==0)]
        pac_pd_recalled = pac_pd_recalled.loc[pac_pd_recalled.clust==0]
        
    # clustered recall
    if behav_mode == 3:
        pac_pd_encoding = pac_pd_encoding.loc[pac_pd_encoding.clust==1]
        pac_pd_recalled = pac_pd_recalled.loc[pac_pd_recalled.clust==1]
    
        
    min_num_trials = 5
    
    store_MI_dict = {'MI_lg': [], 'MI_hg': [], 'MI_lg_z': [], 
                     'MI_hg_z': [], 'p_lg': [], 'p_hg': [], 'encoding': [], 'sub_sess_elec': [], 
                    'num_trials': []}
    
    store_JSD_dict = {'jsd_lg': [], 'jsd_hg': [],'jsd_lg_z': [], 'jsd_hg_z': [], 'p_lg': [], 'p_hg': [], 
                      'jsd_lg_permuted': [], 'jsd_hg_permuted': [], 'sub_sess_elec': [], 
                      'num_trials': []}
    
    store_theta_gamma_PAC_dist = {'lg_encode': [], 'hg_encode': [], 'lg_recall': [], 'hg_recall': []}
    
    # elec labels contains the subject, session number, and electrode name 
    # for each electrode 
    subj_sess_elec_encoding = np.unique(pac_pd_encoding['elec_labels'])
    subj_sess_elec_recalled = np.unique(pac_pd_recalled['elec_labels'])
    
    shared_elec_labels = np.intersect1d(subj_sess_elec_encoding, subj_sess_elec_recalled) 
    
    for sel in shared_elec_labels:
        
        if len([x for x in bad_sessions if x in sel]) > 0:
            print("Skipping bad session")
            continue
        
        electrode_encoding = pac_pd_encoding.loc[pac_pd_encoding.elec_labels==sel]
        electrode_recalled = pac_pd_recalled.loc[pac_pd_recalled.elec_labels==sel]

        print(f"Electrode: {sel}")

        num_trials_elec_encoding = int(electrode_encoding.high_gamma.shape[0]/num_time_points_encoding)
        num_trials_elec_recalled = int(electrode_recalled.high_gamma.shape[0]/num_time_points_recalled)
        
        print(num_trials_elec_encoding, num_trials_elec_recalled)

        # because we are only looking at recalled (clust/noclust) trials, 
        # the trial count should be identical 
        if num_trials_elec_encoding != num_trials_elec_recalled:
            print("TRIAL COUNT IS WEIRD")
            print(sel)
            print(num_trials_elec_encoding, num_trials_elec_recalled)
            print(num_time_points_encoding, num_time_points_recalled)
            continue

        if min(num_trials_elec_encoding, num_trials_elec_recalled) < int(min_num_trials):
            print("trial count too low, skipping elec")
            print(sel)
            continue

        print("ENCODING")
        theta_gamma_hist_encoding = compute_theta_gamma_PAC(electrode_encoding,
                                                            num_time_points_encoding, n_bins)
        print("RECALLED")
        theta_gamma_hist_recalled = compute_theta_gamma_PAC(electrode_recalled, 
                                                            num_time_points_recalled, n_bins)

        MI_lg, MI_hg, MI_lg_z, MI_hg_z, p_val_lg, p_val_hg, dist_lg, dist_hg = save_MI(theta_gamma_hist_encoding, n_bins)

        store_MI_dict['MI_lg'].append(MI_lg)
        store_MI_dict['MI_hg'].append(MI_hg)
        store_MI_dict['MI_lg_z'].append(MI_lg_z)
        store_MI_dict['MI_hg_z'].append(MI_hg_z)
        store_MI_dict['p_lg'].append(p_val_lg)
        store_MI_dict['p_hg'].append(p_val_hg)
        store_MI_dict['encoding'].append(1)
        store_MI_dict['sub_sess_elec'].append(sel)
        store_MI_dict['num_trials'].append(num_trials_elec_encoding)
        
        store_theta_gamma_PAC_dist['lg_encode'].append(dist_lg)
        store_theta_gamma_PAC_dist['hg_encode'].append(dist_hg)


        MI_lg, MI_hg, MI_lg_z, MI_hg_z, p_val_lg, p_val_hg, dist_lg, dist_hg = save_MI(theta_gamma_hist_recalled, n_bins)

        store_MI_dict['MI_lg'].append(MI_lg)
        store_MI_dict['MI_hg'].append(MI_hg)
        store_MI_dict['MI_lg_z'].append(MI_lg_z)
        store_MI_dict['MI_hg_z'].append(MI_hg_z)
        store_MI_dict['p_lg'].append(p_val_lg)
        store_MI_dict['p_hg'].append(p_val_hg)
        store_MI_dict['encoding'].append(0)
        store_MI_dict['sub_sess_elec'].append(sel)
        store_MI_dict['num_trials'].append(num_trials_elec_recalled)
        
        store_theta_gamma_PAC_dist['lg_recall'].append(dist_lg)
        store_theta_gamma_PAC_dist['hg_recall'].append(dist_hg)

        jsd_lg, jsd_hg, p_val_lg, p_val_hg, jsd_lg_z, jsd_hg_z, jsd_lg_permuted, jsd_hg_permuted = \
        save_JSD(theta_gamma_hist_encoding, theta_gamma_hist_recalled)

        store_JSD_dict['jsd_lg'].append(jsd_lg)
        store_JSD_dict['jsd_hg'].append(jsd_hg)
        store_JSD_dict['jsd_lg_z'].append(jsd_lg_z)
        store_JSD_dict['jsd_hg_z'].append(jsd_hg_z)
        store_JSD_dict['p_lg'].append(p_val_lg)
        store_JSD_dict['p_hg'].append(p_val_hg)
        store_JSD_dict['jsd_lg_permuted'].append(jsd_lg_permuted)
        store_JSD_dict['jsd_hg_permuted'].append(jsd_hg_permuted)
        store_JSD_dict['sub_sess_elec'].append(sel)
        store_JSD_dict['num_trials'].append([num_trials_elec_encoding, num_trials_elec_recalled])

        np.savez(f'{savePath}MI_behav_{behav_mode}_roi_{roi_mode}_train', **store_MI_dict)
        np.savez(f'{savePath}JSD_behav_{behav_mode}_roi_{roi_mode}_train', **store_JSD_dict)
        np.savez(f'{savePath}dist_behav_{behav_mode}_roi_{roi_mode}_train', **store_theta_gamma_PAC_dist)