import numpy as np 

import sys 
sys.path.append('/home1/efeghhi/ripple_memory/analysis_code/')
from load_data import *
from analyze_data import *
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

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

def compute_MOVI(distA, distB, n_bins):
    
    '''
    ndarray distA, distB: probability distribution of gamma amplitudes grouped by theta phase.
    For permutated data dist is of shape num_permutations x n_bins
    int n_bins: number of theta phase bins used
    
    Computes MOVI, which is the MI of the difference of the two distributions
    '''
    
    # from chanaz 
    dist_diff = ((distA - distB) + 2/n_bins)/2
    MI_dist_diff = compute_MI(dist_diff, n_bins)
    return MI_dist_diff

def compute_p_value(MI, MI_permuted):
    
    '''
    float MI: modulation index values from real data
    ndarray MI_permuted: modulation index from permuted data
    
    Returns p-value based on permuted modulation index data  
    '''
    
    p_value = 1 - np.argwhere(MI > MI_permuted).shape[0]/MI_permuted.shape[0]
    return p_value


# load data 
def load_pac_pd(encoding_mode):
    
    print("Loading data")
    
    region_name = ['HPC']

    condition_on_ca1_ripples = False
    
    if encoding_mode: 
        catFR_dir = '/scratch/efeghhi/catFR1/ENCODING/'
    else:
        catFR_dir = '/scratch/efeghhi/catFR1/IRIonly/'
        
    data_dict = load_data(catFR_dir, region_name=region_name, encoding_mode=encoding_mode, 
                        condition_on_ca1_ripples=condition_on_ca1_ripples)

    if encoding_mode: 
        data_dict = remove_wrong_length_lists(data_dict)
        
    # ca1
    ca1_elecs = [x for x in HPC_labels if 'ca1' in x]
    data_dict_ca1 = select_region(data_dict, ca1_elecs)
    count_num_trials(data_dict_ca1, "ca1")

    data_dict_region = data_dict_ca1
    
    
    sr = 500 # sampling rate in Hz
    sr_factor = 1000/sr 
    
    if encoding_mode:
        # for encoding trials, neural recording 
        # starts from -700 ms before word onset 
        # and finishes 2300 ms post word onset 
        start_time = -700
        end_time = 2300

        # timepoints of interest
        ripple_start = 400
        ripple_end = 1100

    else:
        # for recall trials, neural recording 
        # starts from -200 ms before word onset 
        # and finishes 2000 ms post word onset 
        start_time = -2000
        end_time = 2000
        
        # this is where we are interested in analyzing 
        # -600 to -100 ms (0 ms is word onset)
        ripple_start = -600
        ripple_end = -100

    # convert to indices based on start time and sampling rate factor
    ripple_start_idx = int((ripple_start - start_time)/sr_factor)
    ripple_end_idx = int((ripple_end-start_time)/sr_factor)

    data_dict_ca1.keys()

    # create clustered int array
    clustered_int = create_semantic_clustered_array(data_dict_region, encoding_mode)
    data_dict_region['clust_int'] = clustered_int

    dd_trials = dict_to_numpy(data_dict_region, order='C')

    ripple_exists = create_ripple_exists(dd_trials, ripple_start_idx, ripple_end_idx, 0)
    dd_trials['ripple_exists'] = ripple_exists
    
    # corresponds to 400 - 1100 ms post word onset if encoding
    # or -600 to -100 ms before word recall if recalled 
    start_time = ripple_start_idx
    end_time = ripple_end_idx
    num_time_points = end_time - start_time
    num_trials = len(dd_trials['theta_phase'])
    n_bins = 18 # number of bins for theta phase 

    theta_raveled = np.ravel(dd_trials['theta_phase'][:, start_time:end_time])
    theta_binned = np.floor((((theta_raveled + np.pi)/(2*np.pi))*n_bins))
    high_gamma_raveled = np.ravel(dd_trials['high_gamma'][:, start_time:end_time])
    low_gamma_raveled = np.ravel(dd_trials['low_gamma'][:, start_time:end_time])
    trial_number = np.repeat(np.arange(num_trials), num_time_points)

    clustered = np.repeat(dd_trials['clust_int'], num_time_points)
    subj_ravel = np.repeat(dd_trials['subj'], num_time_points)
    
    if encoding_mode: 
        correct = np.repeat(dd_trials['correct'], num_time_points)
    else:
        # if using recalled data, just fill all trials with correct set to -1
        # since there is no correct key
        correct = -1*np.ones_like(clustered)
        
    regression_pd = pd.DataFrame({'theta':theta_binned, 'high_gamma': high_gamma_raveled, 'low_gamma':low_gamma_raveled, 
                        'correct': correct, 'clust': clustered, 'subj': subj_ravel, 'trial_num': trial_number})


    return regression_pd, n_bins, num_time_points, np.unique(dd_trials['subj'])
    
            

def compute_gamma_theta_dist(pd):
    
               
    low_gammas_by_theta = []
    high_gammas_by_theta = []
    
    for theta_phase in np.unique(pd.theta):
        
        tp = pd.loc[pd.theta==theta_phase]
        low_gammas_by_theta.append(np.mean(tp.lg))
        high_gammas_by_theta.append(np.mean(tp.hg))
        
    lg_pdist = np.array(low_gammas_by_theta)/np.sum(low_gammas_by_theta)
    hg_pdist = np.array(high_gammas_by_theta)/np.sum(high_gammas_by_theta)
    
    return lg_pdist, hg_pdist
        
def compute_theta_gamma_PAC(single_subject, num_time_points, n_bins,
                            num_permutations=300):
    
    num_trials_subj = np.unique(single_subject.trial_num).shape[0]

    stored_dist_lg = np.zeros((num_trials_subj, n_bins))
    stored_dist_permutations_lg = np.zeros((num_trials_subj, num_permutations, n_bins))
    stored_dist_hg = np.zeros((num_trials_subj, n_bins))
    stored_dist_permutations_hg = np.zeros((num_trials_subj, num_permutations, n_bins))

    for idx, trial_num in enumerate(np.unique(single_subject.trial_num)):

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

def finished_subjects(mode, metric, savePath = '/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/saved_results/'):
    
    '''
    :param int mode: not correct (0), correct (1), unclustered (2), clustered (3) 
    :param str: MOVI or MI
    :param str savePath: folder where previous results are stored to 
    '''
    
    # just load hg, because the subjects are the same
    results= dict(np.load(f"{savePath}hg_{metric}_by_subj_{mode}.npz"))
    
    subjects_ran = sorted(results.keys())
    
    return subjects_ran 
    

def save_MOVI(mode, savePath='/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/'):
    
    low_gamma_MOVI_by_subj = {}
    high_gamma_MOVI_by_subj = {}
    low_gamma_MOVI_z_by_subj = {}
    high_gamma_MOVI_z_by_subj = {}
    
    low_gamma_p_vals_by_subj = {}
    high_gamma_p_vals_by_subj = {}
    
    pac_pd_encoding, n_bins, num_time_points_encoding, subj_encoding = load_pac_pd(1)
    pac_pd_recalled, _, num_time_points_recalled, subj_recalled = load_pac_pd(0)
    
    if mode == 2:
        pac_pd_encoding = pac_pd_encoding.loc[pac_pd_encoding.clust==0]
        pac_pd_recalled = pac_pd_recalled.loc[pac_pd_recalled.clust==0]
    if mode == 3:
        pac_pd_encoding = pac_pd_encoding.loc[pac_pd_encoding.clust==1]
        pac_pd_recalled = pac_pd_recalled.loc[pac_pd_recalled.clust==1]
        
    subj_both = np.intersect1d(subj_encoding, subj_recalled) 
    
    min_num_trials = 30
    
    subjects_ran = finished_subjects(mode, 'MOVI') 
        
    for subj in subj_both:
        
        print(f"Subject: {subj}")
        
        if subj in subjects_ran:
            print("Ran subject already, skipping")
            continue

        single_subject_encoding = pac_pd_encoding.loc[pac_pd_encoding.subj==subj]
        single_subject_recalled = pac_pd_recalled.loc[pac_pd_recalled.subj==subj]

        num_trials_subj_encoding = int(single_subject_encoding.high_gamma.shape[0]/num_time_points_encoding)
        num_trials_subj_recalled = int(single_subject_recalled.high_gamma.shape[0]/num_time_points_recalled)
        
        if min(num_trials_subj_encoding, num_trials_subj_recalled) < int(min_num_trials):

            print("trial count too low, skipping subject")
            continue
            
        print("ENCODING")
        theta_gamma_hist_encoding = compute_theta_gamma_PAC(single_subject_encoding,
                                                            num_time_points_encoding, n_bins)
        print("RECALLED")
        theta_gamma_hist_recalled = compute_theta_gamma_PAC(single_subject_recalled, 
                                                            num_time_points_recalled, n_bins)
                                                 
        
        stored_dist_lg_encoding = theta_gamma_hist_encoding[0]
        stored_dist_hg_encoding = theta_gamma_hist_encoding[1]
        stored_dist_permutations_lg_encoding = theta_gamma_hist_encoding[2]
        stored_dist_permutations_hg_encoding = theta_gamma_hist_encoding[3]
        
        stored_dist_lg_recalled = theta_gamma_hist_recalled[0]
        stored_dist_hg_recalled = theta_gamma_hist_recalled[1]
        stored_dist_permutations_lg_recalled = theta_gamma_hist_recalled[2]
        stored_dist_permutations_hg_recalled = theta_gamma_hist_recalled[3]
        
        MOVI_lg = compute_MOVI(np.mean(stored_dist_lg_encoding, axis=0), 
                               np.mean(stored_dist_lg_recalled, axis=0), n_bins)
        MOVI_hg = compute_MOVI(np.mean(stored_dist_hg_encoding, axis=0), 
                               np.mean(stored_dist_hg_recalled, axis=0), n_bins)
        
        MOVI_lg_p = compute_MOVI(np.mean(stored_dist_permutations_hg_encoding, axis=0), 
                               np.mean(stored_dist_permutations_lg_recalled, axis=0), n_bins)
        MOVI_hg_p = compute_MOVI(np.mean(stored_dist_permutations_hg_encoding, axis=0), 
                               np.mean(stored_dist_permutations_hg_recalled, axis=0), n_bins)
        
        # compute p-value as fraction of permuted MI values that are higher
        # than or equal to the MI value obtained with the real data 
        p_val_lg = compute_p_value(MOVI_lg, MOVI_lg_p)
        p_val_hg = compute_p_value(MOVI_hg, MOVI_hg_p)
        
        low_gamma_p_vals_by_subj[subj] = p_val_lg

        high_gamma_p_vals_by_subj[subj] = p_val_hg

        print("P val: ", p_val_lg, p_val_hg)
        print("MOVI: ", MOVI_lg, MOVI_hg)

        # save MI for non-permuted data
        low_gamma_MOVI_by_subj[subj] = MOVI_lg
        high_gamma_MOVI_by_subj[subj] = MOVI_hg

        # save MI normalized by null distribution
        low_gamma_MOVI_z_by_subj[subj] = (MOVI_lg - np.mean(MOVI_lg_p))/np.std(MOVI_lg_p)
        high_gamma_MOVI_z_by_subj[subj] = (MOVI_hg - np.mean(MOVI_hg_p))/np.std(MOVI_hg_p)


        np.savez(f'{savePath}lg_MOVI_by_subj_{mode}', **low_gamma_MOVI_by_subj)
        np.savez(f'{savePath}hg_MOVI_by_subj_{mode}', **high_gamma_MOVI_by_subj)

        np.savez(f'{savePath}lg_MOVI_by_subj_z_{mode}', **low_gamma_MOVI_z_by_subj)
        np.savez(f'{savePath}hg_MOVI_by_subj_z_{mode}', **high_gamma_MOVI_z_by_subj)

        np.savez(f'{savePath}lg_MOVI_p_vals_by_subj_{mode}', **low_gamma_p_vals_by_subj)
        np.savez(f'{savePath}hg_MOVI_p_vals_by_subj_{mode}', **high_gamma_p_vals_by_subj)

        
def save_MI(encoding_mode, mode, savePath='/home1/efeghhi/ripple_memory/analysis_code/pac_analyses/'):
    
    '''
    :param int encoding_mode: 1 for encoding, 0 for recall data 
    :param int mode: 0 for correct trials, 1 for incorrect, 
    2 for not clustered, 3 for clustered. Note for recall data only modes
    2 and 3 can be passed.
    '''

    pac_pd, n_bins, num_time_points, subj_arr = load_pac_pd(encoding_mode)
    
    min_num_trials = 30
    
    if mode == 0 or mode == 1:
        if encoding_mode == 0:
            print("Cannot pass modes 0 and 1 for recalled data")
            return 0 
    
    if mode == 0:
        pac_pd = pac_pd.loc[pac_pd.correct==0]
    if mode == 1:
        pac_pd = pac_pd.loc[pac_pd.correct==1]
    if mode == 2:
        pac_pd = pac_pd.loc[pac_pd.clust==0]
    if mode == 3:
        pac_pd = pac_pd.loc[pac_pd.clust==1]
    
    # MI stands for modulation index
    # which is (log(n_bins) - H(p))/log(n_bins), which is 0 if the PAC
    # histogram has the same entropy as the uniform distribution, and b/w 0 and 1
    # to the extent its entropy is less than the uniform distribution
    
    low_gamma_MI_by_subj = {}
    high_gamma_MI_by_subj = {}
    low_gamma_MI_z_by_subj = {}
    high_gamma_MI_z_by_subj = {}
    
    low_gamma_p_vals_by_subj = {}
    high_gamma_p_vals_by_subj = {}
    
    for subj in subj_arr:
    
        
        print(f"Subject: {subj}")

        single_subject = pac_pd.loc[pac_pd.subj==subj]

        num_trials_subj = int(single_subject.high_gamma.shape[0]/num_time_points)

        if num_trials_subj < int(min_num_trials):

            print("trial count too low, skipping subject")
            continue
            
        theta_gamma_hist = compute_theta_gamma_PAC(single_subject, 
                                                  num_time_points, n_bins)
        
        stored_dist_lg = theta_gamma_hist[0]
        stored_dist_hg = theta_gamma_hist[1]
        stored_dist_permutations_lg = theta_gamma_hist[2]
        stored_dist_permutations_hg = theta_gamma_hist[3]

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

        low_gamma_p_vals_by_subj[subj] = p_val_lg

        high_gamma_p_vals_by_subj[subj] = p_val_hg

        print("P val: ", p_val_lg, p_val_hg)
        print("MI: ", MI_lg, MI_hg)

        # save MI for non-permuted data
        low_gamma_MI_by_subj[subj] = MI_lg
        high_gamma_MI_by_subj[subj] = MI_hg

        # save MI normalized by null distribution
        low_gamma_MI_z_by_subj[subj] = (MI_lg - np.mean(MI_lg_p))/np.std(MI_lg_p)
        high_gamma_MI_z_by_subj[subj] = (MI_hg - np.mean(MI_hg_p))/np.std(MI_hg_p)

        if encoding_mode == 0:
            recall_str = "recalled"
        else:
            recall_str = ""

        np.savez(f'{savePath}lg_MI_by_subj_{mode}{recall_str}', **low_gamma_MI_by_subj)
        np.savez(f'{savePath}hg_MI_by_subj_{mode}{recall_str}', **high_gamma_MI_by_subj)

        np.savez(f'{savePath}lg_MI_by_subj_z_{mode}{recall_str}', **low_gamma_MI_z_by_subj)
        np.savez(f'{savePath}hg_MI_by_subj_z_{mode}{recall_str}', **high_gamma_MI_z_by_subj)

        np.savez(f'{savePath}lg_MI_p_vals_by_subj_{mode}{recall_str}', **low_gamma_p_vals_by_subj)
        np.savez(f'{savePath}hg_MI_p_vals_by_subj_{mode}{recall_str}', **high_gamma_p_vals_by_subj)


