from load_data import * 
from analyze_data import * 
import sys
sys.path.append('/home1/efeghhi/ripple_memory/')
from brain_labels import HPC_labels, ENT_labels, PHC_labels, temporal_lobe_labels,\
                        MFG_labels, IFG_labels, nonHPC_MTL_labels, ENTPHC_labels, AMY_labels

############### set parameters ###############
save_data = True
start_time = -700 # recording start time relative to word onset (ms)
end_time = 2300 # recording end time relative to word onset (ms)

##############################################

# load data and remove lists of wrong length
catFR_dir_recall = '/scratch/efeghhi/catFR1/IRIonly/'
catFR_dir_encoding = '/scratch/efeghhi/catFR1/ENCODING/'

region_name = '' # if empty string, loads all data

print("LOADING DATA")
data_dict = load_data(catFR_dir_encoding, region_name, encoding_mode=1)
data_dict = remove_wrong_length_lists(data_dict)
data_dict_recall = load_data(catFR_dir_recall, region_name, encoding_mode=0)

breakpoint()

# divide data by region
# ca1
ca1_elecs = [x for x in HPC_labels if 'ca1' in x]
data_dict_ca1 = select_region(data_dict, ca1_elecs)
data_dict_ca1_recall = select_region(data_dict_recall, ca1_elecs)

# ca3
ca3_dg_elecs = [x for x in HPC_labels if ('ca3' in x) or ('dg' in x)]
data_dict_ca3_dg = select_region(data_dict, ca3_dg_elecs)
data_dict_ca3_dg_recall = select_region(data_dict_recall, ca3_dg_elecs)

# AMY
data_dict_amy = select_region(data_dict, AMY_labels)
data_dict_amy_recall = select_region(data_dict_recall, AMY_labels)

# entphc
data_dict_entphc = select_region(data_dict, ENTPHC_labels)
data_dict_entphc_recall = select_region(data_dict_recall, ENTPHC_labels)

# load behavorial data into shape num_trials
print("BEHAVORIAL DATA")
relevant_keys = ['position', 'correct', 'clust', 'subj', 'sess', 'list_num', 'serial_pos']
non_neural_ca1 = reshape_to_trial_num(data_dict_ca1, keys=relevant_keys)
non_neural_ca3_dg = reshape_to_trial_num(data_dict_ca3_dg, keys=relevant_keys)
non_neural_amy = reshape_to_trial_num(data_dict_amy, keys=relevant_keys)
non_neural_entphc = reshape_to_trial_num(data_dict_entphc, keys=relevant_keys)

relevant_keys = ['correct', 'clust', 'subj', 'sess', 'list_num', 'serial_pos']
non_neural_ca1 = reshape_to_trial_num(data_dict_ca1, keys=relevant_keys)
non_neural_ca3_dg = reshape_to_trial_num(data_dict_ca3_dg, keys=relevant_keys)
non_neural_amy = reshape_to_trial_num(data_dict_amy, keys=relevant_keys)
non_neural_entphc = reshape_to_trial_num(data_dict_entphc, keys=relevant_keys)

print("RIPPLE EXISTS")
ripple_exists_ca1_400_1100 = create_ripple_exists(data_dict_ca1, ripple_start=400, ripple_end=1100)
ripple_exists_ca3_dg_400_1100 = create_ripple_exists(data_dict_ca3_dg, ripple_start=400, ripple_end=1100)
ripple_exists_amy_400_1100 = create_ripple_exists(data_dict_amy, ripple_start=400, ripple_end=1100)
ripple_exists_entphc_400_1100 = create_ripple_exists(data_dict_entphc, ripple_start=400, ripple_end=1100)

ripple_exists_ca1_100_1700 = create_ripple_exists(data_dict_ca1, ripple_start=100, ripple_end=1700)
ripple_exists_ca3_dg_100_1700 = create_ripple_exists(data_dict_ca3_dg, ripple_start=100, ripple_end=1700)
ripple_exists_amy_100_1700 = create_ripple_exists(data_dict_amy, ripple_start=100, ripple_end=1700)
ripple_exists_entphc_100_1700 = create_ripple_exists(data_dict_entphc, ripple_start=100, ripple_end=1700)

ripple_exists_ca1_recall = create_ripple_exists(data_dict_ca1, ripple_start=-600, ripple_end=-100, start_time=-2000, end_time=200)
ripple_exists_ca3_dg_recall = create_ripple_exists(data_dict_ca3_dg, ripple_start=-600, ripple_end=-100, start_time=-2000, end_time=200)
ripple_exists_amy_recall = create_ripple_exists(data_dict_amy, ripple_start=-600, ripple_end=-100, start_time=-2000, end_time=200)
ripple_exists_entphc_recall = create_ripple_exists(data_dict_entphc, ripple_start=-600, ripple_end=-100, start_time=-2000, end_time=2000)

# create HFA array
print("HFA")
hfa_ca1_400_1100 = average_hfa_across_elecs(data_dict_ca1, HFA_start=400, HFA_end=1100)
hfa_ca3_dg_400_1100 = average_hfa_across_elecs(data_dict_ca3_dg, HFA_start=400, HFA_end=1100)
hfa_amy_400_1100 = average_hfa_across_elecs(data_dict_amy, HFA_start=400, HFA_end=1100)
hfa_entphc_400_1100 = average_hfa_across_elecs(data_dict_entphc, HFA_start=400, HFA_end=1100)

hfa_ca1_100_1700 = average_hfa_across_elecs(data_dict_ca1, HFA_start=100, HFA_end=1700)
hfa_ca3_dg_100_1700 = average_hfa_across_elecs(data_dict_ca3_dg, HFA_start=100, HFA_end=1700)
hfa_amy_100_1700 = average_hfa_across_elecs(data_dict_amy, HFA_start=100, HFA_end=1700)
hfa_entphc_100_1700 = average_hfa_across_elecs(data_dict_entphc, HFA_start=100, HFA_end=1700)

hfa_ca1_recall = average_hfa_across_elecs(data_dict_ca1_recall, HFA_start=-600, HFA_end=-100, start_time=-2000, end_time=2000)
hfa_ca3_dg_recall = average_hfa_across_elecs(data_dict_ca3_dg_recall, HFA_start=-600, HFA_end=-100, start_time=-2000, end_time=2000)
hfa_amy_recall = average_hfa_across_elecs(data_dict_amy_recall, HFA_start=-600, HFA_end=-100, start_time=-2000, end_time=2000)
hfa_entphc_recall = average_hfa_across_elecs(data_dict_entphc_recall, HFA_start=-600, HFA_end=-100, start_time=-2000, end_time=2000)

# create clustered array (1 if clust, 0 if not clust, -1 everything else)
# modifies clust key of non_neural_X dict 
non_neural_ca1 = create_semantic_clustered_array(non_neural_ca1)
non_neural_ca3_dg = create_semantic_clustered_array(non_neural_ca3_dg)
non_neural_amy = create_semantic_clustered_array(non_neural_amy)
non_neural_entphc = create_semantic_clustered_array(non_neural_entphc)


# combine behavorial data, hfa, and ripples 
ca1_proc = combine_data(non_neural_ca1, hfa_short=hfa_ca1_400_1100, hfa_long=hfa_ca1_100_1700, 
                                        ripple_exists_short=ripple_exists_ca1_400_1100, 
                                        ripple_exists_long=ripple_exists_ca1_100_1700)
ca3_dg_proc = combine_data(non_neural_ca3_dg, hfa_short=hfa_ca3_dg_400_1100, hfa_long=hfa_ca3_dg_100_1700, 
                                        ripple_exists_short=ripple_exists_ca3_dg_400_1100, 
                                        ripple_exists_long=ripple_exists_ca3_dg_100_1700)
amy_proc = combine_data(non_neural_amy, hfa_short=hfa_amy_400_1100, hfa_long=hfa_amy_100_1700, 
                                        ripple_exists_short=ripple_exists_amy_400_1100, 
                                        ripple_exists_long=ripple_exists_amy_100_1700)
entphc_proc = combine_data(non_neural_entphc, hfa_short=hfa_entphc_400_1100, hfa_long=hfa_entphc_100_1700, 
                                        ripple_exists_short=ripple_exists_entphc_400_1100, 
                                        ripple_exists_long=ripple_exists_entphc_100_1700)

# remove -1 indices in clustered 
ca1_clust_only = remove_non_binary_clust(ca1_proc)
ca3_dg_clust_only = remove_non_binary_clust(ca3_dg_proc)
amy_clust_only = remove_non_binary_clust(amy_proc)
entphc_clust_only = remove_non_binary_clust(entphc_proc)

# save data for analyses
if save_data: 
    
    save_dir = '/home1/efeghhi/ripple_memory/session_data/'
    pd.DataFrame(ca1_clust_only).to_csv(f'{save_dir}ca1.csv')
    pd.DataFrame(ca3_dg_clust_only).to_csv(f'{save_dir}ca3_dg.csv')
    pd.DataFrame(amy_clust_only).to_csv(f'{save_dir}amy.csv')
    pd.DataFrame(entphc_clust_only).to_csv(f'{save_dir}entphc.csv')

print("Finished Running")
