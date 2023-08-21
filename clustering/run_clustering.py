from load_data import * 
from analyze_data import * 

############### set parameters ###############

save_data = True
start_time = -700 # recording start time relative to word onset (ms)
end_time = 2300 # recording end time relative to word onset (ms)

##############################################


#data_dict = {'ripple': [], 'HFA': [], 'theta_phase': [], 'clust': [], 'correct': [], 'position': [], 
#'list_num': [], 'subj': [], 'sess': [], 'elec_names':[]}

# load data and remove lists of wrong length
catFR_dir = '/scratch/efeghhi/catFR1/ENCODING'
region_name = '' # if empty string, loads all data
data_dict = load_data(catFR_dir, region_name)
data_dict = remove_wrong_length_lists(data_dict)


# divide data by region
ca1_elecs = ['"ca1"', 'left ca1', 'right ca1']
data_dict_ca1 = select_region(data_dict, ca1_elecs)
non_ca1_elecs = ['"dg"', '"sub"', 'left ca2', 'left ca3', 'left dg', 'right dg', 'right sub', 'left sub']
data_dict_hpc_non_ca1 = select_region(data_dict, non_ca1_elecs)
amy_elecs = [' left amygdala', ' right amygdala', 'left amy', 'left amygdala', 'right amy', 'right amygdala']
data_dict_amy = select_region(data_dict, amy_elecs)

non_neural_ca1 = reshape_to_trial_num(data_dict_ca1, keys=['position', 'correct', 'clust', 'subj', 'sess'])
non_neural_hpc_non_ca1 = reshape_to_trial_num(data_dict_hpc_non_ca1, keys=['position', 'correct', 'clust', 'subj', 'sess'])
non_neural_amy = reshape_to_trial_num(data_dict_amy, keys=['position', 'correct', 'clust', 'subj', 'sess'])

# create ripple exists
ripple_exists_ca1 = create_ripple_exists(data_dict_ca1)
ripple_exists_hpc_non_ca1 = create_ripple_exists(data_dict_hpc_non_ca1)
ripple_exists_amy = create_ripple_exists(data_dict_amy)

# create HFA array
hfa_ca1 = average_hfa_across_elecs(data_dict_ca1)
hfa_hpc_non_ca1 = average_hfa_across_elecs(data_dict_hpc_non_ca1)
hfa_amy = average_hfa_across_elecs(data_dict_amy)

# create clustered array 
non_neural_ca1 = create_semantic_clustered_array(non_neural_ca1)
non_neural_hpc_non_ca1 = create_semantic_clustered_array(non_neural_hpc_non_ca1)
non_neural_amy = create_semantic_clustered_array(non_neural_amy)

ca1_proc = combine_data(non_neural_ca1, hfa=hfa_ca1, ripple_exists=ripple_exists_ca1)
hpc_non_ca1_proc = combine_data(non_neural_hpc_non_ca1, hfa=hfa_hpc_non_ca1, ripple_exists=ripple_exists_hpc_non_ca1)
amy_proc = combine_data(non_neural_amy, hfa=hfa_amy, ripple_exists=ripple_exists_amy)

# remove -1 indices in clustered 
ca1_clust_only = remove_non_binary_clust(ca1_proc)
hpc_non_ca1_clust_only = remove_non_binary_clust(hpc_non_ca1_proc)
amy_clust_only = remove_non_binary_clust(amy_proc)

# save data for analyses
if save_data: 
    save_dir = '/home1/efeghhi/ripple_memory/session_data/'
    pd.DataFrame(ca1_clust_only).to_csv(f'{save_dir}ca1.csv')
    pd.DataFrame(hpc_non_ca1_clust_only).to_csv(f'{save_dir}hpc_non_ca1.csv')
    pd.DataFrame(amy_clust_only).to_csv(f'{save_dir}amy.csv')

print("Finished Running")
