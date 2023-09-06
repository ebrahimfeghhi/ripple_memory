from load_data import * 
from analyze_data import * 
import sys
import warnings
import matplotlib.pyplot as plt
from SWRmodule import *
sys.path.append('/home1/efeghhi/ripple_memory/')
from brain_labels import HPC_labels, ENT_labels, PHC_labels, temporal_lobe_labels,\
                        MFG_labels, IFG_labels, nonHPC_MTL_labels, ENTPHC_labels, AMY_labels

############### set parameters ###############
save_data = True
start_time = -2000 # recording start time relative to word onset (ms)
end_time = 2000 # recording end time relative to word onset (ms)
##############################################
catFR_dir = '/scratch/efeghhi/catFR1/IRIonly/'

region_name = '' # if empty string, loads all data
encoding_mode = 0
data_dict = load_data(catFR_dir, region_name=region_name, encoding_mode=encoding_mode)
if encoding_mode:
    data_dict = remove_wrong_length_lists(data_dict)

ca1_elecs = [x for x in HPC_labels if 'ca1' in x]
data_dict_ca1 = select_region(data_dict, ca1_elecs)

# create clustered int array
clustered_int = create_semantic_clustered_array(data_dict_ca1, encoding_mode)
breakpoint()

