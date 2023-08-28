import numpy as np


def create_ripple_exists(data_dict, ripple_start=400, ripple_end=1100, sr=500, start_time=-700, 
                         end_time=2300):
    
    '''
    
    Inputs: 
    
    :param dict data_dict: 
    :param int ripple_start: start time (ms) relative to recording start time for ripple analysis
    :param int ripple_end: end time (ms) relative to recording start time for ripple analysis
    
    Outputs:
    
    
    '''
    
    ripple_exists_list = []
    
    # convert times to indices
    sr_factor = 1000/sr
    ripple_start_idx = int((ripple_start - start_time) / sr_factor)
    ripple_end_idx = int((ripple_end - start_time) / sr_factor)
    
    # list of length num_session
    # each element in the list is of shape num_trials x num_elecs x num_timesteps
    # where num_timesteps is 1500 b/c sr is 500 Hz and recording is 3000 ms
        
    for ripple_sess in data_dict['ripple']:
        
        # for a given electrode, sum over all electrodes in the specified time range
        # if there are any ripples across electrodes for that trial the value will be > 0, 
        # else the value will 0 
        ripple_exists_sess = np.zeros(ripple_sess.shape[0])
        ripple_exists_idxs = np.argwhere(np.sum(ripple_sess[:, :, ripple_start_idx:ripple_end_idx], axis=(1,2)) > 0)
        ripple_exists_sess[ripple_exists_idxs] = 1
        ripple_exists_list.append(ripple_exists_sess)
        
    return np.hstack(ripple_exists_list)

def average_hfa_across_elecs(data_dict, HFA_start=400, HFA_end=1100, sr=50, start_time=-700, 
                         end_time=2300):
    
    # convert times to indices
    sr_factor = 1000/sr
    HFA_start_idx = int((HFA_start - start_time) / sr_factor)
    HFA_end_idx = int((HFA_end - start_time) / sr_factor)
    HFA = []
    
    for HFA_sess in data_dict['HFA']:
        # average HFA across electrodes
        HFA.append(np.mean(HFA_sess[:, :, HFA_start_idx:HFA_end_idx], axis=(1,2)))

    return np.hstack(HFA)

def reshape_to_trial_num(data_dict, keys):
    
    '''
    :param dict data_dict: dictionary containing session related information 
    :param dict keys: keys of data_dict to add to returned output
    
    This function takes as input data_dict and keys, and modifies the items 
    corresponding to these lists to be np arrays of shape num_trials.
    '''

    data_dict_raveled = {}

    # each key is a list with num_session entries 
    for key, val in data_dict.items():
        if key in keys:
            data_dict_raveled[key] = []
            # add the information from the first electrode (since information is repeated across elecs)
            for sess in val:
                if len(sess.shape) != 2:
                    print(f"{key} is not of the correct shape")
                data_dict_raveled[key].extend(sess[:, 0])
    
    for key, val in data_dict_raveled.items():
        data_dict_raveled[key] = np.asarray(val)
            
    return data_dict_raveled

def create_semantic_clustered_array(data_dict, 
                                    clustered=['A','C'], unclustered=['D', 'Z']):
            
    '''
    :param dict data_dict: dictionary which needs to have the following keys ->
    
    
        clust: indicates what recalls count as clustered. There are four possible recalls:
            1) 'A': adjacent semantic
            2) 'C': remote semantic
            3) 'D': remote unclustered
            4) 'Z': dead end 
            
        position: position that each word was recalled in 
    :param list clustered: which entries count as clustered
    :param list unclustered: which entries count as unclustered 
    
        The default is to use A and C as clustered, and D and Z as unclustered. 
        
    Modifies clust key to be 1 for clustered, 0 for unclustered, and -1 for everything else
    '''
    
    list_length = 12        
    recall_position_np = data_dict['position']
    semantic_array_np = data_dict['clust']
    num_selected_trials = recall_position_np.shape[0]
    number_of_lists = int(num_selected_trials/list_length)
    semantic_clustered_array_np = np.zeros((number_of_lists, list_length))
    dead_ends = 0
    remote_semantic = 0
    adjacent_semantic = 0
    remote_nonsemantic = 0
    adjacent_nonsemantic = 0
    counter = 0
    
    for list_idx in range(0, num_selected_trials, list_length):
        
        # recall_position_np and semantic_array_np contain information 
        # about the word that was recalled and its clustering type, respectively
        # this information is repeated list_length times, so our for loop will 
        # increment by list length 
        recalled_idx = recall_position_np[list_idx] 
        cluster = semantic_array_np[list_idx]
        
        # init values to -1 so that non recalled items are -1 
        cluster_trial_np = np.ones(list_length)*-1
        
        for r, c in zip(recalled_idx, cluster):
            if r > 0 and r <= list_length:
                if c in clustered:
                    cluster_trial_np[r-1] = 1 # change to 1 for clustered recall
                elif c in unclustered:
                    cluster_trial_np[r-1] = 0 # change to 0 for unclustered but recalled
                if c=='A':
                    adjacent_semantic += 1
                if c=='C':
                    remote_semantic += 1
                if c=='D':
                    remote_nonsemantic += 1
                if c=='Z':
                    dead_ends += 1
                
        semantic_clustered_array_np[counter] = cluster_trial_np
        
        counter += 1
        
    semantic_clustered_array_np = np.ravel(semantic_clustered_array_np)
    
    # replace with processed clustered array 
    data_dict['clust'] = semantic_clustered_array_np
    return data_dict

def combine_data(data_dict, **kwargs):
    
    for key, val in kwargs.items():
        data_dict[key] = val
        
    return data_dict
    
def remove_non_binary_clust(data_dict):
    
    clustered = data_dict['clust']
    mask_idxs = np.argwhere(clustered==-1)
    
    for key, val in data_dict.items():
        data_dict[key] = np.delete(val, mask_idxs, axis=0)
        
    return data_dict