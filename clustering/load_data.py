import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def convert_elec_2d(array_3d):
    
    array_3d_swapped = np.swapaxes(array_3d, 0, 1)
    return np.reshape(array_3d_swapped, (-1, array_3d_swapped.shape[-1]))

def load_data(directory, region_name):
    
    '''
    
    Inputs:
    
        :param str directory: directory where data is stored
        :param str region_name: which brain region to load 
        
    Ouputs:

        dict containing session-related data. Each value in
        the dict is a list, where the ith element in that list
        is the infomation for the ith session. The first two 
        dimensions for each value is num_trials x num_elecs, except
        for elec_names, which is of shape num_elecs. 
        
    '''
    
    data_dict = {'ripple': [], 'HFA': [], 'theta_phase': [], 'clust': [], 'correct': [], 'position': [], 
        'list_num': [], 'subj': [], 'sess': [], 'elec_names':[]}

    file_list = os.listdir(directory)
    for f in file_list:
        
        if len(region_name) != 0 and region_name not in f:
            continue
         
        # Open the pickle file for reading
        with open(f'{directory}/{f}', 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)    
            
        # rows of HFA give us number of words presented in that session
        num_trials = loaded_data['HFA_pow'].shape[0]
            
        # 3d array of shape num_trials x num_electrodes x num_timesteps
        data_dict['ripple'].append(np.asarray(loaded_data['ripple_array']))
        data_dict['HFA'].append(np.asarray(loaded_data['HFA_pow']))
        data_dict['theta_phase'].append(np.asarray(loaded_data['theta_phase_array']))

        # vstack to get 2d shape num_trials x num_elecs
        data_dict['correct'].append(np.vstack(loaded_data['encoded_word_key_array']).T)
    
        # reshape 1d array of shape (num__elec x num_trials) to num_trials x num_elec
        data_dict['clust'].append(np.reshape(loaded_data['semantic_clustering_key'], (-1, num_trials)).T)
        data_dict['position'].append(np.reshape(loaded_data['recall_position_array'], (-1, num_trials)).T)
        data_dict['list_num'].append(np.reshape(loaded_data['list_num_key'], (-1, num_trials)).T)
        
        # each of these entries is of shape num_electrodes, so need to repeat num_trials times
        data_dict['elec_names'].append(np.asarray(loaded_data['elec_names']))
        data_dict['subj'].append(np.repeat(np.expand_dims(loaded_data['sub_names'], -1), num_trials, axis=-1).T)
        data_dict['sess'].append(np.repeat(np.expand_dims(loaded_data['sub_sess_names'], -1), num_trials, axis=-1).T)

    return data_dict

def remove_wrong_length_lists(data_dict, list_length=12):
    
    '''
    
    Inputs:

        :param dict data_dict: 
        :param int list_length: desired list_length
    
    Outputs: 
    
        Removes all trials from data_dict that are not of the specified list_length.
        
    '''
    
    list_nums_sessions = data_dict['list_num']
    num_lists_wrong = 0

    for sess, list_num_sess in enumerate(list_nums_sessions):
                
        mask_idxs = []
        list_num = 1 
        ll = 0 # position in list
        list_num_sess_1d = list_num_sess[:, 0] # columns are repeats
        num_trials = 0
            
        for idx, ln in enumerate(list_num_sess_1d):
            
            if ln == list_num:
                ll += 1
            
            else:
                # list is of incorrect length 
                if ll % list_length != 0:
                    mask_idxs.extend([i for i in range(idx-ll, idx)])
                    num_lists_wrong += 1
     
                # reset list position marker and update list_num
                ll = 1
                list_num = list_num_sess_1d[idx]
                    
            num_trials += 1
            
        # for the last list in the session 
        if ll % list_length != 0: 
            mask_idxs.extend([i for i in range(idx-ll, idx)])
            num_lists_wrong += 1
                    
        if len(mask_idxs) > 0:  
            for key, val in data_dict.items():
                if key != 'elec_names':
                    data_dict[key][sess] = np.delete(val[sess], mask_idxs, axis=0)
                    
    return data_dict
            
        
def select_region(data_dict, selected_elecs):
    
    '''
    
    Inputs:
    
        :param dict data_dict: dictionary with session related data 
        :param list selected_elecs: list containing which electrodes to keep 
        
    Ouputs:

        dictionary with data corresponding to selected electrodes.
        
    '''
    
    elec_names = data_dict['elec_names']
    data_dict_selected_elecs = {key: [] for key in data_dict.keys()}

    # for each session, store indices corresponding to the selected_elecs
    for sess, elec_name in enumerate(elec_names):
        
        selected_ind = [int(i) for i, x in enumerate(elec_name) if x in selected_elecs]
        
        if len(selected_ind) == 0:
            continue
        
        data_dict_selected_elecs['elec_names'].append(data_dict['elec_names'][sess][selected_ind])
        
        for key, val in data_dict.items():
            if key != 'elec_names':
                data_dict_selected_elecs[key].append(val[sess][:, selected_ind])
                
    return data_dict_selected_elecs




        
        
