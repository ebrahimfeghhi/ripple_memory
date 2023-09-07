import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def convert_elec_2d(array_3d):
    
    array_3d_swapped = np.swapaxes(array_3d, 0, 1)
    return np.reshape(array_3d_swapped, (-1, array_3d_swapped.shape[-1]))

def clean_elec_names(elec_names, channel_coords):
    
    '''
    :param list coords: MNI coordinates for each electrode
    :param list names: name of region where electrode is place

    For electrodes which do not have hemisphere specified, add in hemisphere
    using MNI coordinates. Also some region names have extraneous double quotes,
    so remove those. 
    '''
        
    updated_names = []
    
    for name, coord in zip(elec_names, channel_coords):
    
        name = name.replace('"', '')  # remove extraneous double quotes if they exist
        name = name.strip() # remove spaces from the beginning and end of string
        
        # if right or left is not specified, add it in 
        if 'right' not in name and 'left' not in name:
            
            x_coord = coord[0]
            
            if x_coord > 0:
                loc = 'right'
            else:
                loc = 'left'
                
            name_with_loc = f"{loc} {name}"
            updated_names.append(name_with_loc)
            
        else:
    
            updated_names.append(name)
            
    return updated_names

def replace_with_ca1_ripples(cleaned_elec_names_hpc, hpc_ripples, cleaned_elec_names, num_trials):
    
    ca1_right = []
    ca1_left = []
    for idx, elec_hpc in enumerate(cleaned_elec_names_hpc):
        if 'right ca1' in elec_hpc:
            ca1_right.append(idx)
        if 'left ca1' in elec_hpc:
            ca1_left.append(idx)

    # sum across electrodes, then replace values above 1 with 1
    if len(ca1_right) > 0:
        ca1_right_ripples = np.where(np.sum(hpc_ripples[:, ca1_right], axis=1)>0, 1, 0)
    if len(ca1_left) > 0:
        ca1_left_ripples = np.where(np.sum(hpc_ripples[:, ca1_left], axis=1)>0, 1, 0)

    ripple_array = np.zeros((num_trials, len(cleaned_elec_names), hpc_ripples.shape[2]))
    # Return None if there are no ca1 electrodes on the same side
    for idx, elec in enumerate(cleaned_elec_names):
         
        if 'right' in elec:
            if len(ca1_right)>0:
                ripple_array[:, idx] = ca1_right_ripples
            else:
                return None
                
        if 'left' in elec:
            if len(ca1_left) > 0:
                ripple_array[:, idx] = ca1_left_ripples
            else:
                return None 

    return ripple_array
    
def load_data(directory, region_name, encoding_mode, condition_on_ca1_ripples=True):
    
    '''
    
    Inputs:
    
        :param str directory: directory where data is stored
        :param str region_name: which brain region to load 
        :param int encoding_mode: 1 means encoding mode, 0 means recall mode
        
    Ouputs:

        dict containing session-related data. Each value in
        the dict is a list, where the ith element in that list
        is the infomation for the ith session. The first two 
        dimensions for each value is num_trials x num_elecs, except
        for elec_names, which is of shape num_elecs. 
        
    '''
    
    
    if encoding_mode: 
        data_dict = {'ripple': [], 'HFA': [], 'theta': [], 'theta_phase': [], 'clust': [], 
                     'correct': [], 'position': [], 'list_num': [], 'subj': [], 'sess': [], 'elec_names':[], 'serial_pos': []}
        num_timesteps_HFA = 150
        
    else:
        data_dict = {'ripple': [], 'HFA': [], 'theta': [], 'theta_phase': [], 'list_num': [], 
                     'subj': [], 'sess': [], 'elec_names':[], 'clust': []}
        num_timesteps_HFA = 200
         
    file_list = os.listdir(directory)
    for f in file_list:
        
        if len(region_name) != 0 and region_name not in f:
            continue
         
        # Open the pickle file for reading
        with open(f'{directory}/{f}', 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)  
        
        # we'll also load HPC data if conditioning on ca1 ripples
        # for non-HPC files 
        if 'HPC' not in f and condition_on_ca1_ripples:
            try: 
                # try both ENTPHC and AMY since don't know which
                # brain region we are loading
                f_ca1 = f.replace("ENTPHC", "HPC")
                f_ca1 = f_ca1.replace("AMY", "HPC")     
                with open(f'{directory}/{f_ca1}', 'rb') as pickle_file:
                    loaded_data_ca1 = pickle.load(pickle_file) 
            except:
                # if conditioning on ca1_ripples, skip sessions that don't have hpc electrodes
                continue
        else:
            loaded_data_ca1 = None

        # rows of HFA give us number of words presented in that session
        num_trials = loaded_data['HFA_pow'].shape[0]
        
        # there's a session in entphc where trials are longer for encoding (185 vs 150) (not sure why)
        # going to exclude any sessions like this
        HFA_timesteps = loaded_data['HFA_pow'].shape[2]
        if HFA_timesteps != num_timesteps_HFA:
            continue
        
        # elec_names is the only entry which is of shape num_elecs
        cleaned_elec_names = clean_elec_names(loaded_data['elec_names'], loaded_data['channel_coords'])
        
        # reload ripple array with ca1 ripples from the ipsilateral side if conditioning on ca1 ripples
        if loaded_data_ca1 is not None:
            cleaned_elec_names_hpc = clean_elec_names(loaded_data_ca1['elec_names'], loaded_data_ca1['channel_coords'])
            hpc_ripples = loaded_data_ca1['ripple_array']
            ripple_array = replace_with_ca1_ripples(cleaned_elec_names_hpc, hpc_ripples, cleaned_elec_names, num_trials)
            
            # ripple array is returned as None if there are no ca1 electrodes on the same side
            # as a non-ca1 electrode. If this happens, skip the entire session. 
            if ripple_array is None:
                continue
            
            data_dict['ripple'].append(ripple_array)
            
        else:
             # 3d array of shape num_trials x num_electrodes x num_timesteps
            data_dict['ripple'].append(np.asarray(loaded_data['ripple_array']))
            
        data_dict['HFA'].append(np.asarray(loaded_data['HFA_pow']))
        data_dict['theta'].append(np.asarray(loaded_data['theta_pow']))
        data_dict['theta_phase'].append(np.asarray(loaded_data['theta_phase_array']))

        if encoding_mode:
            
            # vstack to get 2d shape num_trials x num_elecs
            data_dict['correct'].append(np.vstack(loaded_data['encoded_word_key_array']).T)
            data_dict['serial_pos'].append(np.vstack(loaded_data['serialpos_array']).T)

            # reshape 1d array to num_trials x num_elec 
            data_dict['position'].append(np.reshape(loaded_data['recall_position_array'], (-1, num_trials)).T)
           
        # reshape 1d array  num_trials x num_elec 
        data_dict['clust'].append(np.reshape(loaded_data['semantic_clustering_key'], (-1, num_trials)).T)
        
        data_dict['list_num'].append(np.reshape(loaded_data['list_num_key'], (-1, num_trials)).T)
                
        # each of these entries is of shape num_electrodes, so need to repeat num_trials times
        data_dict['subj'].append(np.repeat(np.expand_dims(loaded_data['sub_names'], -1), num_trials, axis=-1).T)
        data_dict['sess'].append(np.repeat(np.expand_dims(loaded_data['sub_sess_names'], -1), num_trials, axis=-1).T)
        
        data_dict['elec_names'].append(np.asarray(cleaned_elec_names))
        
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
    serial_pos = []

    # loop through sessions 
    for sess, list_num_sess in enumerate(list_nums_sessions):
                
        mask_idxs = []
        list_num = 1 
        ll = 0 # position in list
        list_num_sess_1d = list_num_sess[:, 0] # columns are repeats
        num_trials = 0
            
        for idx, ln in enumerate(list_num_sess_1d):
            
            # if trial is part of the current list
            # increment the position in the list 
            if ln == list_num:
                ll += 1 
            
            # new list 
            else:
                # list is of incorrect length 
                if ll != list_length:
                    mask_idxs.extend([i for i in range(idx-ll, idx)])
                    num_lists_wrong += 1
     
                # reset list position marker and update list_num
                ll = 1
                list_num = list_num_sess_1d[idx]
                    
            num_trials += 1
            
        # for the last list in the session, there is no new list
        # so just check to see if it was of the correct length
        if ll != list_length: 
            mask_idxs.extend([i for i in range(idx-ll, idx)])
            num_lists_wrong += 1
  
        if len(mask_idxs) > 0:  
            for key, val in data_dict.items():
                # elec names are only repeated num_channel times, 
                # so don't need to delete them
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
        
        # store indices corresponding to the desired electrodes
        selected_ind = [int(i) for i, x in enumerate(elec_name) if x in selected_elecs]
        
        if len(selected_ind) == 0:
            continue
        
        # 1D data so we'll selec the proper indices outside the for loop
        data_dict_selected_elecs['elec_names'].append(data_dict['elec_names'][sess][selected_ind])
        
        # remainder of data is 2D
        for key, val in data_dict.items():
            if key != 'elec_names':
                data_dict_selected_elecs[key].append(val[sess][:, selected_ind])
                
    return data_dict_selected_elecs

def count_num_trials(data_dict, dd_name, use_key='HFA'):
    
    '''
    
    Inputs:
    
        :param dict data_dict: dictionary with session related data 
        :param str dd_name: name of data_dict
        :param str use_key: the key to use to count number of trials 
        
    Ouputs:

        Number of trials in data_dict before averaging across elecs
        
    '''
    
    num_trials = 0
    
    key_val = data_dict[use_key]
    for sess in key_val:
        num_trials += sess.shape[0]*sess.shape[1]
        
    print(f"Number of trials in {dd_name}: {num_trials}")
        
    
def dict_to_numpy(data_dict, order):
    
    print(f"order: {order}")
    
    dd_trials = {}
    for key, val in data_dict.items():
        dd_trials[key] = []
        for sess in val:
            if key == 'correct':
                sess = np.where(sess>0, 1, 0)
            # non neural data is 2D (num_trials x num_electrodes)
            # values are repeated along each row 
            if len(sess.shape) == 2:
                # reshape here will give us a 1d vector, where data for electrodes for a given trial
                # are placed next to each other 
                dd_trials[key].extend(np.reshape(sess, -1, order=order))
            if len(sess.shape) == 3:
                # for 3d data reshape will do the same thing, meaning that electrodes from the same trial
                # are placed next to each other             
                dd_trials[key].extend(np.reshape(sess, (-1, sess.shape[-1]), order=order))
 
    for key, val in dd_trials.items():
        dd_trials[key] = np.asarray(val)
        
    return dd_trials
