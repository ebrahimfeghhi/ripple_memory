import pickle 


def save_data_to_pickle(data, directory):
    
    with open(directory, 'wb') as f:
        pickle.dump(data, f)
