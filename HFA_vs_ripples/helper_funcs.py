import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append("/home1/efeghhi/ripple_memory")
from load_data import * 
from analyze_data import * 

savefigs = '/home1/efeghhi/ripple_memory/figures/misc/'

def make_plot(HFA_select, correct_select, saveName, br):
     
    plt.title(br)
    HFA_c= HFA_select[correct_select==1]
    HFA_nc= HFA_select[correct_select==0]
    plt.plot(np.linspace(-700, 2300, HFA_c.shape[1]), np.mean(HFA_c, axis=0), color='tab:blue', label=f'Recalled: {HFA_c.shape[0]}')
    plt.plot(np.linspace(-700, 2300, HFA_nc.shape[1]), np.mean(HFA_nc, axis=0), color='tab:orange', label=f'Not recalled: {HFA_nc.shape[0]}')
    plt.xticks(np.arange(-700, 2300, 350))
    plt.xlabel("Time (ms)")
    plt.ylabel("Normalized HFA power")
    plt.legend()
    plt.savefig(f'{savefigs}/{saveName}_{br}', dpi=300)
    plt.show()
    
def plot_SME_high_low_HFA(data_dict_br, br, HFA_start_idx, HFA_end_idx):
    
    data_dict_np = dict_to_numpy(data_dict_br, order='C')
    
    HFA = data_dict_np['HFA']
    correct = data_dict_np['correct']
    
    HFA_mean_400_1100 = np.mean(HFA[:,HFA_start_idx:HFA_end_idx], axis=1)
    HFA_cutoff = np.quantile(HFA_mean_400_1100, 0.8)
    high_HFA_idxs = np.argwhere(HFA_mean_400_1100 > HFA_cutoff)
    low_HFA_idxs = np.argwhere(HFA_mean_400_1100 < HFA_cutoff)
    print("Num high HFA: ", high_HFA_idxs.shape[0])
    print("Num low HFA: ", low_HFA_idxs.shape[0])
    
    HFA_low = HFA[low_HFA_idxs].squeeze()
    HFA_high = HFA[high_HFA_idxs].squeeze()
    
    correct_low = correct[low_HFA_idxs].squeeze()
    correct_high = correct[high_HFA_idxs].squeeze()
    
    rng = np.random.default_rng(seed=0)
    low_HFA_subsample_idxs = rng.choice(low_HFA_idxs, high_HFA_idxs.shape[0])
    HFA_low_small = HFA[low_HFA_subsample_idxs].squeeze()
    correct_low_small = correct[low_HFA_subsample_idxs].squeeze()
   
    make_plot(HFA_low, correct_low, f'low_HFA_SME_{br}', br) 
    make_plot(HFA_high, correct_high, f'high_HFA_SME_{br}', br)
    make_plot(HFA_low_small, correct_low_small, f'low_small_HFA_SME', br)