import pandas as pd 
pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)
from cmlreaders import get_data_index
import xarray as xarray
import matplotlib.pyplot as plt
from pylab import *
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42 # fix fonts for Illustrator
from general import *
from SWRmodule import *
from HFA_ripples_analysis import HFA_ripples_analysis


# init variables
sub_selection = 'whole'
df = get_data_index("r1") # all RAM subjects
exp = 'catFR1' # 'FR1' 'catFR1' 'RepFR1'

data_folder = 'SWR_scratch'
select_subfield = True
ripple_bin_start_end = [100, 1100]
hpc_region = ['ca1', 'ca3', 'dg']
ripple_region = ['ca1', 'ca3', 'dg']
hpc_ripple_type = 'single_elec'
region_name = 'HPC'


RS = HFA_ripples_analysis(exp=exp, df=df, sub_selection=sub_selection, data_folder=data_folder,
                                        select_subfield=select_subfield, hpc_regions=hpc_region, ripple_regions=ripple_region, 
                                        ripple_bin_start_end=ripple_bin_start_end)

RS.remove_subject_sessions()
RS.load_data_from_cluster(base_path='/scratch/efeghhi/', selected_period='encoding', region_name=region_name, 
                          hpc_ripple_type='single_elec', ripple_bool=False)
RS.getStartArray()
RS.select_idxs_numpy()