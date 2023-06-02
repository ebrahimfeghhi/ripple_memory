import pandas as pd; pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)
import numpy as np
from cmlreaders import CMLReader, get_data_index
from ptsa.data.filters import ButterworthFilter, ResampleFilter, MorletWaveletFilter
import xarray as xarray
import sys
import os
import matplotlib.pyplot as plt
from pylab import *
from copy import copy
from scipy import stats
from scipy.stats import zscore
import seaborn as sns       
import pickle
plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42 # fix fonts for Illustrator
sys.path.append('/home1/john/johnModules')
from brain_labels import HPC_labels, ENT_labels, PHC_labels, temporal_lobe_labels,\
                         MFG_labels, IFG_labels, nonHPC_MTL_labels
from general import *
from SWRmodule import *
import statsmodels.formula.api as smf
from ripples_HFA_SCE import ripple_analysis_SCE
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef
from scipy.stats import zscore
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
from pymer4.models import Lmer



