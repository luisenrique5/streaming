#########################
#### Import Libraries ###
#########################

#linear algebra
import pandas as pd
import numpy as np
import numbers

#visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#visualization
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper", font_scale = 1.8)

#display parameters
from IPython.display import display

#regular expressions
import re

#change directory
import os
import sys
import ast
#run shell commands
import subprocess
import time

#for interpolation
from scipy.interpolate import interp1d

#save pandas dataframe as image
import dataframe_image as dfi

import warnings
#ingone warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

#import all standard column names, colors, rig activity dictionaries
from src.utils.config_backend import *

#pull folder names from config
save_folder = SAVE_FOLDER
input_folder = INPUT_FOLDER

#round and save parameters
round_ndigits = ROUND_NDIGITS
replace_dot = REPLACE_DOT

#set number of digits to display 
pd.set_option('precision', round_ndigits)

####################################
######### Custom Functions #########
####################################

#convert bit size to string showing fraction
def decimal_fraction_str(num):

    if isinstance(num, numbers.Number):

        integer_part = int(num // 1)
        decimal_part = (num % 1)
        ratio = decimal_part.as_integer_ratio()

        return str(integer_part) + '-' + str(ratio[0]) + '/'  \
            + str(ratio[1]) + "\""

    else:

        return num

#convert decimal time (hrs) to hh:mm
def convert_time_decimal_hhmm(time_decimal:float):

    if str(time_decimal) != 'nan':

        hours = int(time_decimal)
        minutes = int(round((time_decimal*60) % 60, 0))

        return str(hours).zfill(2) + ':' + str(minutes).zfill(2)

    else:
        return np.nan

#convert decimal time (min) to mm:ss
def convert_time_decimal_mmss(time_decimal:float):

    if str(time_decimal) != 'nan':

        minutes = int(time_decimal)
        seconds = int(round((time_decimal*60) % 60, 0))

        return str(minutes).zfill(2) + ':' + str(seconds).zfill(2)

    else:
        return np.nan

#make dummy data that contains all required rig activities
def make_dummy_data(bit_sizes, 
                    day = None, 
                    current_well = CURRENT_WELL_ID, 
                    dtime_rt = DTIME_RT,
                    WELLID = WELLID, DATETIME = DATETIME,
                    BS = BS, RSUPER = RSUPER, RSUBACT = RSUBACT,
                    super_state_swap = super_state_swap,
                    sub_activity_state_swap = sub_activity_state_swap,
                    time_based_drill_cols = time_based_drill_cols,
                    survey_cols = survey_cols):
    """This function makes dummy data for time-based and official survey tables for current well.
    Here bit_sizes is a list of available bit sizes, 
    day is datetime object that is used to build DATETIME column,
    current_well is current well id,
    dtime_rt is current well data density (10 seconds by default) defined in config,
    WELLID, DATETIME, BS, RSUPER, RSUBACT represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    super_state_swap is super state dictionary defined in config,
    sub_activity_state_swap is sub activity dictionary defined in config,
    time_based_drill_cols is a list of standard columns for time-based table defined in config,
    survey_cols is a list of standard columns for official survey table defined in config,
    Function returns a dummy time-based and official survey tables."""

    #make bit_sizes_filter by removing 'all' section
    bit_sizes_filter = [bit for bit in bit_sizes if bit != 'all']
    nrows_initial = len(bit_sizes_filter)

    #populate hole_diameter column
    bits = []
    for bit in bit_sizes_filter:
        bits.extend([bit]*2)

    if nrows_initial < 3:
        bits.extend([bits[-1]]*(6 - nrows_initial*2))

    #final number of rows in dummy data
    nrows = len(bits)

    super_state_include = ['OUT OF HOLE', 'DRILL', 'TRIP']
    sub_act_include = ['OUT OF HOLE', 'OTHER', 'ROTATE', 'SLIDE', 'TRIP IN', 'TRIP OUT']

    if nrows_initial > 3:
        super_state_include.extend([super_state_include[-1]]*(nrows_initial - 3))
        sub_act_include.extend([sub_act_include[-1]]*(nrows_initial - 3)*2)

    #populate super states column
    super_act = []
    for activity in super_state_include:
        super_act.extend([super_state_swap[activity]]*2)

    #populate sub activities
    sub_act = []
    for activity in sub_act_include:
        sub_act.append(sub_activity_state_swap[activity])

    df_rt = pd.DataFrame(columns = time_based_drill_cols,
                         data = np.zeros([nrows, len(time_based_drill_cols)]),
                         index = np.arange(nrows))

    if dtime_rt == None:
        #default 10 seconds time step
        time_step = 10
    else:
        time_step = dtime_rt

    if day == None:
        day = pd.to_datetime("today", format='%Y-%m-%d %H:%M:%S').round(freq='S')
    add_dates = []
    for i in np.arange(nrows,0,-1):
        add_dates.append(day - pd.Timedelta(time_step*(i-1), unit='sec'))

    df_rt[DATETIME] = add_dates
    df_rt[BS] = bits
    df_rt[RSUPER] = super_act
    df_rt[RSUBACT] = sub_act

    df_survey_rt = pd.DataFrame(columns = survey_cols,
                                data = np.zeros([nrows, len(survey_cols)]),
                                index= np.arange(nrows))

    df_survey_rt[WELLID] = current_well

    return df_rt, df_survey_rt

#*************************************************************************************************************#
#**********************************  dashboard data and plots: Overview tab  *********************************#
#*************************************************************************************************************#

def filter_ordered_hole_diameters(df_input: pd.DataFrame, hole_diameters: list,
                                  WELLID = WELLID, BS = BS) -> pd.DataFrame:
    """This function takes historic data and selected hole_diameters list
    and it filter hole diameters by order.
    Here df_input is time-based drilling data, 
    hole_diameters is the list of selected hole diameters in a proper order (descending)
    WELLID, BS represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    Function returns a filtered time-based drill dataframe.
    """

    df = df_input.copy()
    #note that hole_diameters should be passed in a proper order
    BSO = BS + '_order'
    WELLIDBS = WELLID + '_' + BS

    df_cut = df[[WELLID, BS]]
    df_order = pd.DataFrame()

    for well_id in df_cut[WELLID].unique():
        #select one well
        df_ = df_cut[df_cut[WELLID] == well_id]
        #hole diameter mapping per well
        hds = np.sort(df_[BS].unique())[::-1]
        hd_map = dict(zip(hds,np.arange(1,len(hds)+1)))
        #add hole diameter order
        df_[BSO] = df_[BS].map(hd_map)
        
        df_order = df_order.append(df_)
        
    df_select = df_order.groupby([WELLID,BS])[BSO].min().reset_index()

    #assign hole diameter - order
    hole_diameters_select_tuple = list(tuple(zip(hole_diameters,np.arange(1,len(hole_diameters)+1))))
    print('The following hole diameters with specified order are kept (hole_diameter, order): \n',
          hole_diameters_select_tuple)

    #define keep column according to hole diameter - order combination
    df_select['keep'] = df_select[[BS,BSO]].apply(tuple,axis=1).isin(hole_diameters_select_tuple).astype(int)

    df_select[WELLIDBS] = df_select[[WELLID, BS]].apply(tuple, axis=1)

    #keep (well_id, hole_diameter) combinations
    keep_hole_diameter = df_select.loc[df_select['keep'] == 1, 'well_id_hole_diameter'].values

    #filter historic data
    #add new column
    df[WELLIDBS] = df[[WELLID, BS]].apply(tuple, axis=1)

    #apply filter
    df = df[df[WELLIDBS].isin(keep_hole_diameter)]

    #assert filtering
    assert(set(df[WELLIDBS].unique()) == set(keep_hole_diameter))

    #drop auixillary (well_id, hole_diameter) column
    df.drop(columns = [WELLIDBS], inplace = True)

    return df

#make folder with specified name including sequence of folders
def mkfolder(folder):
    """This function makes a directory if does not exist yet.
    Here folder is a folder name to be created,
    it can include an entire path, 
    then it will all necessary folders."""

    #make the directories
    split_folder = folder.strip('/').split(sep='/')

    folder = split_folder[0]
    for f in split_folder[1:]:
        folder = "/".join([folder,f])
        try:
            os.mkdir(folder)
        except:
            print(f'Folder {folder} already exists.')

#function to compute the consecutive activity label and duration
def comp_duration(df_input: pd.DataFrame, 
                  activity_col: str, dtime: float, 
                  time_col = TIME, LABEL = LABEL, DUR = DUR) -> pd.DataFrame:
    """This function computes continuous activity label and its duration.
    Here df_input is time-based drilling data, 
    activity_col is the activity column name (e.g. RSUPER or RSUBACT), 
    dtime is a sampling time step in seconds,
    time_col, LABEL, DUR represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    Function returns dataframe has 2 extra columns: 
    LABEL that has consequtive label for a continuous activity,
    DUR provides duration of given continuous activity in seconds."""
    
    #copy dataframe
    df = df_input.copy()

    subactivity=df[activity_col].values #Enovate Rig SubActivity
    consecutive_label=np.zeros(subactivity.shape[0]) #define label vector

    count=0
    for i in range(1,subactivity.shape[0]):
        if subactivity[i]!=subactivity[i-1]:
            count+=1
            consecutive_label[i] = count
        else:
            consecutive_label[i] = count

    df[LABEL]=consecutive_label
    #change column type to int
    df[LABEL] = df[LABEL].astype('int64')

    min_time = pd.Timedelta(seconds=dtime)
    duration_map=df.groupby(LABEL).apply(lambda x: x[time_col].nunique() * min_time)
    df[DUR]=df[LABEL].map(duration_map) 
    
    return df

##*** updated on September 20 2021 ***
#rig label function to classify rig sub activities and super activities
def rig_labels(df_input: pd.DataFrame, df_rig: pd.Series,
               rig_name: str, dtime: float, real_time: bool,
               use_filter_dict = True,
               operator = OPERATOR,
               hd_diff_apply = True, wob_thr_apply = False, show_activity_names = False,
               RSUPER = RSUPER, RSUBACT = RSUBACT,
               TIME = TIME, HD = HD, BD = BD, HL = HL, 
               RPM = RPM, MRPM = MRPM, BRPM = BRPM,
               GPM = GPM, SPP = SPP, WOB = WOB, 
               super_state = super_state, 
               sub_activity_state = sub_activity_state, 
               verbose = False) -> pd.DataFrame:
    """This function computes rig activity labels for super and subactivities.
    Here df_input is time-based drilling data, 
    df_rig is rig design parameters data table,
    dtime is a sampling time step in seconds,
    real_time is an indicator to choose calculation option:
    real_time = False would output a whole data frame with activity labels in RSUPER and RSUBACT columns
    real_time = True would output a last data frame row with the activity labels in RSUPER and RSUBACT columns
    hd_diff_apply is a flag, if True it uses Hole Depth (HD) difference > 0 condition for ROTATE and SLIDE activities,
    otherwise if False uses (df[HD]-df[BD]) <= depth_onb_thr criteria
    wob_thr_apply is a flag, if True it uses wob_thr, if False it does not use wob_thr at all.
    show_activity_names is a 'flag', if True it adds 3 following columns to dataframe:'rig_super_state_name','rig_sub_activity_raw_name','rig_sub_activity_name'.
    RSUPER, RSUBACT, TIME, HD, BD, HL, RPM, MRPM, BRPM, GPM, SPP, WOB represent standard column names,
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    super_state and sub_activity_state are the dictionaries that define mapping from names to number codes.
    verbose = True activates all messages during computation.
    Function returns dataframe has 3 extra columns: 
    RSUPER (rig super activity code), RSUBACT (rig subactivity code), 
    LABEL (consequtive label for a continuous activity).
    """
    
    ### Activity typical sequence
    #Drilling-> Lift Out(Wash Out/Back Reaming) -> Pumps Off(Static/Other) 
    # -> Connection Drilling -> Pumps On(Static/Other)
    # -> Survey(Circulating) -> Letting In (Wash In/Reaming) -> Drilling
    
    #######################################
    ##        define thresholds:         ##
    ##   *** update March 24, 2021 ***   ##
    #######################################
    #depth thresholds, units ft
    depth_onb_thr = df_rig['depth_onb_thr'] #On bottom depth difference threshold Rotary & Sliding, Circulating
    depth_conn_thr = df_rig['depth_conn_thr'] #Connection depth threshold for HD-BD
    depth_conn_start = df_rig['depth_conn_start'] #Connection depth to start
    bd_conn = df_rig['bd_conn'] # Bit depth connection threshold
    depth_super_thr = df_rig['depth_super_thr'] #Depth difference threshold for super activity ft for 2 pipes #100 for 3 pipes
    depth_ooh_thr = df_rig['depth_ooh_thr'] #Depth out of hole threshold
    depth_trip_thr = df_rig['depth_trip_thr'] #Depth trip in-out free threshold #new

    #hook load threshold, klbs hl_conn_drill_thr
    hl_conn_drill_thr = df_rig['hl_conn_drill_thr']  #Hook load threshold for Connection Drilling <500ft BD
    hl_conn_drill1_thr = df_rig['hl_conn_drill1_thr']  #Hook load threshold for Connection Drilling >500ft BD
    hl_conn_trip_thr = df_rig['hl_conn_trip_thr'] #Hook load threshold for Connection Tripping <500ft BD
    hl_conn_trip1_thr = df_rig['hl_conn_trip1_thr']  #Hook load threshold for Connection Tripping >500ft BD
    hl_null_thr = df_rig['hl_null_thr'] #Hook load threshold for Null Values
    
    #sub activities
    gpm_thr = df_rig['gpm_thr'] #Flow rate threshold (gpm)
    rpm_thr = df_rig['rpm_thr'] #RPM threshold (rpm)
    spp_thr = df_rig['spp_thr'] #SPP threshold (psi)
    spp_stat_thr = df_rig['spp_stat_thr'] #SPP static threshold (psi)
    wob_thr = df_rig['wob_thr'] #WOB threshold (klbs)
    rpm_rot_thr = df_rig['rpm_rot_thr'] #RPM thresholds for rotating vs sliding activity
    rpm_stat_thr = df_rig['rpm_stat_thr'] #RPM static threshold

    #time variable to control static activity, trip in/out of hole, and circulating activity
    n_tseq_static = int(df_rig['n_tseq_static'])
    n_tseq_trip = int(df_rig['n_tseq_trip'])
    n_tseq_circ = int(df_rig['n_tseq_circ'])

    if real_time:
        #requires optimal number of observations to identify a rig activity for latest observation
        n_keep = max([n_tseq_static, n_tseq_trip, n_tseq_circ])

        #select only needed portion 
        df = df_input.copy().iloc[-n_keep:]
        smooth = False

    else:
        #copy dataframe
        df = df_input.copy()
        smooth = True
    
    #############################
    ## define and add columns ###
    #############################

    #Sub States: add Block Height position derivative
    cond = 'cond'
    BHch = BH +'_change'
    HDBDch = 'HD-BD_change'
    
    DUR = 'duration' #s
    LABEL = 'consecutive_labels'#temp  

    #Auxiliary columns
    RSUBACTIVITY = 'rig_sub_activity_raw'
    RSUB = 'rig_sub_state'
    RSUBCIR = 'rig_sub_circulate'
    RSUBCIRST = 'rig_sub_circulate_static'
    
    #Auxiliary mapping
    sub_state = {0:'OTHER', 1:'ROTATE', 2:'SLIDE',
                 3:'CNX (drill)', 4:'CIR (static)', 5:'STATIC', 
                 6: 'TRIP OUT', 7: 'TRIP IN', 8: 'NULL', 
                 9: 'CNX (trip)', 10: 'PUMP ON', 11: 'PUMP OFF'}
    sub_state_swap = {value:key for key, value in sub_state.items()}


    sub_cir_state = {0:'OTHER', 1:'WASH IN', 2:'WASH OUT', 
                     3:'REAM DOWN', 4:'REAM UP', 
                     5:'NO CIR', 6:'CIR', 7: 'NA'}
    sub_cir_state_swap = {value:key for key, value in sub_cir_state.items()}

    sub_cir_static_state = {0:'NA', 1:'CIR (static)'}
    sub_cir_static_state_swap = {value:key for key, value in sub_cir_static_state.items()}


    #additional variables
    df[BHch] = np.sign(df[BH].diff(1).fillna(0))
    df[HDBDch] = np.sign((df[HD]-df[BD]).diff(1).fillna(0))

    #ROTATE AND SLIDE condition
    hd_cond_less = (df[HD].diff(1).fillna(0) > 0.0001) if hd_diff_apply else ((df[HD]-df[BD]) <= depth_onb_thr)
    hd_cond_more = (df[HD].diff(1).fillna(0) <= 0.0001) if hd_diff_apply else ((df[HD]-df[BD]) > depth_onb_thr)

    #WOB condition
    wob_cond_less = (df[WOB] <= wob_thr) if wob_thr_apply else np.ones(df.shape[0], dtype = bool)
    wob_cond_more = (df[WOB] >= wob_thr) if wob_thr_apply else np.ones(df.shape[0], dtype = bool)

    #add new auxiliary columns
    for col in [RSUB, RSUBCIR, RSUBCIRST, RSUBACTIVITY]: 
        df[col] = np.zeros(df.shape[0])

    #add super and sub activity columns
    for col in [RSUPER, RSUBACT]:
        df[col] = np.nan
        
    #problems with surface RPM: recompute it from bit and motor RPM
    #df[RPM] = df[BRPM] - df[MRPM]
    
    #change bit depth for data during casing process (=> will labeled as out of hole)
    df.loc[df[CASING] == 1, RSUBACT] = 0
    
    ##################
    ### Sub States ###
    ##################
    
  
     #Sub States: Connection Drilling < 500 ft
    df.loc[((df[HD]-df[BD]) <= depth_conn_thr)                                            
           &(df[BD] > depth_conn_start)                                                   
           &(df[BD] < bd_conn)                                                             
           &wob_cond_less #*** add Ecopetrol March 22, 2021 *** June 28 edit ***           
           &(df[SPP] <= spp_thr) #*** add Ecopetrol March 22, 2021 ***                     
           &(df[HL] <= hl_conn_drill_thr), RSUB] = sub_state_swap['CNX (drill)']           
    
       #Sub States: Connection Drilling >500 ft
    df.loc[((df[HD]-df[BD]) <= depth_conn_thr)                                             
           &(df[BD] > depth_conn_start)                                                    
           &(df[BD] > bd_conn)                                                             
           &wob_cond_less #*** add Ecopetrol March 22, 2021 *** June 28 edit ***           
           &(df[HL] <= hl_conn_drill1_thr), RSUB] = sub_state_swap['CNX (drill)']         

    #Sub States: Connection Tripping < 500 ft
    df.loc[((df[HD]-df[BD]) > depth_conn_thr)                                              
           &(df[BD] > depth_conn_start)                                                     
           &(df[BD] < bd_conn)                                                              
           &(df[HL] <= hl_conn_trip_thr), RSUB] = sub_state_swap['CNX (trip)']              
    
    #Sub States: Connection Tripping > 500ft
    df.loc[((df[HD]-df[BD]) > depth_conn_thr)                                              
           &(df[BD] > depth_conn_start)                                                     
           &(df[BD] > bd_conn)                                                              
           &(df[HL] <= hl_conn_trip1_thr), RSUB] = sub_state_swap['CNX (trip)']             
    
    #assign Rotary drilling or Sliding using logics 
    #Sub States: Rotary drilling
    df.loc[hd_cond_less                                                                     
           &(df[GPM] > gpm_thr)                                                             
           &wob_cond_more                                                                   
           &(df[RPM] > rpm_rot_thr), RSUB] = sub_state_swap['ROTATE']                       

    #Sub States: Sliding
    df.loc[hd_cond_less                                                                     
           &(df[GPM] > gpm_thr)                                                             
           &wob_cond_more                                                                   
           &(df[RPM] <= rpm_rot_thr), RSUB] = sub_state_swap['SLIDE']                       
    
    #Sub States: Trip Out Free
    df[cond] = df[HDBDch].rolling(window = n_tseq_trip).sum().fillna(0)                    
    df.loc[(df[cond] >= (n_tseq_trip-1))                                                    
           &((df[HD]-df[BD]) > depth_trip_thr)                                              
           &(df[HL] > hl_conn_trip_thr)                                                     
           &(df[RPM] <= rpm_thr), RSUB] = sub_state_swap['TRIP OUT']                        
    
    #Sub States: Trip In Free
    df[cond] = df[HDBDch].rolling(window = n_tseq_trip).sum().fillna(0)                     
    df.loc[(df[cond] <= -(n_tseq_trip-1))
           &((df[HD]-df[BD]) > depth_trip_thr)
           &(df[HL] > hl_conn_trip_thr)
           &(df[RPM] <= rpm_thr), RSUB] = sub_state_swap['TRIP IN'] 

    #Sub States: Static: slow movement no more 3s Drill                                      #REPLACE
    df[cond] = df[BHch].rolling(window = n_tseq_static).sum().fillna(0)
    df.loc[(df[cond] == (n_tseq_static-1))
           &((df[HD]-df[BD]) <= depth_super_thr)
           &(df[GPM] <= gpm_thr)
           &(df[HL] > hl_conn_drill1_thr)                                            # &(df[HL] > hl_conn_drill1_thr)
           &(df[RPM] <= rpm_stat_thr)
           &(df[SPP] >= spp_stat_thr), RSUB] = sub_state_swap['STATIC']
    
    #Sub States: Static: slow movement no more 3s Trip                                         #REPLACE
    df[cond] = df[BHch].rolling(window = n_tseq_static).sum().fillna(0)
    df.loc[(df[cond] == (n_tseq_static-1))
           &((df[HD]-df[BD]) > depth_super_thr) 
           &(df[GPM] <= gpm_thr)
           &(df[HL] > hl_conn_trip_thr) #*** replace hl_conn_trip_thr with hl_conn_drill1_thr Ecopetrol March 22, 2021 ***                                                                                 #&(df[HL] > hl_conn_trip_thr)
           &(df[RPM] <= rpm_stat_thr)
           &(df[SPP] >= spp_stat_thr), RSUB] = sub_state_swap['STATIC']


    #Sub States: Circulating                                                    #REPLACE
    df.loc[(df[SPP] > spp_thr)
           &(df[GPM] > gpm_thr)                          
           &hd_cond_more, RSUB] = sub_state_swap['CIR (static)']
           # &(df[HL] > hl_conn_drill_thr) this condition is not necessary#&(df[HL] > hl_conn_drill_thr)   
    
    #Sub States: Switch Pumps On
    df.loc[(df[SPP] < spp_thr)                                                   
           &(df[GPM] > gpm_thr)
           &(df[HL] >= hl_conn_drill_thr)
           &((df[HD]-df[BD]) < depth_super_thr), RSUB] = sub_state_swap['PUMP ON']

    #Sub States: Switch Pumps Off                                                  #REPLACE
    df.loc[(df[SPP] > 0)      
           &(df[GPM] > gpm_thr)
           &(df[GPM] <= 10)                                        
           &(df[HL] >= hl_conn_drill_thr)
           &((df[HD]-df[BD]) < depth_super_thr), RSUB] = sub_state_swap['PUMP OFF']
           #&(df[GPM] <= gpm_thr)#&(df[GPM] > gpm_thr) and add  # &(df[GPM] <= 10)

    #Sub States: Null Values
    df.loc[(df[GPM] < gpm_thr)
           &(df[HL] < hl_null_thr), RSUB] = sub_state_swap['NULL']

    d = df[RSUB].value_counts(normalize=True).round(4)
    d.index = d.index.map(sub_state)

    #display data distribution
    if verbose:
        print('\nSub States data distribution:\n', d)

    ############################
    ### Sub Circulate States ###
    ############################

    #Sub States: Washing In
    df[cond] = df[BHch].rolling(window = n_tseq_circ).sum().fillna(0)                
    df.loc[(df[cond] == -n_tseq_circ)
           &(df[SPP] >= spp_thr)
           &(df[GPM] > gpm_thr)
           &(df[RPM] <= rpm_thr)
           &(df[HL] > hl_conn_drill_thr)
           &hd_cond_more, RSUBCIR] = sub_cir_state_swap['WASH IN']

    #Sub States: Washing Out                                                          
    df[cond] = df[BHch].rolling(window = n_tseq_circ).sum().fillna(0)
    df.loc[(df[cond] == n_tseq_circ)
           &(df[SPP] >= spp_thr)
           &(df[GPM] > gpm_thr)
           &(df[RPM] <= rpm_thr)
           &(df[HL] > hl_conn_drill_thr)
           &hd_cond_more, RSUBCIR] = sub_cir_state_swap['WASH OUT']

    #Sub States: Reaming                                                      
    df[cond] = df[BHch].rolling(window = n_tseq_circ).sum().fillna(0)
    df.loc[(df[cond] == -n_tseq_circ)
           &(df[SPP] >= spp_thr)
           &(df[GPM] > gpm_thr)
           &(df[RPM] > rpm_thr)
           &(df[HL] > hl_conn_drill_thr)
           &hd_cond_more, RSUBCIR] = sub_cir_state_swap['REAM DOWN']

    #Sub States: Back Reaming
    df[cond] = df[BHch].rolling(window = n_tseq_circ).sum().fillna(0)
    df.loc[(df[cond] == n_tseq_circ)
           &(df[SPP] >= spp_thr)
           &(df[GPM] > gpm_thr)
           &(df[RPM] > rpm_thr)
           &(df[HL] > hl_conn_drill_thr)
           &hd_cond_more, RSUBCIR] = sub_cir_state_swap['REAM UP']

    #Sub States: No Circulate
    df.loc[(df[GPM] <= gpm_thr)
          |(df[SPP] <= spp_thr), RSUBCIR] = sub_cir_state_swap['NO CIR']
    
    #Sub States: NA
    df.loc[df[RSUB] == sub_state_swap['ROTATE'], RSUBCIR] = sub_cir_state_swap['NA']
    df.loc[df[RSUB] == sub_state_swap['SLIDE'], RSUBCIR] = sub_cir_state_swap['NA']
    
    d = df[RSUBCIR].value_counts(normalize=True)
    d.index = d.index.map(sub_cir_state)

    #display data distribution
    if verbose:
        print('\nSub Circulate States data distribution:\n', d)

    ###################################
    ### Sub Circulate Static States ###
    ###################################

    #Sub Circulate Static 
    df.loc[(df[RSUB] == sub_state_swap['CIR (static)'])
           & (df[RSUBCIR] == sub_state_swap['OTHER']), RSUBCIRST] = sub_cir_static_state_swap['CIR (static)']

    d = df[RSUBCIRST].value_counts(normalize=True)
    d.index = d.index.map(sub_cir_static_state) 

    #display data distribution
    if verbose:
        print('\nSub Circulate Static States data distribution:\n', d)

    #assign Rotary drilling or Sliding using logics 
    #Sub States: Rotary drilling
    df.loc[hd_cond_less
           &(df[GPM] > gpm_thr)
           &wob_cond_more
           &(df[RPM] > rpm_rot_thr), RSUB] = sub_state_swap['ROTATE']

    #Sub States: Sliding
    df.loc[hd_cond_less
           &(df[GPM] > gpm_thr)
           &wob_cond_more
           &(df[RPM] <= rpm_rot_thr), RSUB] = sub_state_swap['SLIDE']
    
    # # ***New on 7/2/2021*** #
    # # only for Alchemist data
    # if operator.lower() == 'alchemist_energy':
    #    depth_start_10000 = 10000 #Depth for new slides from >10000ft 
    #    rpm_rot_thr_10000 = 40 #RPM thresholds for rotating vs sliding activity for >10000 ft

    #    #assign Rotary drilling or Sliding using logics from >10000 ft   
    #    #Sub States: Rotary drilling >10000 ft
    #    df.loc[hd_cond_less
    #           & (df[HD]>= depth_start_10000)
    #           &(df[GPM] > gpm_thr)
    #           &wob_cond_more
    #           &(df[RPM] > rpm_rot_thr_10000), RSUB] = sub_state_swap['ROTATE']

    #    #Sub States: Sliding >10000 ft
    #    df.loc[hd_cond_less
    #           & (df[HD] >= depth_start_10000)
    #           &(df[GPM] > gpm_thr)
    #           &wob_cond_more
    #           &(df[RPM] <= rpm_rot_thr_10000), RSUB] = sub_state_swap['SLIDE']

    #################################
    ### Assign final Sub Activity ###
    #################################

    df.loc[(df[RSUB] == sub_state_swap['ROTATE']), RSUBACTIVITY] = sub_activity_state_swap['ROTATE']
    df.loc[(df[RSUB] == sub_state_swap['SLIDE']), RSUBACTIVITY] = sub_activity_state_swap['SLIDE']
    df.loc[(df[RSUB] == sub_state_swap['PUMP OFF']), RSUBACTIVITY] = sub_activity_state_swap['PUMP ON/OFF']
    df.loc[(df[RSUB] == sub_state_swap['PUMP ON']), RSUBACTIVITY] = sub_activity_state_swap['PUMP ON/OFF']
    df.loc[(df[RSUB] == sub_state_swap['CNX (trip)']), RSUBACTIVITY] = sub_activity_state_swap['CNX (trip)']
    df.loc[(df[RSUB] == sub_state_swap['CNX (drill)']), RSUBACTIVITY] = sub_activity_state_swap['CNX (drill)']
    df.loc[(df[RSUB] == sub_state_swap['STATIC']), RSUBACTIVITY] = sub_activity_state_swap['STATIC']
    df.loc[(df[RSUB] == sub_state_swap['NULL']), RSUBACTIVITY] = sub_activity_state_swap['NULL']
    df.loc[(df[RSUB] == sub_state_swap['TRIP IN']), RSUBACTIVITY] = sub_activity_state_swap['TRIP IN']
    df.loc[(df[RSUB] == sub_state_swap['TRIP OUT']), RSUBACTIVITY] = sub_activity_state_swap['TRIP OUT']

    df.loc[(df[RSUBCIR] == sub_cir_state_swap['REAM UP']), RSUBACTIVITY] = sub_activity_state_swap['REAM UP']
    df.loc[(df[RSUBCIR] == sub_cir_state_swap['REAM DOWN']), RSUBACTIVITY] = sub_activity_state_swap['REAM DOWN']
    df.loc[(df[RSUBCIR] == sub_cir_state_swap['WASH IN']), RSUBACTIVITY] = sub_activity_state_swap['WASH IN']
    df.loc[(df[RSUBCIR] == sub_cir_state_swap['WASH OUT']), RSUBACTIVITY] = sub_activity_state_swap['WASH OUT']
    df.loc[(df[RSUBCIRST] == sub_cir_static_state_swap['CIR (static)']), RSUBACTIVITY] = sub_activity_state_swap['CIR (static)']

    d = df[RSUBACTIVITY].value_counts(normalize=True)
    d.index = d.index.map(sub_activity_state) 

    #display data distribution
    if verbose:
        print('\nSub States final data distribution:\n', d)

    ####################
    ### Super States ###
    ####################
    
    #Super States: Drilling
    df.loc[(df[HD]-df[BD]) <= depth_super_thr, RSUPER] = super_state_swap['DRILL']
    #Super States: Tripping
    df.loc[(df[HD]-df[BD]) > depth_super_thr, RSUPER] = super_state_swap['TRIP']
    #Super States: Out of Hole
    df.loc[(df[BD] < depth_ooh_thr), RSUPER] = super_state_swap['OUT OF HOLE']

    #fix mislabeled Sub states for Out of Hole super state
    df.loc[df[RSUPER]==super_state_swap['OUT OF HOLE'], RSUB] = sub_state_swap['OTHER']

    for col in ['TRIP IN','TRIP OUT']:
        df.loc[(df[RSUPER] == super_state_swap['DRILL'])&
               (df[RSUB] == sub_state_swap[col]), RSUB] = sub_state_swap['OTHER']

    for col in ['TRIP IN','TRIP OUT']:
        df.loc[(df[RSUPER] == super_state_swap['DRILL'])
              &(df[RSUB] == sub_state_swap['OTHER'])
              &(df[RSUBACTIVITY] == sub_activity_state_swap[col]), RSUBACTIVITY] = sub_activity_state_swap['OTHER']
    
    #connection trip while drilling 'Connection Drilling': 'CNX (drill)','Connection Tripping': 'CNX (trip)'
    df.loc[(df[RSUPER]==super_state_swap['DRILL'])
           &(df[RSUB] == sub_state_swap['CNX (trip)']), RSUB] = sub_state_swap['NULL']

    #connection drill while tripping
    df.loc[(df[RSUPER]==super_state_swap['TRIP'])
           &(df[RSUB] == sub_state_swap['CNX (drill)']),RSUB] = sub_state_swap['NULL']
    
    #####################################
    ### out of hole to sub activities ###
    #####################################
    df.loc[(df[RSUPER] == super_state_swap['OUT OF HOLE']), RSUBACTIVITY] = sub_activity_state_swap['OUT OF HOLE']
    
    d = df[RSUPER].value_counts(normalize=True)
    d.index = d.index.map(super_state)
    
    #change column type
    #df[[RSUBACTIVITY,RSUPER]] = df[[RSUBACTIVITY,RSUPER]].astype(int)

    #display data distribution
    if verbose:
        print('\nSuper States final data distribution:\n', d)
    
    #####################
    ### add smoothing ###
    #####################
  
    df = comp_duration(df, activity_col = RSUBACTIVITY, dtime = dtime)
    
    #define different smooting windows in seconds; do not smooth 'NULL'
    #*** update October 25, 2021 ***
    if use_filter_dict:
        filter_dict = ast.literal_eval(df_rig[f'filter_dict'])
        if verbose:
            print('\n*** Using default filter_dict:', filter_dict)
            print(f'Please double check that filter_dict corresponds to the current RT data resolution of {dtime} seconds.***\n')
    else:
        filter_dict = ast.literal_eval(df_rig[f'filter_dict_{int(dtime)}'])
    
    #convert smooth time step according to to sampling time step (dtime)
    for key, value in filter_dict.items():
        #print(value, dtime)
        filter_dict[key] = np.clip(int(np.round(value/dtime)), a_min=1, a_max = None)
            
    #create subactivity column 
    df[RSUBACT] = df[RSUBACTIVITY]
    
    ####################################################
    ### fix sliding data using info about mud motor ####
    ####################################################
    
    #replace slide with rotary sliding if no mud motor is used
    df.loc[(df[RSUBACT] == sub_state_swap['SLIDE'])
           &(df[MM] == 0), RSUBACT] = sub_state_swap['ROTATE']
           #*** replace 'ROTATE' with 'CIR (static)' Ecopetrol March 22, 2021 ***
    
    ##################################
    ### smooth subactivity column ####
    ##################################
    if smooth:

        for key, value in filter_dict.items():
            df.loc[(df[DUR] < pd.Timedelta(seconds=value))&
                   (df[RSUBACTIVITY] == sub_activity_state_swap[key]), RSUBACT] = np.nan

        #fill forwars and change variable type
        df[RSUBACT].fillna(method = 'ffill',inplace=True) 
    
    ##recompute labels and duration for smooth labels
    df = comp_duration(df, activity_col = RSUBACT, dtime = dtime)

    #change type
    # for col in [RSUPER, RSUBACT, LABEL]:
    #     df[col] = df[col].astype(int)
        
    ###############################
    ### drop auxiliary columns ####
    ###############################
    
    if show_activity_names:
        ### this condition is 'flag' to add 3 following columns:
        ### [RSUBACTIVITY + '_name'] == 'rig_sub_activity_raw_name';
        ### [RSUBACT + '_name'] == rig_sub_activity_name'
        ### [RSUPER + '_name'] =='rig_super_state_name'
        
        df[RSUPER + '_name'] = df[RSUPER].map(super_state)#.map(super_activity_dashboard)

        #keep raw subactivities to check smoothing
        df[RSUBACTIVITY + '_name'] = df[RSUBACTIVITY].map(sub_activity_state)#.map(sub_activity_dashboard)

        ## add sub state naming column
        df[RSUBACT + '_name'] = df[RSUBACT].map(sub_activity_state)#.map(sub_activity_dashboard)

        ## drop columns: ['rig_super_state','rig_sub_activity']. if dropped some other function will not work
        # df.drop(columns = [RSUPER,RSUBACT],inplace = True)

    
    
    df.drop(columns=[cond, BHch, HDBDch,
                     RSUBACTIVITY, RSUB, RSUBCIR, RSUBCIRST],inplace=True) 
    
    
    
    if real_time:
        output = df.iloc[-1]
        
        print('\n*** Last observation rig activities: ***\n')
        print('RIG SUPER STATE:', super_state[output.loc[RSUPER]])
        print('RIG SUBSTATE:', sub_activity_state[output.loc[RSUBACT]])
        
        print('\n*** Last observation full data output: ***')
        display(output)
    else:
        output = df
        
    return output

#function to construct the consecutive connection activity labels 
def conn_activity_label(df_input: pd.DataFrame, 
                        activity: str, LABEL: str,
                        activity_col = RSUBACT, 
                        activity_dict = sub_activity_state_swap) -> pd.DataFrame:
    """This function computes continuous connection activity label a specific activity.
    Here df_input is time-based drilling data, 
    activity is an activity name,
    LABEL is computed continuous activity label column name,
    activity_col is the activity column name (e.g. RSUPER or RSUBACT), 
    activity_dict is a dictionary that define mapping from names to number codes.
    Function returns dataframe has 1 extra column: LABEL."""
    
    df = df_input.copy()

    subactivity=df[activity_col].values
    consecutive_label=np.zeros(subactivity.shape[0])

    count=0
    for i in range(1, subactivity.shape[0]):
        if subactivity[i] == activity_dict[activity]:
            if subactivity[i]!=subactivity[i-1]:
                count+=1
                consecutive_label[i] = count
            else:
                consecutive_label[i] = count
    
    #Add consequtive label
    df[LABEL]=consecutive_label    
    df[LABEL] = df[LABEL].astype(int)
    
    return df

#function to construct the consecutive activity labels between connection (trip)
def conn_activity_label_between(df_input: pd.DataFrame, 
                                activity: str, LABEL: str,
                                activity_col = RSUBACT, activity_dict = sub_activity_state_swap) -> pd.DataFrame:
    """This function computes continuous connection activity label between a specific activity.
    Here df_input is time-based drilling data, 
    activity is an activity name,
    LABEL is computed continuous activity label column name,
    activity_col is the activity column name (e.g. RSUPER or RSUBACT), 
    activity_dict is a dictionary that define mapping from names to number codes.
    Function returns dataframe has 1 extra column: LABEL."""
    
    df = df_input.copy()
    
    subactivity=df[activity_col].values
    consecutive_label=np.zeros(subactivity.shape[0])

    count=0
    
    for i in range(1,subactivity.shape[0]):
        if subactivity[i] != activity_dict[activity] :
            consecutive_label[i] = count
        elif (subactivity[i-1] != subactivity[i]):
            count += 1
            consecutive_label[i] = -1
        else:
            consecutive_label[i] = -1
    
    #Add consequtive label for Connection Tripping activity
    df[LABEL]=consecutive_label    
    df[LABEL] = df[LABEL].astype(int)
    
    return df

#mapping function to identify trip in/trip out activities
def get_trip_in_trip_out(s: pd.Series, 
                         activity_dict: dict) -> int:
    """This is a function to identify prevailing activity between 
    connections: trip in, trip out or nan.
    Here s is a Series,
    activity_dict is a dictionary that define mapping from names to number codes.
    Function returns trip in, trip out or nan."""
    
    try:
        output = s.loc[s.isin([sub_activity_state_swap['TRIP IN'],
                               sub_activity_state_swap['TRIP OUT']])].mode().values[0]
    except:
        output = np.nan
    
    return output

#function to compute tripping table that includes pipe speed
def compute_df_trip(df_input: pd.DataFrame, add_stand = False,
                    trip_speed_max = 5000,
                    WELLID = WELLID, BS = BS, PIPESP = PIPESP,
                    LABELct = LABELct, LABELbtwn = LABELbtwn,
                    BD = BD, dBD = dBD, TIME = TIME, dTIME = dTIME, 
                    RSUBACT = RSUBACT, STAND = STAND,
                    activity_dict = sub_activity_state_swap) -> pd.DataFrame:
    """This function computes trip dataframe.
    Here df_input is time-based drilling data, 
    add_stand is a flag to include STAND column (when True), do not include otherwise,
    WELLID, BS, PIPESP, LABELct, LABELbtwn, 
    BD, dBD, TIME, dTIME represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    activity_dict is a dictionary that define mapping from names to number codes.
    Function returns df_trip dataframe."""

    df = df_input.copy()

    group_cols = [WELLID, LABELct]
    group_cols_btwn = [WELLID, LABELbtwn]
    trip_cols = [WELLID, BD + '_first', BD + '_last', 
                 TIME + '_first', TIME + '_last', dBD, dTIME, PIPESP, TRIPSPEED, BS, RSUBACT]

    if add_stand:
        group_cols.append(STAND)
        group_cols_btwn.append(STAND)
        trip_cols.append(STAND)

    #group by connection activity
    df_trip = df.groupby(group_cols)[[BD, TIME]].agg(['first', 'last']).dropna()
    df_trip.columns = ['_'.join(col) for col in df_trip.columns]

    #calculate differences
    df_trip[dBD] = abs(df_trip[BD + '_first'].shift(-1) - df_trip[BD + '_last'])
    #delta time (minutes)
    df_trip[dTIME] = (df_trip[TIME + '_first'].shift(-1) - df_trip[TIME + '_last']).clip(lower = 0)
    #connection time (minutes)
    df_trip[dTIME + '_conn'] = (df_trip[TIME + '_last'] - df_trip[TIME + '_first']).clip(lower = 0)
    #calculate pipe speed (ft/h) and set minimum value to 0
    df_trip[PIPESP] = (df_trip[dBD]/df_trip[dTIME]).clip(lower=0) * 60
    #calculate trip speed (ft/h) and set minimum value to 0
    df_trip[TRIPSPEED] = (df_trip[dBD]/df_trip[dTIME + '_conn']).clip(lower=0) * 60
    #add section size
    df_trip[BS] = df.groupby(group_cols)[BS].apply(lambda x: max(x.unique())).dropna()
    #reindex
    df_trip = df_trip.reset_index()#.set_index(LABELct)
    #remove 0th activity
    df_trip = df_trip.loc[df_trip[LABELct] != 0]
    #drop missing and inf
    df_trip.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_trip.dropna(inplace=True)

    #compute auxiliary table
    df_btw = df.loc[df[LABELbtwn] != 0].groupby(group_cols_btwn)[RSUBACT]\
            .apply(get_trip_in_trip_out, activity_dict = sub_activity_state_swap).to_frame().dropna().reset_index()

    #*** fix on Aug 12: account for missing cnx (trip) ***#
    #merge
    if (not(df_btw.empty))&(not(df_trip.empty)):
        df_trip = df_trip.merge(df_btw, how='left', 
                                left_on = group_cols, right_on = group_cols_btwn)\
                                .drop(columns = [LABELbtwn]).set_index(LABELct)
        #rearrange columns
        df_trip = df_trip[trip_cols]
    else:
        df_trip = pd.DataFrame([[0]*len(trip_cols)], columns = trip_cols).rename_axis(LABELct)

    #*** fix on Dec 3 2021: filter low pipe and trip speed value ***#
    #compute percentiles:
    q_min, q_max = df_trip.loc[df_trip[PIPESP] > 0.001, PIPESP].quantile(q = [0.05, 0.999])
    print('\ndf_trip size before filtering low pipe speed:', df_trip.shape[0])
    df_trip = df_trip.loc[(df_trip[PIPESP] > q_min)&(df_trip[PIPESP] < q_max),:]
    print('\ndf_trip size after filtering low pipe speed:', df_trip.shape[0])

    print('\ndf_trip size before filtering low trip speed:', df_trip.shape[0])
    df_trip = df_trip.loc[(df_trip[TRIPSPEED] < trip_speed_max),:]
    print('\ndf_trip size after filtering low trip speed:', df_trip.shape[0])

    return df_trip

#function to compute well construction rate plan
def compute_df_wcr_plan(df_plan: pd.DataFrame, 
                        save_folder = save_folder, save = True) -> pd.DataFrame:
    """This function computes well construction rate plan.
    Here df_plan is a time-depth plan for a current well,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data,
    Function returns df_wcr_plan - well construction plan dataframe. """

    time_conversion = 1/(60*24) #convert to days

    wcr_total = df_plan[HD].max()/(df_plan[TIME].max() * time_conversion)
    wcr_total

    df_gr = df_plan.groupby(BS)[[HD,TIME]].agg(['min','max'])

    df_wcr_plan = pd.DataFrame()

    df_wcr_plan[DD] = (df_gr.loc[:,(HD,'max')]-df_gr.loc[:,(HD,'min')])
    df_wcr_plan[TIMEDAY] = (df_gr.loc[:,(TIME,'max')]-df_gr.loc[:,(TIME,'min')]) * time_conversion

    #add value for all sections
    t = pd.DataFrame(df_wcr_plan.sum(axis=0)).T
    t.rename(index={0: 'all'}, inplace=True)

    df_wcr_plan = df_wcr_plan.append(t)

    df_wcr_plan[WCRC] = df_wcr_plan[DD]/df_wcr_plan[TIMEDAY]

    df_wcr_plan.reset_index(inplace=True)
    df_wcr_plan.rename(columns={'index': BS},inplace=True)

    #save computed wcr plan
    if save:
        df_wcr_plan.round(round_ndigits).to_csv(f'{save_folder}csv/wcr_plan.csv', index=False)

    return df_wcr_plan

#function to well construction rate
def compute_WCR(df_input: pd.DataFrame, df_wcr_plan: pd.DataFrame, 
                hole_diameter: float, 
                dtime_dict: dict, well_name_dict: dict,
                WELL = WELLID, HD = HD, TIME = TIME,
                DD = DD, TIMEDAY = TIMEDAY, WCRC = WCRC, WELLIDN = WELLIDN,
                wcr_unit = 'ft/day',
                replace_dot = replace_dot, round_ndigits = round_ndigits,
                save_folder = save_folder, save = True):

    """This function computes and plots well construction rate data.
    Here df_input is time-based drilling data, 
    df_wcr_plan is well construction rate plan for a current well,
    dtime_dict is a dictionary that define mapping from well ids to time step/data density, 
    well_name_dict is a dictionary that define mapping from well ids.
    WELL, HD, TIME represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,
    replace_dot is a symbol to replace dot in table name containing bit size 
    (e.g. if replace_dot = 'p' then '12.5' -> '12p5')
    round_ndigits is number of digits to round calculations,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function returns WCR_summary_well, WCR_summary dataframes containing 
    per well and overall well constructino rates calculations."""
    
    df = df_input.copy().dropna(subset=[HD, TIME, BS])

    #select specific hole size section
    if hole_diameter != 'all':
        df = df.loc[df[BS] == hole_diameter]

    bs = str(hole_diameter).replace('.', replace_dot)
    s = 'SECTION ' + decimal_fraction_str(hole_diameter) if (hole_diameter != 'all') else ''

    #calculate total length drilled
    total_length = df.groupby(WELL)[HD].apply(lambda x: x.max()-x.min())
    
    #account for variable sampling time step in seconds and then convert to days #.loc[df[RSUPER] == 1]
    total_time_days = ((df.groupby(WELL)[TIME].nunique() * pd.Series(dtime_dict)).dropna())\
                           .dt.total_seconds()/3600/24
    # #account for missing time intervals
    # missing = pd.Series()
    # for well in df[WELL].unique():

    #     #print('Sampling time step distribution for ', well)
    #     t_step_dist = (df.loc[df[WELL] == well, TIME].diff()*60).round()\
    #                      .dropna().astype(int).value_counts(normalize=True)
    #     #convert to days
    #     missing[well_name_dict[well]] = (t_step_dist.index[1:].values.sum() 
    #                      - t_step_dist.index[1:].shape[0] * t_step_dist.index[0])/3600/24
                
    # missing.index = missing.index.map(well_name_dict_swap)
                
    #add missing part
    #total_time_days += missing

    #calculate well construction rate (WCR) in ft/day
    WCR = total_length/total_time_days
    WCR_summary_well = pd.DataFrame()
    
    WCR_summary_well[DD] = total_length #'DRILL DEPTH (ft)'
    WCR_summary_well[TIMEDAY] = total_time_days #'TOTAL TIME (days)'
    WCR_summary_well[WCRC] = WCR #WCR (ft/day)
    WCR_summary_well[WELLIDN] = WCR_summary_well.index.astype(int).astype(str) + ' - ' + WCR_summary_well.index.map(well_name_dict)
    #reset index to well_id - well_name
    #WCR_summary_well = WCR_summary_well.reset_index().set_index(WELLIDN)

    ##select all available wells
    WCR_summary = WCR_summary_well.iloc[:,:-1].apply(['min','max','mean']).T\
                                  .rename(columns = {'mean':'avg'})
    
    ##add plan
    WCR_summary[PLAN] = (df_wcr_plan.loc[df_wcr_plan[BS].astype(str) == str(hole_diameter),:].drop(columns=[BS]).T)

    #####################################
    #### Well construction rate plot ####
    #####################################

    #change WCR index for plotting
    WCR.index = WCR.index.astype(int).astype(str) + ' - ' + WCR.index.map(well_name_dict)

    #WCR barplot
    _, ax = plt.subplots(1,1,figsize=(9,5))
    
    #.sort_values(ascending=True)
    WCR.plot(kind='barh',ax=ax, color=color_purple, label='time/length')
    #min
    ax.axvline(WCR.min(),linestyle = '--', color = color_underperf, label='min',lw=2)
    #average: mean
    ax.axvline(WCR.mean(),linestyle = '--', color = color_historic, label='avg',lw=2)
    #plan
    ax.axvline(WCR_summary.loc[WCRC, PLAN],linestyle = '--', color = color_neutral, label='plan', lw = 2)
    #max
    ax.axvline(WCR.max(),linestyle = '--', color = color_overperf, label='max',lw=2)

    #label plot
    ax.legend(['MIN','AVG','PLAN','MAX'], bbox_to_anchor=(-0.1,1.3), 
               loc='upper left', fancybox=True, shadow=False, ncol=4, fontsize=14)

    #fill area
    xmax, ymax, ymin = [WCR.max()*1.3, max(ax.get_ylim()), -0.5]
    ax.fill_between(x=[0,WCR.min()], y1=[ymin, ymin], y2=[ymax,ymax], color=color_underperf, alpha=0.2)
    ax.fill_between(x=[WCR.max(), xmax], y1=[ymin, ymin], y2=[ymax,ymax], color=color_overperf, alpha=0.2)

    #label
    ax.set_xlabel(f'WCR ({wcr_unit})')
    ax.set_ylabel('')
    try:
        ax.set_xlim(0, xmax)
    except:
        print('No limits available for RT')
    ax.set_title(s)
    #plt.title('Well Construction Rate')
    plt.tight_layout()
    
    if save:

        plt.savefig(f'{save_folder}plot/wcr_summary_bs_{bs}.png',dpi=150)
        WCR_summary_well.round(round_ndigits).to_csv(f'{save_folder}csv/wcr_summary_well_bs_{bs}.csv', index=True)
        WCR_summary.round(round_ndigits).to_csv(f'{save_folder}csv/wcr_summary_bs_{bs}.csv',index=True, index_label='units')

    plt.close()

    return WCR_summary_well, WCR_summary

#function to plot welldifficulty per well on reach or depth analysis background
def plot_well_difficulty(df_survey: pd.DataFrame, df_survey_rt: pd.DataFrame,
                        df_analysis: pd.DataFrame, df_historic_extreme: pd.DataFrame,
                        df_direct_plan: pd.DataFrame,
                        analysis_type: str, fit = '_fit',
                        dd_range_color_dict = dd_range_color_dict,
                        TVD = TVD, UWDEP = UWDEP, DDI = DDI, DDIR = DDIR,
                        tvd_max = None, uwdep_max = None,
                        depth_unit = 'ft',
                        save_folder = save_folder, save = True):
    """This function plots well difficulty: depth and reach analysis.
    Here df_survey is official survey data,
    df_survey_rt is current well official survey data,
    df_analysis is a static analysis specific dataframe 
    df_historic_extreme is a static dataframe with a historic maximum ever drilled curve,
    (e.g. df_depth or df_reach loaded from input data folder)
    analysis_type is type of analysis (e.g. 'depth' or 'reach')
    TVD, UWDEP, DDI, DDIR represent standard column name, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,
    tvd_max is a maximum plotting limit for TVD,
    uwdep_max is a maximum plotting limit for UWDEP,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function only plots depth or reach analysis."""

    _, ax = plt.subplots(1,1,figsize=(12,5))

    #plot background
    if analysis_type == 'depth':
        #add colors
        df_analysis[COLOR] = well_diff_colors
        #append start and end
        uwdep_interval_max = [0,45000]
        df_analysis = df_analysis.append(pd.DataFrame({DEPTHTYPE: ['Start','End'], DEPTHTVDTO: uwdep_interval_max, 
                                                COLOR: [np.nan, np.nan]})).sort_values(DEPTHTVDTO)\
                                                .reset_index().drop(columns = 'index')
        #area plots/hatched areas                                   
        for i in range(1, df_analysis.shape[0] - 1):
            tvd_top = [df_analysis.iloc[i-1][DEPTHTVDTO]]*2
            tvd_bot = [df_analysis.iloc[i][DEPTHTVDTO]]*2

            ax.fill_between(x = uwdep_interval_max, y1 = tvd_top, y2 = tvd_bot, 
                            label = df_analysis.iloc[i][DEPTHTYPE], color = df_analysis.iloc[i][COLOR], 
                            alpha = 0.2, zorder = i)

    elif analysis_type == 'reach':

        #area plots/hatched areas
        [temp_x, temp_y] = df_analysis[['unwrap_departure_to','tvd_low']].dropna().values.T
        ax.fill_between(x=temp_x[:2], y1=temp_y[:2], y2=temp_y[-1:-3:-1], 
                        color=well_diff_colors[0],alpha=0.2,label='Low',zorder=3)

        [temp_x, temp_y] = df_analysis[['unwrap_departure_to','tvd_medium']].dropna().values.T
        ax.fill_between(x=temp_x[:2], y1=temp_y[:2], y2=temp_y[-1:-3:-1], 
                        color=well_diff_colors[1],alpha=0.2,label='Medium',zorder=2)

        [temp_x, temp_y] = df_analysis[['unwrap_departure_to','tvd_extended']].dropna().values.T
        ax.fill_between(x=temp_x[:2], y1=temp_y[:2], y2=temp_y[-1:-3:-1], 
                        color=well_diff_colors[2],alpha=0.2, label='Extended',zorder=1)

        [temp_x, temp_y] = df_analysis[['unwrap_departure_to','tvd_very_extended']].dropna().values.T
        ax.fill_between(x=[0,temp_x[0]], y1=[0, 0], y2=[temp_y[-1],temp_y[-1]], 
                        color=well_diff_colors[3],alpha=0.2, label='Very Extended',zorder=0)

    else:
        print('Select available analysis_type: depth or reach.')

    #add another axis to show second legend
    ax2 = ax.twinx()

    #plot historical extreme envelope
    ax2.plot(df_historic_extreme[UWDEP + '_to'], df_historic_extreme[TVD + '_to'],'--k')

    #find well with most variability in ddi
    well_ddi = df_survey.groupby(WELLID)[DDIR].nunique().idxmax()
    for i, well in enumerate(df_survey[WELLID].unique()):

        df_survey_ = df_survey.loc[df_survey[WELLID] == well]
        #palette_ddi = [dd_range_color_dict[col] for col in df_survey_[DDIR].unique()]
        l = 'brief' if i==0 else False 

        # g = sns.lineplot(x = UWDEP,y = TVD, hue = DDIR, 
        #                 data = df_survey_,
        #                 ax = ax2, palette = palette_ddi, legend = l, linewidth=2.0)
        
        ddi_ranges = [x for x in df_survey_[DDIR].unique() if str(x) != 'nan']
        for ddi_range in ddi_ranges:
            data = df_survey_[df_survey_[DDIR] == ddi_range]
            leg_label = ddi_range if well == well_ddi else None
            ax2.plot(data[UWDEP], data[TVD],'-', lw = 2.0, color = dd_range_color_dict[ddi_range], label = leg_label)

        #ax2.legend_.set_title(None)
        #plt.setp(g.lines, zorder=100)
        
    
    #add current well trajectory
    if df_survey_rt[DDI].isnull().any():
        #add current well trajectory with solid color (no ddi option)
        ax2.plot(df_survey_rt[UWDEP], df_survey_rt[TVD],'-k', lw = 4.0, zorder=200)
    else: 
        #add current well with ddi computed
        #palette_ddi = [dd_range_color_dict[col] for col in df_survey_rt[DDIR].unique()]
        # sns.lineplot(x = UWDEP,y = TVD, hue = DDIR, 
        #             data = df_survey_rt, ax=ax2, palette = palette_ddi, legend = None, linewidth = 4.0, 
        #             style = True, zorder=120)

        ddi_ranges = [x for x in df_survey_rt[DDIR].unique() if str(x) != 'nan']
        for ddi_range in ddi_ranges:
            data = df_survey_rt[df_survey_rt[DDIR] == ddi_range]
            ax2.plot(df_survey_rt[UWDEP], df_survey_rt[TVD],'-k', lw = 4.0, zorder=200)#, color = dd_range_color_dict[ddi_range])


    #add plan
    ax2.plot(df_direct_plan[UWDEP], df_direct_plan[TVD],'--k', lw = 2.0, zorder=150)

    ax2.get_yaxis().set_visible(False)
    #show two legends
    ax.legend(bbox_to_anchor = (0,1.42), loc='upper left', fancybox=True, shadow=False, ncol=5, fontsize=14)
    ax2.legend(bbox_to_anchor = (0,1.25), loc='upper left', fancybox=True, shadow=False, ncol=5, fontsize=14)

    #set limits
    if tvd_max == None:
        tvd_max = max([df_survey_rt[TVD].max(), df_survey[TVD].max()]) * 1.1
        uwdep_max = max([df_survey_rt[UWDEP].max(), df_survey[UWDEP].max()]) * 1.1

    ax.set(xlim = (0, uwdep_max), ylim = (0, tvd_max)) 
    ax2.set(xlim = (0, uwdep_max), ylim = (0, tvd_max)) 
    ax.invert_yaxis(), ax2.invert_yaxis()
    ax.set(ylabel = f'TVD ({depth_unit})', xlabel = f'UNWRAPPED DEPARTURE ({depth_unit})')
    plt.suptitle('RT (thick) ' + 'PLAN (--)',fontsize=14, y=0.8)

    plt.tight_layout()

    if save:
        plt.savefig(f'{save_folder}plot/ddi_{analysis_type}{fit}.png',dpi=150)

    plt.close()

    return None
    
#compute performance boundary table
def compute_perform_boundary(df_input: pd.DataFrame, hole_diameter: float, 
                             bound_cols: list, index_col: str, table_name: str,
                             bound_window = 20, BS = BS,
                             replace_dot = replace_dot, round_ndigits = round_ndigits, 
                             save_folder = save_folder, save = True) -> pd.DataFrame:
    """This function performance boundary table.
    Here df_input is time-based drilling data,
    hole_diameter is a section diameter size,
    bound_cols is a list of columns to compute boundaries for,
    index_col is a columns name for index axis (e.g. TIME or HD),
    table_name is part of a table name to save computed data,
    bound_window is a smooth window used in performance boundary calculation,
    BS represent standard column name, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    replace_dot is a symbol to replace dot in table name containing bit size 
    (e.g. if replace_dot = 'p' then '12.5' -> '12p5')
    round_ndigits is number of digits to round calculations,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function returns performance boundary vs index_col dataframe."""
    
    bs = str(hole_diameter).replace('.', replace_dot)

    #copy input data
    df = df_input.copy()

    if hole_diameter != 'all':
        df = df.loc[df[BS] == hole_diameter]

    df_bound = pd.DataFrame()
    
    for bound_col in bound_cols:

        #best boundary
        df_bound[bound_col + '_best'] = df.groupby(index_col)[bound_col].max().rolling(window = bound_window, center=True).max()
        #worst boundary
        df_bound[bound_col + '_worst'] = df.groupby(index_col)[bound_col].min().rolling(window = bound_window, center=True).min()
        #average
        df_bound[bound_col + '_avg'] = (df_bound[bound_col + '_best'] + df_bound[bound_col + '_worst'])/2

    #drop missing values
    df_bound.dropna(inplace=True)
    df_bound.reset_index(inplace=True)
    
    #save csv data
    if save:
        df_bound.round(round_ndigits).to_csv(f'{save_folder}csv/{table_name}_best_worst_avg_bs_{bs}.csv', index = False)

    return df_bound

#function to plot time vs depth
def plot_time_depth(df_input: pd.DataFrame, 
                    df_rt_input: pd.DataFrame, df_plan_input: pd.DataFrame, 
                    df_bound: pd.DataFrame, hole_diameter: float, depth_col: str, 
                    depth_unit = 'ft',
                    add_xlims_rt = True,
                    WELL = WELLID, TIME = TIME, HD = HD, BS = BS,
                    perform_colors = perform_colors, 
                    replace_dot = replace_dot,
                    save_folder = save_folder, save = True):
    """This function plots time versus depth graphs.
    Here df_input is time-based drilling data,
    df_rt_input is current well time-based drilling data,
    df_plan_input is a time-depth plan for a current well,
    df_bound is a performance boundary data,
    hole_diameter is a section diameter size,
    depth_col is a depth column name (e.g. HD, BD),
    WELL, TIME, HD, BS, represent standard column name, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,
    perform_colors is a list or colors for plotting,
    replace_dot is a symbol to replace dot in table name containing bit size 
    (e.g. if replace_dot = 'p' then '12.5' -> '12p5')
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function only plots and save the graph, but return none."""

    bs = str(hole_diameter).replace('.', replace_dot)

    #copy input data  
    df = df_input.copy()
    df_rt = df_rt_input.copy()
    df_plan = df_plan_input.copy()
      
    #select hole size section        
    if hole_diameter != 'all':
        df = df.loc[df[BS] == hole_diameter]
        df_rt = df_rt.loc[df_rt[BS] == hole_diameter]
        df_plan = df_plan[df_plan[BS] == hole_diameter]
    
    ############################
    #### Time vs Depth plot ####
    ############################
    _, ax = plt.subplots(1,1,figsize=(15.5,5))

    #shift df_bound to allign with current well
    df_bound[TIME] += (df_rt[TIME].min() - df_bound[TIME].min())

    #factor to converst seconds to days
    time_conversion = 1/(24*60)
    #apply time conversion for boundary data
    df_bound[TIME] *= time_conversion

    #plot historic wells
    for i, well in enumerate(df[WELL].unique()):
        df_ = df[df[WELL]==well]
        l = 'HISTORIC' if i==0 else ''
        ax.plot((df_[TIME] - df_[TIME].min() + df_rt[TIME].min())*time_conversion, 
                df_[depth_col], lw=1, label=l, color = perform_colors[3],alpha=0.5)

    #best boundary
    ax.fill_between(x = df_bound[TIME], y1 = 20000 * np.ones(df_bound.shape[0]),
                    y2 = df_bound[depth_col + '_best'], color = perform_colors[2], alpha=0.1, label=None)
    ax.plot(df_bound[TIME], df_bound[depth_col + '_best'], lw=2, color = perform_colors[2], label='BEST')
    
    #worst boundary
    ax.fill_between(x = df_bound[TIME], y1 = np.zeros(df_bound.shape[0]),
                    y2 = df_bound[depth_col + '_worst'], color = perform_colors[0], alpha=0.1, label=None)
    ax.plot(df_bound[TIME], df_bound[depth_col + '_worst'], lw=2, color = perform_colors[0], label='WORST')
    
    #add average
    ax.plot(df_bound[TIME], (df_bound[depth_col + '_best'] + df_bound[depth_col + '_worst'])/2, 
            lw=2, label='AVG', color = perform_colors[1], alpha=0.5)
    
    #add plan
    ax.plot(df_plan[TIME]*time_conversion, df_plan[HD], lw=3, color='k', label = 'PLAN', linestyle = '--')
     
    #current well
    ax.plot(df_rt[TIME]*time_conversion, df_rt[depth_col],
            lw=4,label = ('RT'), color=perform_colors[-1])

    if add_xlims_rt:
        ax.set_xlim(np.array([df_rt[TIME].min(), df_rt[TIME].max()]) * time_conversion)
    else:
        ax.set_xlim(np.array([min([df[TIME].min(),df_rt[TIME].min()]), 
                              max([df[TIME].max(),df_rt[TIME].max()])]) * time_conversion)

    ax.set_ylim(df[depth_col].min() * 0.8, df[depth_col].max() * 1.3)

    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(0,1.2), loc='upper left', fancybox=True, shadow=False, ncol=6, fontsize=16)

    ax.set_xlabel('TIME (days)')
    ax.set_ylabel(' '.join(depth_col.split('_')).upper() + f' ({depth_unit})')
    
    section = 'SECTION ' + decimal_fraction_str(hole_diameter) if (hole_diameter != 'all') else ''
    ax.set_title(section, y=1.17)

    plt.tight_layout()

    if save:
        plt.savefig(f'{save_folder}plot/{depth_col}_vs_time_bs_{bs}.png',dpi=150)

    plt.close()

    return None

#function to plot time vs depth
def plot_time_depth_rt_noenvelopes(df_rt_input: pd.DataFrame, 
                                   df_plan_input: pd.DataFrame, 
                                   hole_diameter: float, depth_col: str, 
                                   current_well_name: str,
                                   depth_unit = 'ft',
                                   TIME = TIME, HD = HD,
                                   perform_colors = perform_colors, 
                                   color_RT = color_RT,
                                   replace_dot = replace_dot,
                                   plot_folder = 'plot',
                                   save_folder = save_folder, save = True):

    """This function plots time versus depth graphs.
    Here df_input is time-based drilling data,
    df_rt_input is current well time-based drilling data,
    df_plan_input is a time-depth plan for a current well,
    df_bound is a performance boundary data,
    hole_diameter is a section diameter size,
    depth_col is a depth column name (e.g. HD, BD),
    WELL, TIME, HD, BS, represent standard column name, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,
    perform_colors is a list or colors for plotting,
    replace_dot is a symbol to replace dot in table name containing bit size 
    (e.g. if replace_dot = 'p' then '12.5' -> '12p5')
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function only plots and save the graph, but return none."""

    bs = str(hole_diameter).replace('.', replace_dot)

    #change RT color
    perform_colors[-1] = color_RT

    #copy input data  
    df_rt = df_rt_input.copy()
    df_plan = df_plan_input.copy()
        
    ############################
    #### Time vs Depth plot ####
    ############################
    _, ax = plt.subplots(1,1,figsize=(15.5,5))

    #factor to converst seconds to days
    time_conversion = 1/(24*60)
    #add plan
    ax.plot(df_plan[TIME]*time_conversion, df_plan[HD], lw=3, color='k', label = 'PLAN', linestyle = '--')
     
    #add current well
    ax.plot(df_rt[TIME]*time_conversion, df_rt[depth_col],
            lw=4,label = (current_well_name), color=perform_colors[-1])

    ax.set_xlim(0, max([df_rt[TIME].max(),df_plan[TIME].max()]) * time_conversion * 1.03)
    ax.set_ylim(0, df_plan[HD].max() * 1.03)

    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(0,1.2), loc='upper left', fancybox=True, shadow=False, ncol=6, fontsize=16)

    ax.set_xlabel('TIME (days)')
    ax.set_ylabel(' '.join(depth_col.split('_')).upper() + f' ({depth_unit})')
    
    #section = 'SECTION ' + str(hole_diameter) + '\"' if hole_diameter != 'all' else ''
    #ax.set_title(section, y=1.17)

    plt.tight_layout()

    if save:
        plt.savefig(f'{save_folder}{plot_folder}/{depth_col}_vs_time_rt_bs_all.png',dpi=150)

    plt.close()

    return None

#function to plot and save time distribution for selected and current well
def rig_activity_summary(df_input: pd.DataFrame, group_col: str,
                         super_activity: str, hole_diameter: float,
                         wells_select: list, current_well: int,
                         dtime_dict: dict, refs = None,
                         add_naming = '', MMDD = MMDD,
                         WELL = WELLID, TIME = TIME, AVG = AVG, 
                         RSUPER = RSUPER, RSUBACT = RSUBACT,
                         super_activity_dict = super_state_swap,
                         sub_activity_dict = sub_activity_state,
                         color_dict = rig_activity_color_dict, 
                         rig_activity_order = rig_activity_order,
                         round_ndigits = round_ndigits, replace_dot = replace_dot,
                         csv_folder = 'csv', plot_folder = 'plot',
                         save_folder = save_folder,  save = True):

    """This function computes and plots rig activity time distribution.
    Here df_input is time-based drilling data,
    group_col is columns name to groupby and compute time distribution for (e.g. WELLID, MMDD),
    super_activity is a specific super activity name (e.g. 'all', 'DRILL', 'TRIP'),
    hole_diameter is a section diameter size,
    wells_select is a list of selected historic well ids,
    current_well is a well id of a current well,
    dtime_dict is a dictionary that define mapping from well ids to time step/data density, 
    add_naming is a string to add to computed data table name (e.g. '_per_day' for group_col = MMDD),
    WELL, TIME, RSUPER, RSUBACT represent standard column name, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,
    super_activity_dict is a dictionary that define mapping from names to number codes,
    sub_activity_dict is a dictionary that define mapping from number codes to names,
    color_dict is a dictionary of colors that define mapping between activity code and color,
    rig_activity_order is a list of subactivities in a specific order,
    replace_dot is a symbol to replace dot in table name containing bit size
    (e.g. if replace_dot = 'p' then '12.5' -> '12p5')
    round_ndigits is number of digits to round calculations,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function returns time distribution table for all subactivities, as well as plots it."""

    bs = str(hole_diameter).replace('.', replace_dot)
    
    #copy input data
    df = df_input.copy()
    
    #select by rig super activity
    if super_activity == 'all':
        cond = np.ones(df.shape[0],dtype=bool)
    else:
        cond = (df[RSUPER] == super_activity_dict[super_activity])
        
    #select by hole section
    if hole_diameter == 'all':
        cond &= np.ones(df.shape[0],dtype=bool)
    else:
        cond &= (df[BS] == hole_diameter)
        
    #introduce auxillary dataframe, select only Drill Super Activity
    df_aux = (df.loc[cond]).groupby([group_col, RSUBACT])[TIME].nunique().reset_index()

    #account for variable sampling time step in seconds
    if group_col == WELL:
        df_aux['dtime'] = df_aux[WELL].map(dtime_dict)
    else:
        df_aux['dtime'] = dtime_dict[df[WELL].unique()[0]]

    #convert time from count to seconds
    df_aux[TIME] *= df_aux['dtime']
    #drop columns that are no longer needed
    df_aux.drop(columns=['dtime'],inplace=True)
    #convert df to series
    if df_aux.shape[0] > 1:
        df_aux = df_aux.set_index([group_col, RSUBACT]).squeeze().dt.total_seconds()/3600
    else:
        df_aux = df_aux.set_index([group_col, RSUBACT])[TIME].dt.total_seconds()/3600
    #pivot data
    total_time_sub_act = df_aux.reset_index().pivot(index = group_col, columns = RSUBACT, values = TIME)
    #map columns to activity names instead of ids
    total_time_sub_act.rename(columns = sub_activity_dict, inplace = True)

    ##remove inappropriate activities
    try:
        if (super_activity == 'DRILL'):
            total_time_sub_act.drop(columns = ['CNX (trip)'], inplace = True)
        if (super_activity == 'TRIP'):
            total_time_sub_act.drop(columns = ['CNX (drill)'], inplace = True)
    except:
        pass

    if add_naming != '_rt':                    
        #check if any well is missing
        if group_col == WELL:
            for w in wells_select + [current_well]:
                if w not in total_time_sub_act.index.tolist():
                    #print('Missing current well')
                    total_time_sub_act = total_time_sub_act.append(pd.DataFrame({w: np.zeros(total_time_sub_act.shape[1])}).T)
            #remove duplicate well_ids
            #total_time_sub_act = total_time_sub_act[total_time_sub_act.index.isin(df[WELL].unique())] 
      
    #reorder activities that are present
    rig_activity_order_add = [col for col in rig_activity_order if col not in total_time_sub_act.columns]
    for col in rig_activity_order_add:
        total_time_sub_act[col] = 0

    if add_naming == '_per_day':
        total_time_sub_act = total_time_sub_act.merge(pd.DataFrame(df[MMDD].unique(), columns = [MMDD]), 
                                                      how = 'outer', left_index=True, right_on = MMDD)\
                                               .set_index(MMDD).sort_index()
    total_time_sub_act.fillna(0, inplace = True)

    #reorder columns
    if super_activity == 'all':
        total_time_sub_act = total_time_sub_act[rig_activity_order]
    elif super_activity == 'DRILL':
        total_time_sub_act = total_time_sub_act[rig_activity_drill_order]
    elif super_activity == 'TRIP':
        total_time_sub_act = total_time_sub_act[rig_activity_trip_order]

    #calculate average from the final table
    if add_naming != '_rt':                    
        if group_col == WELL:
            #add average value for per well plot only
            add = pd.DataFrame(total_time_sub_act.loc[wells_select]\
                    .mean(axis=0)).rename(columns={0:AVG}).T
            total_time_sub_act = total_time_sub_act.append(add)
    
    #print(wells_select)
    #print('total_time_sub_act: \n',total_time_sub_act)

    if total_time_sub_act.empty:
        total_time_sub_act = total_time_sub_act.append(pd.Series(0, index=total_time_sub_act.columns, 
                                                                 name = current_well))

    if not(total_time_sub_act.empty):
        #############
        ### Plots ###
        #############

        if add_naming == '_rt':
            fig, axes = plt.subplots(2,1,figsize=(10,7))
            ylabel = ''
        else:
            fig, axes = plt.subplots(1,2,figsize=(20,7))
            ylabel = group_col.replace('_', ' ').upper()

        ################################
        ### Time distribution: hours ###
        ################################
        
        ax = axes[0]
        
        total_time_sub_act.plot.barh(stacked=True, ax=ax, color=color_dict)

        ax.grid(axis='y')
        ax.set(xlabel = 'TIME (hrs)', ylabel = ylabel)
        ax.get_legend().remove()

        if add_naming == '_rt':
            ax.legend(bbox_to_anchor=(1.05,1.05))
            ax.axes.yaxis.set_visible(False)

        #add R1, R2 highlights
        if (refs != None):
            xmin, xmax = ax.get_xlim()
            df_RS = pd.DataFrame({group_col: refs, 'label': ['R1','R2'], 'xmax': [xmax]*2})
            df_RS = df_RS.merge(total_time_sub_act.reset_index()[[group_col]], how='right', 
                                left_on = group_col, right_on = group_col).set_index(group_col)

            ax2 = ax.twinx()
            df_RS.plot.barh(y = 'xmax', width=0.8, ax = ax2, color = color_neutral, alpha=0.2)
            ax2.set_xlim(xmin, xmax)
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticklabels(df_RS['label'].replace(np.nan, '').values)
            ax2.grid(None)
            ax2.set_ylabel(None)
            ax2.get_legend().remove()

        ##################################
        ### Time distribution: percent ###
        ##################################

        ax = axes[1]
        #calculate percentage
        total_time_sub_act_pct = total_time_sub_act.div(total_time_sub_act.sum(axis=1), axis=0)*100
        total_time_sub_act_pct.plot.barh(stacked=True, ax=ax, color=color_dict)
        
        ax.grid(axis='y')
        ax.set(xlabel = 'TIME (%)', ylabel = ylabel)

        if add_naming != '_rt':
            ax.legend(bbox_to_anchor=(1.08,1))
        else:
            ax.get_legend().remove()
            ax.axes.yaxis.set_visible(False)

        #add R1, R2 highlights
        if (refs != None):
            xmin, xmax = ax.get_xlim()
            df_RS = pd.DataFrame({group_col: refs, 'label': ['R1','R2'], 'xmax': [xmax]*2})
            df_RS = df_RS.merge(total_time_sub_act.reset_index()[[group_col]], how='right', 
                                left_on = group_col, right_on = group_col).set_index(group_col)

            ax2 = ax.twinx()
            df_RS.plot.barh(y = 'xmax', width=0.8, ax = ax2, color = color_neutral, alpha=0.2)
            ax2.set_xlim(xmin, xmax)
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticklabels(df_RS['label'].replace(np.nan, '').values)
            ax2.grid(None)
            ax2.set_ylabel(None)
            ax2.get_legend().remove()

        #add multiindex for displaying purposes
        if add_naming != '_rt':
            title = super_activity.upper() + 'ING' if (super_activity != 'all') else 'DRILLING & TRIPPING'
            title_add = ': SECTION ' + decimal_fraction_str(hole_diameter) if (hole_diameter != 'all') else ''
            title_all = title + ' ACTIVITIES' + title_add
            
            plt.suptitle(title_all, y=0.95, fontsize=16)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save:
            
            #save plot
            plt.savefig(f'{save_folder}{plot_folder}/time_dist_act_{super_activity.lower()}_bs_{bs}{add_naming}.png',dpi=150)
            
            #save time dispribution in hrs
            df_save = total_time_sub_act.round(round_ndigits)
            df_save.rename(columns = {col: re.sub('[^a-zA-Z_]+', '', col.replace(' ', '_').replace('/', '_')).lower()
                                    for col in df_save.columns}, inplace=True)
            df_save.rename(columns = {'null':'nulo'}, inplace=True)
            #rename index and columns name
            df_save.index.names = [group_col]
            df_save.columns.names = [None]
            df_save.to_csv(f'{save_folder}{csv_folder}/time_dist_act_hours_{super_activity.lower()}_bs_{bs}{add_naming}.csv',index=True)
            
            #save time dispribution in %
            df_save = total_time_sub_act_pct.round(round_ndigits)
            df_save.rename(columns = {col: re.sub('[^a-zA-Z_]+', '', col.replace(' ', '_').replace('/', '_')).lower() 
                                    for col in df_save.columns}, inplace=True)
            df_save.rename(columns = {'null':'nulo'}, inplace=True)
            #rename index and columns name
            df_save.index.names = [group_col]
            df_save.columns.names = [None]
            
            df_save.to_csv(f'{save_folder}{csv_folder}/time_dist_act_pct_{super_activity.lower()}_bs_{bs}{add_naming}.csv',index=True)
            
        plt.close()

    return total_time_sub_act

#sub function to plot kpi
def kpi_boxplot_sub(s: pd.Series, 
                    q50_current: float,
                    title: str, ax,  
                    kpi_colors = kpi_colors, 
                    color_historic = color_historic, 
                    color_RT = color_RT) -> pd.DataFrame:

    """This function computes and plots kpi data per activity.
    Here s is a series of a specific variable (e.g. ROP, connection time, etc.),
    q50_current - 50th percentile of a current well for the same variable,
    hole_diameter is a section diameter size,
    kpi_colors is a list of colors,
    color_historic is historic data color
    color_RT is real time data color."""

    #calculate quantiles
    if not(s.empty):
        quantiles = s.dropna().quantile([0.01, 0.25, 0.5, 0.75, 0.95])
        q1, q25, q50, q75, q99 = quantiles.values

        if '(mm:ss)' in title:
            min_q = 10.0  if 'Weight' not in title else 60.0
            q99 = min([min_q, q99])
            quantiles[-1] = q99
            xticks_labels = [convert_time_decimal_mmss(label) for label in quantiles.values]
        else:
            xticks_labels = quantiles.values.round(1)

        #height
        h = 0.25
        #transparency
        alpha = 1
        
        #add rectangles to show good and bad regions
        #bad zone
        rect = patches.Rectangle((q1, 0), q25-q1, h, linewidth = 2, alpha = alpha,
                                edgecolor = kpi_colors[0], facecolor = kpi_colors[0])
        ax.add_patch(rect)

        #good zone
        rect = patches.Rectangle((q75, 0), q99-q75, h, linewidth = 2, alpha = alpha,
                                edgecolor = kpi_colors[-1], facecolor = kpi_colors[-1])

        # Add the patch to the Axes
        ax.add_patch(rect)

        #average zone
        rect = patches.Rectangle((q25, 0), q75-q25, h, linewidth=2, alpha=alpha,
                                edgecolor = kpi_colors[2], facecolor = kpi_colors[2])

        # Add the patch to the Axes
        ax.add_patch(rect)

        #add historic wells average
        ax.axvline(x = q50, ymin=0, ymax=1, linewidth=4, color=color_historic)
        #add current well
        ax.axvline(x = q50_current, ymin=0, ymax=1, linewidth=4, color=color_RT)
        #add title
        ax.set_title(title, fontweight="bold", fontsize=18)
        #set limits
        ax.set_ylim(0, h)
        ax.set_xticks(quantiles.values.round(1))
        ax.set_xticklabels(xticks_labels)

        y_axis = ax.axes.get_yaxis()
        y_axis.set_visible(False)
        
        df_save = pd.DataFrame({'q1': q1,
                                'q25': q25,
                                'q50': q50,
                                'q75': q75,
                                'q99': q99,
                                'q50_current_well': q50_current
                            },index=[title]).T

    else:

        df_save = pd.DataFrame({'q1': 0,
                                'q25': 0,
                                'q50': 0,
                                'q75': 0,
                                'q99': 0,
                                'q50_current_well': 0
                            },index=[title]).T

    return df_save

#function to plot kpi
def kpi_boxplot(df_, df_wt_wt_, df_trip_, df_conn_drill_, df_conn_trip_, 
                    hole_diameter: float,
                    wells_select: list, current_well: int, 
                    WELL = WELLID, ROP = ROP, 
                    CONNTTIME = CONNTTIME, CONNDTIME = CONNDTIME,
                    PIPESP = TRIPSPEED, WTWT = WTWT, dTIME = dTIME, 
                    rop_unit = 'ft/h', trip_speed_unit = 'ft/h',
                    activity_dict = sub_activity_state_swap,
                    kpi_colors = kpi_colors, 
                    replace_dot = replace_dot, round_ndigits = round_ndigits, 
                    save_folder = save_folder,  save = True) -> pd.DataFrame:

    """This function computes and plots kpi data.
    Here df_ is time-based drilling data,
    df_wt_wt_ is weight to weight dataframe,
    df_trip_ is trip dataframe,
    df_conn_drill_ is connection time while drilling dataframe,
    df_conn_trip_ is connection time while tripping dataframe,
    hole_diameter is a section diameter size,
    wells_select is a list of selected historic well ids,
    current_well is a well id of a current well,
    WELL, ROP, CONNTTIME, CONNDTIME, PIPESP, WTWT, dTIME represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    activity_dict is a dictionary that define mapping from names to number codes.
    kpi_colors is a list of colors,
    replace_dot is a symbol to replace dot in table name containing bit size 
    (e.g. if replace_dot = 'p' then '12.5' -> '12p5')
    round_ndigits is number of digits to round calculations,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function returns kpi dataframe."""
    
    #copy input data
    df = df_.copy()
    df_wt_wt = df_wt_wt_.copy()
    df_trip = df_trip_.copy()
    df_conn_drill = df_conn_drill_.copy()
    df_conn_trip = df_conn_trip_.copy()
    
    s = 'SECTION ' + decimal_fraction_str(hole_diameter) if (hole_diameter != 'all') else ''

    bs = str(hole_diameter).replace('.', replace_dot)
      
    #select hole size section
    if hole_diameter != 'all':

        df = df.loc[df[BS] == hole_diameter]
        df_trip = df_trip.loc[df_trip[BS] == hole_diameter]
        df_conn_drill = df_conn_drill.loc[df_conn_drill[BS] == hole_diameter]
        df_conn_trip = df_conn_trip.loc[df_conn_trip[BS] == hole_diameter]
        df_wt_wt = df_wt_wt.loc[df_wt_wt[BS] == hole_diameter]
        
    #store data quantiles together
    df_box = pd.DataFrame()
    
    _, axes = plt.subplots(11,1,figsize=(10,14))
    
    title = f'Drilling ROP ({rop_unit})'
    ax = axes[0]
    s_drill = df.loc[df[WELL].isin(wells_select)&
                     (((df[RSUBACT] == activity_dict['ROTATE']))
                      |((df[RSUBACT] == activity_dict['SLIDE']))), ROP]
    q50_current = df.loc[df[WELL].isin([current_well])&
                     (((df[RSUBACT] == activity_dict['ROTATE']))
                      |((df[RSUBACT] == activity_dict['SLIDE']))), ROP].median()

    df_box = kpi_boxplot_sub(s_drill, q50_current, title, ax).fillna(0) 
    ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))
    
    title = f'Rotating ROP ({rop_unit})'
    ax = axes[1]
    s_drill = df.loc[df[WELL].isin(wells_select)&
                     (df[RSUBACT] == activity_dict['ROTATE']), ROP]
    q50_current = df.loc[df[WELL].isin([current_well])&
                     (df[RSUBACT] == activity_dict['ROTATE']), ROP].median()

    df_box[title] = kpi_boxplot_sub(s_drill, q50_current, title, ax).fillna(0)
    ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))
    
    title = f'Sliding ROP ({rop_unit})'
    ax = axes[2]
    s_drill = df.loc[df[WELL].isin(wells_select)&
                     (df[RSUBACT] == activity_dict['SLIDE']), ROP]
    q50_current = df.loc[df[WELL].isin([current_well])&
                     (df[RSUBACT] == activity_dict['SLIDE']), ROP].median()
    
    if not(s_drill.empty):
        df_box[title] = kpi_boxplot_sub(s_drill, q50_current, title, ax).fillna(0)  
    else:
        df_box[title] = 0 
    ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))
    
    title = 'Connection Time (mm:ss)'
    ax = axes[3]
    if not(df_conn_drill.empty):
        s_conn = df_conn_drill.loc[df_conn_drill[WELL].isin(wells_select), CONNDTIME]\
                            .append(df_conn_trip.loc[df_conn_trip[WELL].isin(wells_select), CONNTTIME])
        q50_current = df_conn_drill.loc[df_conn_drill[WELL].isin([current_well]), CONNDTIME]\
                            .append(df_conn_trip.loc[df_conn_trip[WELL].isin([current_well]), CONNTTIME]).median()
        df_box[title] = kpi_boxplot_sub(s_conn, q50_current, title, ax, kpi_colors = kpi_colors[-1::-1]).fillna(0) 
        ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))
    else:
        s_conn = np.nan
        q50_current = np.nan
        df_box[title] = 0
   
    title = 'Connection Time Drill (mm:ss)'
    ax = axes[4]

    if not(df_conn_drill.empty):
        s_conn = df_conn_drill.loc[df_conn_drill[WELL].isin(wells_select), CONNDTIME]
        q50_current = df_conn_drill.loc[df_conn_drill[WELL].isin([current_well]), CONNDTIME].median()
        df_box[title] = kpi_boxplot_sub(s_conn, q50_current, title, ax, kpi_colors = kpi_colors[-1::-1]).fillna(0) 
        ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))
 
    else:
        s_conn = np.nan
        q50_current = np.nan
        df_box[title] = 0

    title = 'Connection Time Trip (mm:ss)'
    ax = axes[5]

    if not(df_conn_trip.empty):
        s_conn = df_conn_trip.loc[df_conn_trip[WELL].isin(wells_select), CONNTTIME]
        q50_current = df_conn_trip.loc[df_conn_trip[WELL].isin([current_well]), CONNTTIME].median()
        df_box[title] = kpi_boxplot_sub(s_conn, q50_current, title, ax, kpi_colors = kpi_colors[-1::-1]).fillna(0) 
        ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))

    else:
        s_conn = np.nan
        q50_current = np.nan
        df_box[title] = 0

                   
    title = f'Tripping Speed ({trip_speed_unit})'
    ax = axes[6]
    s_trip = df_trip.loc[df_trip[WELL].isin(wells_select), PIPESP]
    q50_current = df_trip.loc[df_trip[WELL].isin([current_well]), PIPESP].median()
    
    if not(s_trip.empty):
        df_box[title] = kpi_boxplot_sub(s_trip, q50_current, title, ax).fillna(0) 
    else:
        df_box[title] = 0
    ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))
    
    title = f'Tripping In Speed ({trip_speed_unit})'
    ax = axes[7]
    s_trip = df_trip.loc[(df_trip[WELL].isin(wells_select))&
                         (df_trip[RSUBACT] == activity_dict['TRIP IN']), PIPESP]
    q50_current = df_trip.loc[(df_trip[WELL].isin([current_well]))&
                               (df_trip[RSUBACT] == activity_dict['TRIP IN']), PIPESP].median()
    
    if not(s_trip.empty):
        df_box[title] = kpi_boxplot_sub(s_trip, q50_current, title, ax).fillna(0) 
    else:
        df_box[title] = 0
    ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))

    title = f'Tripping Out Speed ({trip_speed_unit})'
    ax = axes[8]
    s_trip = df_trip.loc[(df_trip[WELL].isin(wells_select))&
                          (df_trip[RSUBACT] == activity_dict['TRIP OUT']), PIPESP]
    q50_current = df_trip.loc[(df_trip[WELL].isin([current_well]))&
                               (df_trip[RSUBACT] == activity_dict['TRIP OUT']), PIPESP].median()
    
    if not(s_trip.empty):
        df_box[title] = kpi_boxplot_sub(s_trip, q50_current, title, ax).fillna(0) 
    else:
        df_box[title] = 0
    ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))
    
    title = 'Pipe Movement (mm:ss)'
    ax = axes[9]
    s_trip = df_trip.loc[df_trip[WELL].isin(wells_select), dTIME]
    #delete top 4% of data
    s_trip = s_trip.loc[s_trip < s_trip.quantile([0.96]).values[0]]
    q50_current = df_trip.loc[df_trip[WELL].isin([current_well]), dTIME].median()
    
    if not(s_trip.empty):
        df_box[title] = kpi_boxplot_sub(s_trip, q50_current, title, ax, kpi_colors = kpi_colors[-1::-1]).fillna(0) 
    else:
        df_box[title] = 0
    ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))

    title = 'Weight to Weight (mm:ss)'
    ax = axes[10]
    s_wtwt = df_wt_wt.loc[df_wt_wt[WELL].isin(wells_select), WTWT]
    #delete top 4% of data
    s_wtwt = s_wtwt.loc[s_wtwt < s_wtwt.quantile([0.96]).values[0]]

    q50_current = df_wt_wt.loc[df_wt_wt[WELL].isin([current_well]), WTWT].median()
    
    if not(s_wtwt.empty):
        df_box[title] = kpi_boxplot_sub(s_wtwt, q50_current, title, ax, kpi_colors = kpi_colors[-1::-1]).fillna(0) 
    else:
        df_box[title] = 0
    ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))

    try:
        #add legend with title
        plt.figlegend(['HISTORIC AVG','CANDIDATE WELL','WORST','BEST','INTERQUARTILE'], 
                    loc = 'upper center', ncol = 5, title = s,
                    bbox_to_anchor = (0.5,1.05), fontsize = 14)
        plt.tight_layout(pad = 0.9)

        #rename columns
        df_box.columns = [re.sub(r'\([^)]*\)', '', col).strip(' ').replace(' ', '_').lower() 
                        for col in df_box.columns]

        if save:
            #save kpi data
            df_box.round(round_ndigits).to_csv(f'{save_folder}csv/kpi_boxplot_bs_{bs}.csv',index=True, index_label='q')
            #save plot
            plt.savefig(f'{save_folder}plot/kpi_boxplot_bs_{bs}.png', dpi=150, bbox_inches="tight")

        plt.close()

    except:
        #save kpi data
        df_box.round(round_ndigits).to_csv(f'{save_folder}csv/kpi_boxplot_bs_{bs}.csv',index=True, index_label='q')

    return df_box

#function that adds tvd and well_section to time-based drill data
def add_tvd_well_section(df_input: pd.DataFrame, 
               df_survey_input: pd.DataFrame, 
               DATETIME = DATETIME, 
               HD = HD, TVD = TVD,
               WSEC = WSEC) -> pd.DataFrame:
    """This function takes time-based drilling and official_survey data 
    and adds interpolated tvd column to time_based_drill table.
    df_input is time-based drilling dataframe,
    df_survey_input is official survey dataframe,
    DATETIME, HD, TVD, WSEC represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    This function returns updated time-based drilling table with tvd column added."""

    #copy original data for only one well
    df = df_input.copy()
    df_survey = df_survey_input.copy()

    #merge time-based drill and official survey data
    df_ = df.merge(df_survey[[HD, TVD, WSEC]], how = 'outer', 
                   right_on = HD, left_on = HD).sort_values(by = [HD, DATETIME])\
                   .reset_index().drop(columns = ['index'])

    #fill first TVD with hole depth value
    cond_no_init_survey = (df_[HD] <= df_survey[HD].min())
    df_.loc[cond_no_init_survey, TVD] = df_.loc[cond_no_init_survey, HD]
    #fill missing initial value for WSEC 
    df_.loc[cond_no_init_survey, WSEC] = 'V'

    #find last datetime value available
    last_datetime_index = df_.loc[~df_[DATETIME].isnull(), DATETIME].index[-1]
    last_datetime = df_.loc[~df_[DATETIME].isnull(), DATETIME].values[-1]
    #find last tvd value
    last_tvd = df_.loc[(~df_[TVD].isnull()), TVD].values[-1]
    df_.loc[df_[DATETIME] == last_datetime, TVD] = last_tvd

    #remove data in the end after last datetime data available
    df_ = df_.iloc[df_.index <= last_datetime_index]
    #interpolate TVD
    tvd_interpolate = interp1d(df_survey[HD].values, df_survey[TVD].values)

    #linearly interpolate tvd in time-based drill dataframe
    cond_no_survey = cond_no_init_survey | (df_[HD] > df_survey[HD].max())
    df_.loc[~cond_no_survey, TVD] = tvd_interpolate(df_.loc[~cond_no_survey, HD].values)

    #fill forward missing well_section values
    df_[WSEC].fillna(method='ffill', inplace=True)

    #drop missing datetime
    df_.dropna(subset = [DATETIME], inplace=True)
    #fill forward missing tvd at the end
    df_[TVD].fillna(method='ffill', inplace=True)

    return df_

#function that adds any survey feature (tvd, incl, azm, dls) to time-based drill data 
#by interpolating official survey data
def add_survey(df_input: pd.DataFrame, df_survey_input: pd.DataFrame, 
               feature: str, 
               HD = HD, TVD = TVD, 
               DATETIME = DATETIME) -> pd.DataFrame:
    """This function takes time-based drilling and official_survey data 
    and adds interpolated feature column to time_based_drill table.

    Parameters
    ----------
    df_input : pd.DataFrame
              Time-based drilling data.
    df_survey_input : pd.DataFrame 
                     Official survey data.
    feature : string
            It can be any features in the survey
    Returns
    -------
    df_ : pd.DataFrame
          Updated time-based drilling table with tvd and well_section columns added."""

    #copy original data for only one well
    df = df_input.copy()
    df_survey = df_survey_input.copy()
    
    #convert to float
    df_survey[HD] = df_survey[HD].astype(float)

    #merge time-based drill and official survey data
    df_ = df.merge(df_survey[[HD, feature]], how = 'outer', 
                   right_on = HD, left_on = HD).sort_values(by = [HD, DATETIME])\
                   .reset_index().drop(columns = ['index'])

    #fill first 'feature' with hole depth value
    cond_no_init_survey = (df_[HD] <= df_survey[HD].min())
    if feature == TVD:
        df_.loc[cond_no_init_survey, feature] = df_.loc[cond_no_init_survey, HD]
    else:
        df_.loc[cond_no_init_survey, feature] = 0

    #find last datetime value available
    last_datetime_index = df_.loc[~df_[DATETIME].isnull(), DATETIME].index[-1]
    last_datetime = df_.loc[~df_[DATETIME].isnull(), DATETIME].values[-1]
    #find last tvd value
    last_feature = df_.loc[(~df_[feature].isnull()), feature].values[-1]
    df_.loc[df_[DATETIME] == last_datetime, feature] = last_feature

    #remove data in the end after last datetime data available
    df_ = df_.iloc[df_.index <= last_datetime_index]

    #interpolate 'feature'
    feature_interpolate = interp1d(df_survey[HD].values, df_survey[feature].values)

    #linearly interpolate 'feature' in time-based drill dataframe
    cond_no_survey = cond_no_init_survey | (df_[HD] > df_survey[HD].max())
    df_.loc[~cond_no_survey, feature] = feature_interpolate(df_.loc[~cond_no_survey, HD].values)

    #drop missing datetime
    df_.dropna(subset = [DATETIME], inplace=True)
    #fill forward missing 'feature' at the end
    df_[feature].fillna(method='ffill', inplace=True)

    #sort by DATETIME
    df_ = df_.sort_values(by = [DATETIME])
    
    return df_

#function to compute TVD, NS, EW, DLS for official survey
def calculate_tvd_ns_ew_dls(df_survey: pd.DataFrame,
                            HD = HD, INC = INC, AZM = AZM,
                            TVD = TVD, NS = NS, EW = EW, DLS = DLS) -> pd.DataFrame:
    """This function calculates tvd, ns, ew, dls using minimum curvature methon.
    Here df_survey is official survey data, 
    HD, INC, AZM, TVD, NS, EW, DLS represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    This fuction requires presense of HD (ft), INC (deg), AZM (deg) 
    in passed dataframe and it returns dataframe with computed
    TVD (ft), NS (ft), EW (ft) and DLS (degree/100ft) columns (default units)."""
    
    df = df_survey.copy()

    #define local variables
    INCR = INC + '_rad' #inclination, radian
    AZMR = AZM + '_rad' #azimuth, radians
    DL = 'DL_rad' #dog leg severity, radians
    CF = 'CF_rad' #curvature factor, radians
 
    #add inclination and azimuth in radians
    df[INCR] = np.radians(df[INC])
    df[AZMR] = np.radians(df[AZM])

    #shift direction in difference
    n = -1

    #dog leg severity, radians
    #DL = Arccos (Cos (WD2 - WD1) - Sin WD1 * Sin WD2 * (1 - Cos (HAZ2 - HAZ1)))
    df[DL] = np.arccos(np.cos(df[INCR].diff(1)) 
             - np.sin(df[INCR]) * np.sin(df[INCR].shift(n)) * (1-np.cos(df[AZMR].diff(1))))

    #fill missing values with zero
    df[DL].fillna(0, inplace = True)
    #compute DLS
    df[DLS] = ((df[DL]*100/df[HD].diff(1)) * (180/np.pi)).fillna(0)
    #account for division by 0 when df[HD].diff(1) == 0, i.e. replace inf
    df[DLS].replace([np.inf, -np.inf], 0, inplace = True)
    
    #curvature factor, radians
    #CF = 2 / DL * (Tan (DL / 2))
    df[CF] = 2/df[DL] * (np.tan(df[DL]/2))
    #fill missing values with one
    df[CF].fillna(1, inplace = True)
            
    #TVD = SUM (((MD2 - MD1) * (Cos WD2 + Cos WD1) / 2) * CF)
    df[TVD] = (np.cumsum(df[HD].diff(1) * (np.cos(df[INCR].shift(n)) + np.cos(df[INCR])) * df[CF]/2) \
               + df[HD].min()).fillna(method = 'ffill')
    #fill first TVD value with measured_depth
    df[TVD].iloc[0] = df[HD].iloc[0]

    #North = SUM ((MD2 - MD1)*((Sin WD1 * Cos HAZ1 + Sin WD2 * Cos HAZ2) / 2) * CF)
    df[NS] = (np.cumsum(df[HD].diff(1) * (np.sin(df[INCR])*np.cos(df[AZMR]) 
                                        + np.sin(df[INCR].shift(n))*np.cos(df[AZMR].shift(n))) * df[CF]/2))\
                                        .fillna(method = 'ffill')
    #fill first NS value with zero
    df[NS].iloc[0] = 0

    #East = SUM ((MD2 - MD1) * ((Sin WD1 * Sin HAZ1 + Sin WD2 * Sin HAZ2) / 2) * CF)
    df[EW] = (np.cumsum(df[HD].diff(1) * (np.sin(df[INCR])*np.sin(df[AZMR]) 
                                        + np.sin(df[INCR].shift(n))*np.sin(df[AZMR].shift(n))) * df[CF]/2))\
                                        .fillna(method = 'ffill')
    #fill first EW value with zero
    df[EW].iloc[0] = 0

    #drop auxiliary columns
    df.drop(columns = [INCR, AZMR, DL, CF])
    
    return df

#function to calculate Mechanical Specific Energy
def calculate_mse(df_input: pd.DataFrame, 
                  drill_cols = [TQ, RPM, ROP, WOB, BS], 
                  MSE = MSE) -> pd.DataFrame:
    """This function taked dataframe df and computes
    Mechanical Specific Energy (MSE) in kpsi units
    using specified drill column names present in df
    and it stores computed values in column named mse_col
    required units: 
    TQ - Torque (kft·lbf), 
    RPM - Revolutions Per Minute (rpm),
    ROP - Rate Of Penetration (ft/h),
    WOB - Weight On Bit (klb), 
    bit_diameter (in)."""

    from math import pi
    
    df = df_input.copy()

    #unpack drilling columns names
    TQ, RPM, ROP, WOB, BS = drill_cols
    #add MSE column to the loaded dataframe
    df[MSE] = np.nan
    
    #for ROP > 0 use full expression
    df.loc[df[ROP] > 0, MSE] = ((480*df.loc[df[ROP] > 0, TQ]*df.loc[df[ROP] > 0, RPM] \
                                     / (df.loc[df[ROP] > 0, ROP] * df.loc[df[ROP] > 0, BS]**2) +
                                     4*df.loc[df[ROP] > 0, WOB] / (pi * df.loc[df[ROP] > 0, BS]**2))*0.35)
    
    #for ROP == 0 use relevant part of the expression
    df.loc[df[ROP] == 0, MSE] = 0.35 * 4 * df.loc[df[ROP] == 0, WOB] / (pi * df.loc[df[ROP] == 0, BS]**2)
    
    return df

#*************************************************************************************************************#
#**********************************  dashboard data and plots: Drilling tab  *********************************#
#*************************************************************************************************************#

####################################
######### Custom Functions #########
####################################

#function to get y axis limits to show 
def get_yaxis_lims(df_input: pd.DataFrame, super_activity: str,
                   index: str, index_lims: list, depth_unit = 'ft',
                   TIME = TIME, DATETIME = DATETIME, HD = HD):

    """This function generates necessary variables 
    for plotting for drilling and tripping tabs.
    Here df_input is time-based drilling data,
    super_activity is super activity name ('DRILL', or 'TRIP'),
    index is selected index for plotting (TIME or HD),
    index_lims are specified limits for index variable,
    TIME, DATETIME, HD represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    Function returns the following variables:
    y_col is a column name for y-axis (e.g. HD or TIME),
    y_unit is a unit name for y-axis (e.g. 'ft' or 'hrs'),
    units_conversion is a multiplier to convert y_col units,
    ylims is y-axis limits to show on the graph."""
    
    #copy input data
    df = df_input.copy()
    #select proper columns to plot
    if super_activity == 'DRILL':
        y_col = HD
        y_unit = f' ({depth_unit})'
        units_conversion = 1
    elif super_activity == 'TRIP':
        y_col = TIME
        y_unit = ' (hr)'
        units_conversion = 1/60
    else:
        print("\n Incorrect selection for super_activity: choose 'DRILL' or 'TRIP'. \n")

    #find depth limits to show time filter data
    if index == DATETIME:
        filter_col = DATETIME
    elif index == HD:
        filter_col = HD
    else:
        print('\n Incorrect selection for index variable: choose DATETIME or HD. \n')

    ylims = (df.loc[(df[filter_col] >= index_lims[0])
                        &(df[filter_col] <= index_lims[1]), y_col]*units_conversion).agg([min,max]).values

    #convert nans to 0
    ylims = np.nan_to_num(ylims, copy=True, nan=0.0)

    return y_col, y_unit, units_conversion, ylims

#generic function to highlight selected days
def select_day_highlight(df_input: pd.DataFrame, 
                         R: str, R_name: str, 
                         units_conversion: float, 
                         y_col: str, ax,
                         MMDD = MMDD, 
                         color = color_neutral) -> pd.DataFrame:
    """This function highlight selected day and plots it.
    Here df_input is time-based drilling data, 
    R is reference day to highlight in MM.DD format,
    R_name is reference name to show on the plot, e.g. 'R1' or 'R2',
    units_conversion is a multiplier to convert y_col units,
    y_col is column name for y-axis (e.g. HD or TIME),
    ax is axes to add hatched area,
    MMDD is Month.Day column name,
    color is selected hatched area color,
    Function plots hatched area for reference day 
    and returns dataframe with selected region data."""
   
    #find depth limits
    s = df_input.loc[(df_input[MMDD] == R), y_col].agg([min, max])
    
    #unit conversion
    s *= units_conversion
        
    ymin, ymax = (s.loc['min'], s.loc['max'])
    #print('y-lims: ',ymin, ymax)
        
    #find x limits
    xmin, xmax = ax.get_xlim()
    #print('x-lims: ',xmin, xmax)
    
    if (ymin == ymax):
        ax.axhline(y = ymin, color = color)
        
    else:
        #R zone
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2,
                                 edgecolor = color, facecolor = color, alpha=0.1)

        # Add the patch to the Axes
        ax.add_patch(rect)
    
    #add label
    ax.text(xmax * 0.75, (ymax+ymin)/2, R_name, fontsize=18)
    
    #selected region data
    df_R = pd.DataFrame({MMDD: R, y_col + '_min': ymin, 
                         y_col + '_max': ymax}, index = [R_name])
    
    return df_R

#function computes ROP and MSE thresholds for available model constants (BS, FORMs)
def compute_model_consts(df_input: pd.DataFrame, 
                        ROP_thr: float, MSE_thr: float,
                        ROP = ROP, MSE = MSE, 
                        ROPm = ROP + '_median', MSEm = MSE + '_median') -> pd.Series:
    """This function takes historic drilling data and computes ROP and MSE thresholds.
    Here df_input is time-based drilling data,
    ROP_thr is a percentile thresholds (in 0-100 range) for ROP,
    MSE_thr is a percentile thresholds (in 0-100 range) for MSE,
    ROP, MSE represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    ROPm is ROP threshold column name,
    MSEm is MSE threshold column name,
    Function returns series with computed median values for ROP and MSE."""

    d = {}
    d[ROPm] = np.percentile(df_input[ROP], q = ROP_thr)
    d[MSEm] = np.percentile(df_input[MSE], q = MSE_thr)

    return pd.Series(d)

#function adds ROP-MSE class column for current well data
def add_rop_mse(df_input: pd.DataFrame,
                df_input_rt: pd.DataFrame,
                ROP = ROP, MSE = MSE, ROP_MSE = ROP_MSE, RSUBACT = RSUBACT, 
                activity_dict = sub_activity_state_swap) -> pd.DataFrame:
    """This function takes historic drilling data and computes ROP_MSE column.
    Here df_input is historic time-based drilling data,
    df_input_rt is current well time-based drilling data,
    ROP, MSE, ROP_MSE, RSUBACT represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    activity_dict is a subactivity dictionary.
    Function returns current well time-based drilling data with ROP_MSE column, 
    that takes values in a range [0,3], where
    {0: HROP&LMSE (GOOD), 
    1: HROP&HMSE (OK), 
    2: LROP&LMSE (SO-SO),
    3: LROP&HMSE (POOR)}.
    """
    df = df_input.copy()
    df_rt = df_input_rt.copy()

    #find model constants from historic data
    # model_consts = df.loc[(df[RSUBACT] == activity_dict['ROTATE'])|(df[RSUBACT] == activity_dict['SLIDE'])]\
    #                 .groupby([BS,FORM])\
    #                 .apply(compute_model_consts, ROP_thr=50, MSE_thr=50).reset_index()

    model_consts = df.loc[(df[RSUBACT] == activity_dict['ROTATE'])|(df[RSUBACT] == activity_dict['SLIDE'])]\
                     .groupby([BS,FORM])[ROP,MSE].median().reset_index()\
                     .rename(columns = {ROP: ROP+'_median', MSE: MSE+'_median'})

    #display(model_consts)
    
    #in case of multiple execution
    try:
        df_rt.drop(columns = [ROP + '_median', MSE + '_median', ROP_MSE], inplace = True)
    except:
        pass

    df_rt = df_rt.merge(model_consts, how = 'left', left_on = [BS, FORM], right_on = [BS, FORM])
    
    #define ROP&MSE classes: 0 - ROP_H+MSE_L, 1 - ROP_H+MSE_H, 2 - ROP_L+MSE_L, 3 - ROP_L+MSE_H
    df_rt[ROP_MSE] = np.ones(df_rt.shape[0]).astype(int)

    #class 1: ROP_H+MSE_H
    df_rt.loc[((df_rt[MSE] < df_rt[MSE + '_median']) & 
               (df_rt[ROP] >= df_rt[ROP + '_median'])),ROP_MSE] *= 0

    #class 2: ROP_L+MSE_L
    df_rt.loc[((df_rt[MSE] < df_rt[MSE + '_median']) & 
               (df_rt[ROP] < df_rt[ROP + '_median'])),ROP_MSE] *= 2

    #class 3: ROP_L+MSE_H
    df_rt.loc[((df_rt[MSE] >= df_rt[MSE + '_median']) & 
               (df_rt[ROP] < df_rt[ROP + '_median'])),ROP_MSE] *= 3

    #make nan if not rotate or slide
    df_rt.loc[~((df_rt[RSUBACT] == activity_dict['ROTATE'])|(df_rt[RSUBACT] == activity_dict['SLIDE'])),
             [ROP + '_median', MSE + '_median',ROP_MSE]] = np.nan

    return df_rt

#function that computes kpi color for different activities
def get_kpi_color(df_input: pd.DataFrame,
                  df_kpi_input: pd.DataFrame,
                  LABEL: str, compute_col: str, kpi_col: str,
                  name: str, kpi_ascend: bool,
                  WELLID = WELLID, BS = BS, HD = HD, BD = BD,
                  TIME = TIME, DATETIME = DATETIME, KPIC = KPIC,
                  replace_dot = replace_dot, round_ndigits = round_ndigits, 
                  read_folder = save_folder, save_folder = save_folder, save = True) -> pd.DataFrame:
    """This function computes kpi color for different activities.
    Here df_input is time-based drilling data,
    df_kpi_input is kpi activity data (e.g. df_conn_drill),
    LABEL is a column name for consecutive label (e.g. LABELcd),
    compute_col is a compute column name (e.g. CONNDTIME)
    kpi_col is kpi column name (e.g. CONNDTIME)
    name is an activity file name (e.g. 'conn_trip', 'conn_drill'),
    kpi_ascend is a flag for ascend or descend best direction for kpi,
    WELLID, BS, HD, TIME, DATETIME, KPIC represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,  
    replace_dot is a symbol to replace dot in table name containing bit size 
    (e.g. if replace_dot = 'p' then '12.5' -> '12p5')
    round_ndigits is number of digits to round calculations,   
    save_folder is a folder to save computed dataframe,
    save is a flag to save data. 
    Function returns kpi activity dataframe with kpi color codes in kpi_col column."""

    #*** fix on Aug 12: add activity in case they are missing ***#
    kpi_cols = [DATETIME, HD, BD, TIME, BS, compute_col, 
                compute_col + '_worst',compute_col + '_avg', 
                compute_col + '_best', KPIC]
 
    if df_kpi_input.empty:
        df_kpi = pd.DataFrame([[0]*len(kpi_cols)], columns = kpi_cols)
    elif df_kpi_input[compute_col].sum() == 0:
        df_kpi = pd.DataFrame([[0]*len(kpi_cols)], columns = kpi_cols)
    else:
        df = df_input.copy()
        #df_kpi = df_kpi_input.copy()

        df_aux = df.groupby([WELLID, LABEL])[[HD, BD, DATETIME, TIME]].min().reset_index()
        df_kpi = df_kpi_input.reset_index().merge(df_aux, how = 'left', 
                            left_on=[WELLID, LABEL], right_on =[WELLID, LABEL])

        #add kpi boundaries that depend on bit size
        df_kpi[compute_col + '_worst'] = np.nan
        df_kpi[compute_col + '_avg'] = np.nan
        df_kpi[compute_col + '_best'] = np.nan

        #get current well bit sizes
        bit_sizes = df[BS].unique()
        #remove no nans
        bit_sizes = [x for x in bit_sizes if x == x]

        #compute kpi boundaries for all section sizes
        for hole_diameter in bit_sizes:
            
            bs = str(hole_diameter).replace('.', replace_dot)

            kpi = pd.read_csv(f'{read_folder}csv/kpi_boxplot_bs_{bs}.csv',index_col=[0])
            kpi_low = kpi.loc['q25', kpi_col]
            kpi_avg = kpi.loc['q50', kpi_col]
            kpi_high = kpi.loc['q75', kpi_col]
            
            df_kpi.loc[df_kpi[BS] == hole_diameter, compute_col + '_best'] = kpi_low
            df_kpi.loc[df_kpi[BS] == hole_diameter, compute_col + '_avg']  = kpi_avg
            df_kpi.loc[df_kpi[BS] == hole_diameter, compute_col + '_worst']  = kpi_high

        #class 1: historic performance
        df_kpi[KPIC] = np.ones(df_kpi.shape[0]).astype(int)

        if kpi_ascend:
            #class 0: overperform
            df_kpi.loc[(df_kpi[compute_col] > df_kpi[compute_col + '_best']), KPIC] *= 0
            #class 2: underperform
            df_kpi.loc[(df_kpi[compute_col] < df_kpi[compute_col + '_worst']), KPIC] *= 2
        else:
            #class 0: overperform
            df_kpi.loc[(df_kpi[compute_col] < df_kpi[compute_col + '_best']), KPIC] *= 0
            #class 2: underperform
            df_kpi.loc[(df_kpi[compute_col] > df_kpi[compute_col + '_worst']), KPIC] *= 2

        df_kpi = df_kpi[kpi_cols].fillna(0)

    if save:
        df_kpi.round(round_ndigits).to_csv(f'{save_folder}csv/{name}_time_current_well.csv', index = False)

    return df_kpi

def get_kpi_color_nokpi(df_input: pd.DataFrame,
                        df_kpi_input: pd.DataFrame,
                        LABEL: str, compute_col: str, kpi_col: str, name: str, 
                        WELLID = WELLID, BS = BS, HD = HD, BD = BD,
                        TIME = TIME, DATETIME = DATETIME, 
                        round_ndigits = round_ndigits, 
                        csv_folder = 'csv',
                        save_folder = save_folder, save = True) -> pd.DataFrame:

    """This function computes kpi color for different activities.
    Here df_input is time-based drilling data,
    df_kpi_input is kpi activity data (e.g. df_conn_drill),
    LABEL is a column name for consecutive label (e.g. LABELcd),
    compute_col is a compute column name (e.g. CONNDTIME)
    kpi_col is kpi column name (e.g. CONNDTIME)
    name is an activity file name (e.g. 'conn_trip', 'conn_drill'),
    kpi_ascend is a flag for ascend or descend best direction for kpi,
    WELLID, BS, HD, TIME, DATETIME, KPIC represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,  
    replace_dot is a symbol to replace dot in table name containing bit size 
    (e.g. if replace_dot = 'p' then '12.5' -> '12p5')
    round_ndigits is number of digits to round calculations,   
    save_folder is a folder to save computed dataframe,
    save is a flag to save data. 
    Function returns kpi activity dataframe with kpi color codes in kpi_col column."""
    
    df = df_input.copy()

    kpi_cols = [DATETIME, HD, BD, TIME, BS, compute_col, WELLID, LABEL, RSUBACT]

    if kpi_col != compute_col:
        kpi_cols.append(kpi_col)
 
    if (df_kpi_input.empty) | (df_kpi_input[compute_col].sum() == 0):
        df_kpi = pd.DataFrame([[0]*len(kpi_cols)], columns = kpi_cols)
        df_kpi[DATETIME] = df[DATETIME].max()
    else:
        df_aux = df.groupby([WELLID, LABEL])[[HD, BD, DATETIME, TIME]].min().reset_index()
        df_kpi = df_kpi_input.reset_index().merge(df_aux, how = 'left', 
                            left_on=[WELLID, LABEL], right_on =[WELLID, LABEL])

    if save:
        df_kpi.round(round_ndigits).to_csv(f'{save_folder}{csv_folder}/{name}_nokpi_time_current_well.csv', index = False)

    return df_kpi


#function plots drilling performance
def drill_plot_performance(df_input: pd.DataFrame, 
                           df_form: pd.DataFrame,
                           df_bound: pd.DataFrame,
                           df_wt_wt_rt_kpi: pd.DataFrame,
                           df_conn_drill_rt_kpi: pd.DataFrame,
                           df_conn_trip_rt_kpi: pd.DataFrame,
                           y_col: str, y_unit: str, 
                           units_conversion: float, ylims: np.array,
                           refs: list, form_colors_dict: dict,
                           HD = HD, ROP = ROP, ROP_MSE = ROP_MSE,
                           FORM = FORM, FORMTOP = FORMTOP, FORMBOT = FORMBOT,
                           KPIC = KPIC, kpi_dict = kpi_dict,
                           perform_colors = perform_colors,
                           mse_rop_colors = mse_rop_colors,
                           save = True, save_folder = save_folder) -> None:
    """This function plots drilling performance graphs.
    Here df_input is time-based drilling data,
    df_form is formation tops dataframe,
    df_bound is ROP performance boundary dataframe,
    df_wt_wt_rt_kpi is weight to weight time dataframe,
    df_conn_drill_rt_kpi is conection drill time dataframe, 
    df_conn_trip_rt_kpi is conection trip time dataframe,
    y_col is column name for y-axis (e.g. HD or TIME),
    y_unit is unit name for y-axis (e.g. 'ft' or 'hrs')
    y_unit is a unit name for y-axis (e.g. 'ft' or 'hrs'),
    units_conversion is a multiplier to convert y_col units,
    ylims is y-axis limits to show on the graph,
    refs is a list of reference days to highlight in MM.DD format,
    form_colors_dict is a formation color dictionary,
    HD, ROP, ROP_MSE, FORM, FORMBOT, FORMTOP, KPIC represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions, 
    perform_colors is a list of performance colors,
    mse_rop_colors is a list of colors for MSE/ROP plot,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data. 
    Function returns plots drilling performance graphs and returns None."""

    df = df_input.copy()
    data = df.dropna(subset=[ROP_MSE, HD])

    #plot
    n_cols = 5

    _, axes = plt.subplots(1, n_cols, figsize = (2.5*n_cols,10), sharey=True)

    #####################
    ###Formation Tops ###
    #####################
    ax = axes[0]

    #add formation boundaries
    for bnd_top, bnd_bot, form in df_form[[FORMTOP, FORMBOT, FORM]].values:

        #convert units
        bnd_top *= units_conversion
        bnd_bot *= units_conversion

        color = form_colors_dict[form]
        height = bnd_bot - bnd_top
        
        rect = patches.Rectangle((0,bnd_top), 1, height, linewidth=2,
                                edgecolor=color, facecolor=color, alpha=0.5)
        #add the patch to the Axes
        ax.add_patch(rect)

        #find formation middle point to show text
        ax.axhline(bnd_top, lw=3, color='k')
        #add formation labels
        y_text = bnd_top + height/2 
        
        #display less symbols if formation name is too long
        if y_text < ylims[1]:
           ax.text(0.05, y_text + 35, form[:15], fontsize=12, fontweight="bold")

    ax.set(xlabel = '', 
            ylabel = y_col.replace('_',' ').upper() + y_unit, 
            xlim = (0, 1),
            ylim = ylims)

    #add highlights
    df_R = select_day_highlight(data, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(data, refs[1], 'R2', units_conversion, y_col, ax))

    ax.set_title('TOP')
    ax.invert_yaxis()

    ###################
    ###ROP vs Depth ###
    ###################
    ax = axes[1]
        
    #worst boundary
    ax.fill_betweenx(y=df_bound[HD], x1 = np.zeros(df_bound.shape[0]), x2 = df_bound[ROP + '_worst'],
                    color=perform_colors[0],alpha=0.1, label=None)
    ax.plot(df_bound[ROP + '_worst'], df_bound[HD], lw=1, color=perform_colors[0], label='WORST')

    #best boundary
    ax.fill_betweenx(y=df_bound[HD], x1 = np.ones(df_bound.shape[0])*df[ROP].max()*1.8, 
                    x2 = df_bound[ROP + '_best'],
                    color=perform_colors[2],alpha=0.1, label=None)
    ax.plot(df_bound[ROP + '_best'], df_bound[HD], lw=1, color=perform_colors[2], label='BEST')

    #average
    ax.plot(df_bound[ROP + '_avg'], df_bound[HD], lw=1.5, color=perform_colors[1], label='AVG')

    #current
    ax.plot(df[ROP], df[HD], lw=2, color=perform_colors[-1], label='RT')

    ax.legend(loc='center')

    xlim_rop = df[ROP].quantile(0.99) if df[ROP].shape[0] > 1 else 300

    #xmin, xmax = ax.get_xlim()
    ax.set(xlabel = '', ylabel = '', 
           xlim = (0 , xlim_rop),
           ylim = ylims)

    #add highlights
    df_R = select_day_highlight(data, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(data, refs[1], 'R2', units_conversion, y_col, ax))

    ax.set_title('ROP (ft/hr)')
    ax.invert_yaxis()

    ###################
    ##### MSE/ROP #####
    ###################
    ax = axes[2]
    mse_rop_colors_map = dict(zip([0,1,2,3], mse_rop_colors))
    sns.scatterplot(y=HD, x=ROP_MSE, data=data, hue = ROP_MSE, 
                    palette = mse_rop_colors_map, s=100, ax=ax, edgecolor=None)

    ax.plot(data[ROP_MSE].astype(int), data[HD], lw=0.5, color='k', alpha=0.5)

    try:
        ax.get_legend().remove()
    except:
        pass

    ax.set_xticks([0,3])
    ax.set_xticklabels(['GOOD','POOR'])

    ax.set(xlabel = '', ylabel = '', 
           xlim = (-0.5 , 3.5),
           ylim = ylims)

    #add highlights
    df_R = select_day_highlight(data, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(data, refs[1], 'R2', units_conversion, y_col, ax))

    ax.set_title('MSE/ROP')
    ax.invert_yaxis()

    ####################
    ## WT to WT (min) ##
    ####################
    ax = axes[3]

    hrs_conversion = 1/60 #to show in hrs

    for _, row in df_wt_wt_rt_kpi.iterrows():
      
        ax.plot(np.array([0, row[WTWT]])*hrs_conversion, np.array([row[HD]]*2), 
                lw=4, color=kpi_dict[row[KPIC]])
    
    xlim_wt = df_wt_wt_rt_kpi[WTWT].quantile(0.92)*hrs_conversion if df_wt_wt_rt_kpi[WTWT].shape[0] > 1 else 1

    ax.set(xlabel = '', ylabel = '', 
           xlim = (0 , xlim_wt),
           ylim = ylims)
    
    #add highlights
    df_R = select_day_highlight(data, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(data, refs[1], 'R2', units_conversion, y_col, ax))

    ax.set_title('WT to WT (hr)')
    ax.invert_yaxis()

    ######################
    ## SLP to SLP (min) ##
    ######################
    ax = axes[4]

    #connection drill time
    for _, row in df_conn_drill_rt_kpi.iterrows():
    
        ax.plot(np.array([0, row[CONNDTIME]]), np.array([row[HD]]*2), 
                lw=4, color=kpi_dict[row[KPIC]])
    
    #connection trip time
    for _, row in df_conn_trip_rt_kpi.iterrows():
    
        ax.plot(np.array([0, row[CONNTTIME]]), np.array([row[HD]]*2), 
                lw=4, color=kpi_dict[row[KPIC]])

    xlim_conn = df_conn_trip_rt_kpi[CONNTTIME].quantile(0.999) if df_conn_trip_rt_kpi[CONNTTIME].shape[0] > 1 else 5

    ax.set(xlabel = '', ylabel = '', 
           xlim = (0, xlim_conn),
           ylim = ylims)
    #print(df_conn_trip_rt_kpi[CONNTTIME].quantile(1.00))
    #add highlights
    df_R = select_day_highlight(data, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(data, refs[1], 'R2', units_conversion, y_col, ax))

    ax.set_title('SLP to SLP (min)')
    ax.invert_yaxis()

    plt.tight_layout()

    if save:
        plt.savefig(f'{save_folder}plot/drill_performance.png',dpi=150)

    plt.close()

    return None

#function to plot available paremeters versus selected index
def parameters_plot(df_input: pd.DataFrame, super_activity: str,
                    y_col: str, y_unit: str, 
                    units_conversion: float, ylims: np.array,
                    refs: list,
                    used_cols: list, used_units: list,
                    save_folder = save_folder, save = True) -> None:
    """This function plots drilling or tripping parameters.
    Here df_input is time-based drilling data,
    y_col is column name for y-axis (e.g. HD or TIME),
    y_unit is unit name for y-axis (e.g. 'ft' or 'hrs')
    y_unit is a unit name for y-axis (e.g. 'ft' or 'hrs'),
    units_conversion is a multiplier to convert y_col units,
    ylims is y-axis limits to show on the graph,
    refs is a list of reference days to highlight in MM.DD format,
    used_cols is list of parameters columns names (e.g. [SPP, GPM, WOB]
    used_units is a list of the units of selected parameter columns 
    (e.g. ['(psi)', '(gal/min)', '(klb)']), 
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function plots drilling or tripping parameters and returns None."""

    df = df_input.copy()

    #calculate number of subplots
    n_cols = len(used_cols)

    _, axes = plt.subplots(1, n_cols, figsize = (2.5 * n_cols, 10), sharey = True)

    for i, (col, unit) in enumerate(zip(used_cols,used_units)):
        
        ax = axes[i]
        (df.set_index(col)[y_col]*units_conversion).plot(ax = ax,color = 'k')
        
        ax.set_title((' '.join([col.upper()] + [unit]).replace('_', ' ')))
            
        #set common limits
        ax.set(xlabel = '', ylabel = y_col.replace('_',' ').upper() + y_unit, 
               xlim = (0 , df[col].max()*1.3),
               ylim = ylims)
        
        #add highlight
        df_R = select_day_highlight(df, refs[0], 'R1', units_conversion, y_col, ax)
        df_R = df_R.append(select_day_highlight(df, refs[1], 'R2', units_conversion, y_col, ax))
        
        ax.invert_yaxis()
         
    plt.tight_layout()

    if save:
        #save R1/R2 data
        df_R.round(round_ndigits).to_csv(f'{save_folder}csv/ref1_ref2_highlight_{y_col}.csv', index=True, index_label='r')
        #save plot
        plt.savefig(f'{save_folder}plot/{super_activity.lower()}_parameters.png',dpi=150)
    
    plt.close()

    return None

#function to compute rotate and slide drill depth intervals per stand
def drill_per_stand(df_input: pd.DataFrame, stand_length: float,
                    y_col: str, y_unit: str, 
                    units_conversion: float, ylims: np.array, refs: list,
                    bins_labels_dict: dict,
                    HD = HD, STAND = STAND, RSUBACT = RSUBACT,
                    activity_dict = sub_activity_state, 
                    color_dict = rig_activity_color_dict,
                    round_ndigits = round_ndigits,
                    save_folder = save_folder, save = True) -> pd.DataFrame:
    """This function plots drilling or tripping parameters.
    Here df_input is time-based drilling data, 
    stand_length is a stand length in ft,
    y_col is column name for y-axis (e.g. HD or TIME),
    y_unit is unit name for y-axis (e.g. 'ft' or 'hrs')
    y_unit is a unit name for y-axis (e.g. 'ft' or 'hrs'),
    units_conversion is a multiplier to convert y_col units,
    ylims is y-axis limits to show on the graph,
    refs is a list of reference days to highlight in MM.DD format,
    bins_labels_dict is a stand label/number - measured depth midpoint of a stand,
    HD, STAND, RSUBACT represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,
    activity_dict is a rig subactivity dictionary,
    color_dict is a rig subactivity color dictionary,
    round_ndigits is number of digits to round calculations,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function returns rotate and slide drill depth intervals 
    computed per stand dataframe and plots it for specified interval.
    """

    df = df_input.copy()

    #swap key: value
    activity_dict_swap = {value:key for key, value in activity_dict.items()}

    df.dropna(subset = [STAND], inplace = True)
    df[STAND] = df[STAND].astype(int)

    stand_cols = [STAND, 'rotate', 'slide', HD + '_middle']

    #*** fix on Aug 16: add rotate and slide columns in case they are missing ***#
    if ([1] in df[RSUBACT].unique()) | ([2] in df[RSUBACT].unique()):
        df_per_stand = df.loc[df[RSUBACT].isin([activity_dict_swap['ROTATE'],
                                                activity_dict_swap['SLIDE']])]\
                                        .groupby(STAND)[RSUBACT]\
                        .value_counts(normalize=True)*100

        df_per_stand = df_per_stand.unstack(level=1).fillna(0).reset_index()\
                                .rename(columns = activity_dict)
        #shift depth to midpoint of a stand 
        df_per_stand[HD] = df_per_stand[STAND].map(bins_labels_dict)

        #plot
        _, ax = plt.subplots(1,1,figsize=(4,10))

        #*** fix on Aug 11: add rotate and slide columns in case they are missing ***#
        for col in ['ROTATE','SLIDE']:
            if not(df_per_stand.columns.isin([col]).any()):
                df_per_stand[col] = 0
            
        df_per_stand.set_index(HD)[['ROTATE','SLIDE']].plot.barh(stacked=True, ax = ax,
                                                                color=[color_dict['ROTATE'],
                                                                color_dict['SLIDE']])

        #add highlight
        df_R = select_day_highlight(df, refs[0], 'R1', units_conversion, STAND, ax)
        df_R = df_R.append(select_day_highlight(df, refs[1], 'R2', units_conversion, STAND, ax))

        #process stand data
        df_per_stand.columns.names = [None]
        df_per_stand.rename(columns = {HD: HD + '_middle'}, inplace = True)
        #lower case
        df_per_stand.columns = [col.lower() for col in df_per_stand.columns]

        ax.set_xlim(0,100)
        ax.set_ylabel(y_col.replace('_',' ').upper() + y_unit)
        ax.set_yticks(df_per_stand.index[::5])
        ax.set_yticklabels(df_per_stand[HD + '_middle'][::5].values)
        ax.set_ylim(abs(df_per_stand[HD + '_middle'].astype(float) - ylims[0]).idxmin(),
                    abs(df_per_stand[HD + '_middle'].astype(float) - ylims[1]).idxmin())
        ax.invert_yaxis()
        ax.set_title('SLD vs ROT (%)')
        ax.legend(title=None)

        plt.tight_layout()
        if save:
            plt.savefig(f'{save_folder}plot/slide_rotate_per_stand.png',dpi=150)
        plt.close()

    else:
        df_per_stand =  pd.DataFrame([[0]*len(stand_cols)], columns = stand_cols)
        cols = ['r', MMDD, STAND + '_min', STAND + '_max']
        df_R =  pd.DataFrame([[np.nan]*len(cols)], columns = cols).set_index('r')

    if save:
        #save per stand data data
        df_per_stand[stand_cols].round(round_ndigits).to_csv(f'{save_folder}csv/rotate_slide_depth_per_stand.csv', index=False)
        #save R1/R2 data
        df_R.round(round_ndigits).to_csv(f'{save_folder}csv/ref1_ref2_highlight_stand.csv', index=True, index_label='r')

    return df_per_stand

#function to plot directional graph
def directional_plot(df_survey: pd.DataFrame, df_plan: pd.DataFrame, 
                     df_input: pd.DataFrame, 
                     y_col: str, y_unit: str, 
                     units_conversion: float, ylims: np.array,
                     refs: list,
                     used_cols: list, used_units: list, 
                     add_zones = False,
                     color_RT = color_RT, color = color_neutral,
                     DLS = DLS, HD = HD, highlight_depth = [],
                     plot_folder = 'plot', 
                     save_folder = save_folder, save = True) -> None:

    """This function plots directional graph.
    Here df_survey is official survey data, 
    df_plan is current well directional drilling data,
    df_input is time-based drilling data,
    y_col is column name for y-axis (e.g. HD or TIME),
    y_unit is unit name for y-axis (e.g. 'ft' or 'hrs')
    y_unit is a unit name for y-axis (e.g. 'ft' or 'hrs'),
    units_conversion is a multiplier to convert y_col units,
    ylims is y-axis limits to show on the graph,
    refs is a list of reference days to highlight in MM.DD format,
    used_cols is list of parameters columns names (e.g. [INC, AZM, DLS]
    used_units is a list of the units of selected parameter columns 
    (e.g. ['(deg)', '(deg)', '(deg/100ft)']), 
    color_RT is color to plot real time data,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function plots directional drilling data and returns None."""

    df = df_survey.copy()
    df_plan = df_plan.copy()

    #calculate number of subplots
    n_cols = len(used_cols)

    _, axes = plt.subplots(1, n_cols, figsize = (3 * n_cols, 10), sharey = True)

    for i, (col, unit) in enumerate(zip(used_cols, used_units)):
        
        ax = axes[i]
        (df.set_index(col)[y_col]*units_conversion).plot(ax = ax, color = color_RT, label = 'ACTUAL', lw = 3)
        (df_plan.set_index(col)[y_col]*units_conversion).plot(ax = ax, color = 'k', label = 'PLAN', linestyle = '--')

        
        ax.set_title((' '.join([col.upper()] + [unit]).replace('_', ' ')))
        
        xmin, xmax = (0, 1.2*max([df[col].max(),df_plan[col].max()]))
        
        if y_col == HD:
            ylims = (0, 1.01*max([df[y_col].max(),df_plan[y_col].max()]))
        #set common limits
        ax.set(xlabel = '', ylabel = y_col.replace('_',' ').upper() + y_unit, 
               xlim = (xmin, xmax),
               ylim = ylims)
        
        #add highlight
        if refs != None:
            df_R = select_day_highlight(df_input, refs[0], 'R1', units_conversion, y_col, ax)
            df_R = df_R.append(select_day_highlight(df_input, refs[1], 'R2', units_conversion, y_col, ax))
        
        if highlight_depth != []:
            rect = patches.Rectangle((xmin, highlight_depth[0]), 
                                      xmax-xmin, highlight_depth[1]-highlight_depth[0], linewidth=2,
                                 edgecolor = color, facecolor = color, alpha=0.1)
            ax.add_patch(rect)

        ax.invert_yaxis()

        #show legend on first plot
        if i == 0:
            ax.legend()

        if (add_zones) & (col == DLS):
            #add hatched areas
            ax2 = ax.twinx()

            df_plan[y_col] *= units_conversion

            ss = 0.1
            ax2.fill_betweenx(y = df_plan[y_col], x1 = df_plan[col]*(1-ss), x2 = df_plan[col]*(1+ss), 
                              color = color_overperf, alpha=0.2, label = "0% - 10%")

            ss = 0.2
            ax2.fill_betweenx(y = df_plan[y_col], x1 = df_plan[col]*(1-ss), x2 = df_plan[col]*(1-0.1), 
                            color = color_historic, alpha=0.2, label = "11% - 20%")
            ss = 0.3
            ax2.fill_betweenx(y = df_plan[y_col], x1 = df_plan[col]*(1-ss), x2 = df_plan[col]*(1-0.2), 
                            color = color_underperf, alpha=0.2, label = "21% - 30%")

            ax2.legend(loc = 'upper right', title = 'DLS Variation')

            ss=0.2
            ax2.fill_betweenx(y = df_plan[y_col], x1 = df_plan[col]*(1+0.1), x2 = df_plan[col]*(1+ss), 
                            color = color_historic, alpha=0.2, label = '0 - 20% DLS variation')

            ss=0.3
            ax2.fill_betweenx(y = df_plan[y_col], x1 = df_plan[col]*(1+0.2), x2 = df_plan[col]*(1+ss), 
                            color = color_underperf, alpha=0.2, label = '0 - 30% DLS variation')

            ax2.invert_yaxis()
            ax2.set_ylim(ax.get_ylim())
         
    plt.tight_layout()

    if save:
        #save plot
        plt.savefig(f'{save_folder}{plot_folder}/directional_evaluation.png',dpi=150)
    
    plt.close()

    return None

#helper function to compute depth drilled or tripped
def activity_depth(df_input: pd.DataFrame, 
                   activity: str, depth_col: str, compute_col: str, 
                   group_col: str, dtime_dict: dict,
                   activity_dict = sub_activity_state_swap, 
                   WELLID = WELLID, DATETIME = DATETIME, 
                   RSUBACT = RSUBACT, HD = HD) -> pd.DataFrame:
    """This function computes depth drilled or tripped.
    Here df_input is time-based drilling data,
    activity is a rig subactivity,
    depth_col is depth column name (e.g. HD, BD),
    compute_col is a compute column name (e.g. ROP, PIPESP),
    group_col is a columns name to groupby (e.g. MMDD),
    dtime_dict is a dictionary that define mapping from well ids to time step/data density, 
    activity_dict is a rig subactivity dictionary, 
    WELLID, DATETIME, RSUBACT represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    Function returns dataframe with computed depth for selected subactivity."""

    df = df_input.copy()

    df[depth_col + '_diff'] = df[depth_col].diff()
    df[DATETIME + '_diff'] = df[DATETIME].diff()

    df = df.loc[df[RSUBACT] == activity_dict[activity]]

    #*** fix on Aug 13: add activity in case they are missing ***#
    df_perform = pd.DataFrame(index = df_input[group_col].unique()).rename_axis(group_col)

    if df.empty:

        print(f'No {activity} activity in performance calculations, fill with zeroes.')
        df_perform[activity.lower().replace(' ', '_') + '_depth'] = 0
        df_perform[activity.lower().replace(' ', '_') + '_' + compute_col.lower()] = 0

    else:

        time_step = dtime_dict[df[WELLID].unique()[0]]

        cond = (df[DATETIME + '_diff'] == time_step)
        
        df_perform[activity.lower().replace(' ', '_') + '_depth'] = df.loc[cond]\
                                                                      .groupby(group_col)[depth_col + '_diff'].sum().abs()
        df_perform[activity.lower().replace(' ', '_') + '_' + compute_col.lower()] = df.loc[cond]\
                                                                                       .groupby(group_col)[compute_col].median()
            
    return df_perform

#function to compute and plot depth drilled or tripped
def compute_plot_depth(df_input: pd.DataFrame,
                       activities: list, super_activity: str,
                       dtime_dict: dict, refs: list,
                       MMDD = MMDD, HD = HD, BD = BD, 
                       TIME = TIME, STAND = STAND,
                       ROP = ROP, PIPESP = PIPESP, RSUBACT = RSUBACT,
                       color_dict = rig_activity_color_dict,
                       save_folder = save_folder, save = True) -> pd.DataFrame:
    """This function computes and plots depth drilled or tripped.
    Here df_input is time-based drilling data,
    activities is a list of rig sub activities names (e.g. ['ROTATE', 'SLIDE']),
    super_activity is a super activity name (e.g. 'DRILL', 'TRIP'),
    dtime_dict is a dictionary that define mapping from well ids to time step/data density, 
    refs is a list of reference days to highlight in MM.DD format,
    MMDD, HD, BD, ROP, PIPESP, RSUBACT represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    color_dict is a rig subactivity color dictionary,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function plots and returns complete depth drilled or tripped dataframe."""

    #select super activity parameters
    if super_activity == 'DRILL':
        depth_col = HD 
        compute_col = ROP 
    elif super_activity == 'TRIP':
        depth_col = BD 
        compute_col = PIPESP

    df_perform = activity_depth(df_input, activities[0], depth_col, compute_col, MMDD, 
                                dtime_dict)
    df_perform = df_perform.merge(activity_depth(df_input, activities[1], depth_col, compute_col, MMDD, 
                                dtime_dict),
                                how = 'outer', right_index=True, left_index=True)

    #depth columns for selected activity
    depth_cols = [(activity.lower().replace(' ', '_') + '_depth') for activity in activities]

    if super_activity == 'TRIP':
        #recompute pipe speed and reassign
        df_perform_fix = activity_depth(df_input, activities[0], depth_col, compute_col, MMDD, 
                                        dtime_dict, RSUBACT = RSUBACT + '_trip')
        df_perform_fix = df_perform_fix.merge(activity_depth(df_input, activities[1], depth_col, compute_col, MMDD, 
                                          dtime_dict, RSUBACT = RSUBACT + '_trip'),
                                          how = 'outer', right_index=True, left_index=True)

        for col in ['trip_in_pipe_speed','trip_out_pipe_speed']:
            df_perform[col] = df_perform_fix[col]

        #initialize pct
        for col in depth_cols:
            df_perform[col + '_pct'] = 0

        #add depths while wash_in, wash_out, ream_up, ream_down 
        for activity in ['WASH IN', 'WASH OUT', 'REAM UP', 'REAM DOWN']:
            df_perform = df_perform.merge(activity_depth(df_input, activity, BD, ROP, MMDD, dtime_dict)\
                                   .drop(columns = [(activity + '_' + ROP).lower().replace(' ','_')]),
                                         how = 'outer', left_index = True, right_index = True)

        #recompute trip_in/out_depth_pct: with respect to move_up and move_down
        df_perform['trip_in_depth_pct'] = df_perform['trip_in_depth'] / \
                                          df_perform[['trip_in_depth','wash_in_depth','ream_down_depth']].sum(axis=1) * 100
        df_perform['trip_out_depth_pct'] = df_perform['trip_out_depth'] / \
                                           df_perform[['trip_out_depth','wash_out_depth','ream_up_depth']].sum(axis=1) * 100

    else:

        #add percentage columns based only on selested activities depth
        for col in depth_cols:
            df_perform[col + '_pct'] = df_perform[col] / df_perform[depth_cols].sum(axis=1) * 100

    #add HD, TIME, STAND boundaries for drill_performance_per_day table only
    if super_activity == 'DRILL': 
        add_cols = [HD, TIME, STAND]
        df_aux = df_input.groupby(MMDD)[add_cols].agg(['min','max'])
        df_aux.columns = [col[0] + '_' + col[1] for col in df_aux.columns]

        #convert cumulative time minutes to hours
        df_aux[TIME + '_min'] *= 1/60
        df_aux[TIME + '_max'] *= 1/60

        df_perform = df_perform.merge(df_aux, how = 'left', 
                                    right_index = True, left_index=True)
    #fill missing
    for column in df_perform.select_dtypes(include=['category']):
        df_perform[column] = df_perform[column].cat.add_categories(0)
    df_perform.fillna(0, inplace=True)
    
    ##plot
    _, ax = plt.subplots(1,1,figsize=(8,6))

    color_depth = {col: color_dict[col[:-6].replace('_', ' ').upper()] for col in depth_cols}

    df_perform[depth_cols].plot.barh(stacked=True, ax=ax, color = color_depth)

    ax.grid(axis='x')
    ax.set(ylabel = 'DATE (month.day)', xlabel = f'DEPTH {super_activity}ED (ft)')

    legend = ax.get_legend()
    ax.legend([str(x._text)[:-6].upper().replace('_', ' ') for x in legend.texts])
    ax.tick_params(labeltop=False, labelright=True)
    xmin, xmax = ax.get_xlim()

    #add R1, R2 highlights
    df_RS = pd.DataFrame({MMDD: refs, 'label':['R1','R2'], 'xmax':[xmax]*2})
    df_RS = df_RS.merge(df_perform.reset_index()[[MMDD]], how='right', 
                left_on = MMDD, right_on = MMDD).set_index(MMDD)

    ax2 = ax.twinx()
    df_RS.plot.barh(y = 'xmax', width=0.8, ax = ax2, color = color_neutral, alpha=0.2)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticklabels(df_RS['label'].replace(np.nan, '').values)

    ax2.grid(None)
    ax2.set_ylabel(None)
    ax2.get_legend().remove()

    plt.tight_layout()

    if save:
        #save plot
        plt.savefig(f'{save_folder}plot/{super_activity.lower()}_performance_per_day.png',dpi=150)
        #save data
        df_perform.round(round_ndigits).to_csv(f'{save_folder}csv/{super_activity.lower()}_performance_per_day.csv',
                                               index=True, index_label=MMDD)

        # #save full performance table for tripping activity
        # if super_activity == 'TRIP':
        #     df_perform.round(round_ndigits).to_csv(f'{save_folder}csv/{super_activity.lower()}_performance_per_day_full.csv',
        #                                                       index=True, index_label=MMDD)
    
    plt.close()

    return df_perform

#function plots drill or trip performance for selected reference days
def plot_refs_performance(df_perform: pd.DataFrame, 
                          activities: list, super_activity: str, 
                          compute_col: str, refs: list,
                          color_dict = rig_activity_color_dict,
                          save_folder = save_folder, save = True) -> None:
    """This function plots drill or trip performance for selected reference days.
    Here df_perform is a computed drill or trip performance dataframe,
    activities is a list of subactivity names (e.g. ['ROTATE', 'SLIDE']),
    super_activity is a super activity name (e.g. 'DRILL', 'TRIP'),
    compute_col is a name of compute column (e.g. ROP, PIPESP),
    refs is a list of reference days to highlight in MM.DD format,
    color_dict is a rig subactivity color dictionary,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function plots perfarmance per day and returns None.
    """

    _, axes = plt.subplots(1, 2, figsize = (10, 5))

    for i, activity in enumerate(activities):
        #R1 plot
        ax = axes[i]

        df_R = df_perform.loc[refs[i]]
        colors = [color_dict[activity], 'grey'] if i==0 else ['grey', color_dict[activity]]

        df_R.iloc[-2:].plot.pie(ax=ax, legend=False, \
                                autopct=None, fontsize=16,\
                                shadow=False, startangle=0, labels=None,
                                colors= colors,
                                wedgeprops=dict(width=.3))

        ax.set_ylabel('')
        ax.set_title(f'{activity}: R{i+1} = ' + refs[i])

        act_name = activity.lower().replace(' ','_')
        #add text
        ax.text(-0.25,0.1,str(int(df_R[act_name + '_depth_pct'])) + '%',fontsize=40)
        ax.text(-0.6,-0.1,str(int(df_R[act_name + '_depth'])) + ' (ft), ' 
                        + str(int(df_R[act_name + '_' + compute_col.lower()])) + ' (ft/h)',fontsize=20)

    plt.tight_layout()

    if save:
        plt.savefig(f'{save_folder}plot/{super_activity.lower()}_performance_ref1_ref2.png',dpi=150)

    plt.close()

    return None

#*************************************************************************************************************#
#**********************************  dashboard data and plots: Tripping tab  *********************************#
#*************************************************************************************************************#

#compute pipe speed and movement boundaries
def compute_pipe_speed_envelope(df_input: pd.DataFrame,
                                df_trip_input: pd.DataFrame,
                                LABEL: str, name: str,
                                WELLID = WELLID, BS = BS, HD = HD,
                                PIPESP = PIPESP, PIPEMV = PIPEMV, dTIME = dTIME,
                                TIME = TIME, DATETIME = DATETIME, KPIC = KPIC,
                                round_ndigits = round_ndigits,
                                save_folder = save_folder, save = True) -> pd.DataFrame:
    """This function computes pipe speed and movement performance boundaries.
    Here df_input is time-based drilling data,
    df_trip_input is a tripping activity data,
    LABEL is column name for consecutive labels,
    name is a saved file name prefix,
    WELLID, BS, HD, PIPESP, PIPEMV, dTIME, TIME, DATETIME,
    KPIC represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,
    round_ndigits is number of digits to round calculations,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function returns pipe speed and movement performance boundaries dataframe."""

    df_rt = df_input.copy()
    df_trip_rt = df_trip_input.copy()
    #compute pipe speed
    df_trip_rt_kpi = get_kpi_color(df_rt, df_trip_rt, LABEL, PIPESP, 'tripping_speed',
                                   name, False, save = False).drop(columns = [KPIC])

    #*** fix on Aug 14: add activity in case they are missing ***#
    kpi_cols = [DATETIME, HD, TIME, BS, 
                PIPESP, PIPESP + '_worst',PIPESP + '_avg', PIPESP + '_best',
                PIPEMV, PIPEMV + '_worst',PIPEMV + '_avg', PIPEMV + '_best']

    if df_trip_rt_kpi.empty:
        df_trip_rt_kpi = pd.DataFrame([[0]*len(kpi_cols)], columns = kpi_cols)
    elif df_trip_rt_kpi[PIPESP].sum() == 0:
        df_trip_rt_kpi = pd.DataFrame([[0]*len(kpi_cols)], columns = kpi_cols)
    else:
        #add average
        df_trip_rt_kpi[PIPESP + '_avg'] = (df_trip_rt_kpi[PIPESP + '_best'] + df_trip_rt_kpi[PIPESP + '_worst'])/2
        #fix pipe_speed kpi boundaries:
        df_add = df_trip_rt_kpi[[PIPESP + '_best', PIPESP + '_worst']].copy()
        df_trip_rt_kpi[PIPESP + '_best'] = df_add[PIPESP + '_worst']
        df_trip_rt_kpi[PIPESP + '_worst'] = df_add[PIPESP + '_best']

        #compute pipe movement
        cols_select = [PIPEMV, PIPEMV + '_worst', PIPEMV + '_best']
        df_trip_rt_kpi[cols_select] = get_kpi_color(df_rt, df_trip_rt.rename(columns = {dTIME: PIPEMV}), LABEL, PIPEMV, PIPEMV,
                                                    name, True, save = False)[cols_select]
        #add average
        df_trip_rt_kpi[PIPEMV + '_avg'] = (df_trip_rt_kpi[PIPEMV + '_best'] + df_trip_rt_kpi[PIPEMV + '_worst'])/2

    if save:
        df_trip_rt_kpi.round(round_ndigits)[kpi_cols].to_csv(f'{save_folder}csv/{name}_time_current_well.csv', index = False)

    return df_trip_rt_kpi

#plot tripping performance
def trip_plot_performance(df_input: pd.DataFrame, 
                          df_form: pd.DataFrame,
                          df_bound: pd.DataFrame,
                          df_conn_trip_rt_kpi: pd.DataFrame,
                          y_col: str, y_unit: str, 
                          units_conversion: float, ylims: np.array,
                          refs: list, form_colors_dict: dict,
                          FORM = FORM, FORMTOPTIME = FORMTOPTIME, FORMBOTTIME = FORMBOTTIME,
                          CONNTTIME = CONNTTIME, PIPESP = PIPESP, PIPEMV = PIPEMV,
                          save = True, save_folder = save_folder) -> None:
    """This function plots tripping performance graphs.
    Here df_input is time-based drilling data,
    df_form is formation tops dataframe,
    df_bound is PIPESP and PIPEMV performance boundary dataframe,
    df_conn_trip_rt_kpi is conection trip time dataframe,
    y_col is column name for y-axis (e.g. HD or TIME),
    y_unit is unit name for y-axis (e.g. 'ft' or 'hrs')
    y_unit is a unit name for y-axis (e.g. 'ft' or 'hrs'),
    units_conversion is a multiplier to convert y_col units,
    ylims is y-axis limits to show on the graph,
    refs is a list of reference days to highlight in MM.DD format,
    form_colors_dict is a formation color dictionary,
    FORM, FORMTOPTIME, FORMBOTTIME, CONNTTIME, PIPESP, PIPEMV represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions, 
    save_folder is a folder to save computed dataframe,
    save is a flag to save data. 
    Function plots tripping performance graphs and returns None."""

    df = df_input.copy()
    df_trip = df_bound.copy()

    #plot
    n_cols = 4

    _, axes = plt.subplots(1, n_cols, figsize = (2.5*n_cols,10), sharey=True)

    #####################
    ###Formation Tops ###
    #####################
    ax = axes[0]

    #add formation boundaries
    for bnd_top, bnd_bot, form in df_form[[FORMTOPTIME, FORMBOTTIME, FORM]].values:
        
        #convert units
        bnd_top *= units_conversion
        bnd_bot *= units_conversion

        #get colors
        color = form_colors_dict[form]
        height = bnd_bot - bnd_top
        
        rect = patches.Rectangle((0, bnd_top), 1, height, linewidth=2,
                                edgecolor=color, facecolor=color, alpha=0.5)
        #add the patch to the Axes
        ax.add_patch(rect)

        #find formation middle point to show text
        ax.axhline(bnd_top, lw = 3, color = 'k')
        #add formation labels
        y_text = bnd_top + height/2 
        
        if y_text < ylims[1]:
            ax.text(0.05, y_text + 1, form[:15], fontsize = 12, fontweight = "bold")

    ax.set(xlabel = '', 
          ylabel = y_col.replace('_',' ').upper() + y_unit, 
          xlim = (0, 1),
          ylim = ylims)

    #add highlights
    df_R = select_day_highlight(df, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(df, refs[1], 'R2', units_conversion, y_col, ax))

    ax.set_title('TOP')
    ax.invert_yaxis()

    ########################
    ## PIPE SPEED (ft/hr) ##
    ########################
    ax = axes[1]

    xmax = df_trip[PIPESP].max()

    #worst boundary 
    ax.fill_betweenx(y=df_trip[y_col]*units_conversion, x1 = np.zeros(df_trip.shape[0]),
                    x2 = df_trip[PIPESP + '_worst'],
                    color = color_underperf, alpha=0.1, label=None)
    ax.plot(df_trip[PIPESP + '_worst'], df_trip[y_col]*units_conversion, 
            lw=1, color = color_underperf, label = 'WORST')

    #best boundary
    ax.fill_betweenx(y=df_trip[y_col]*units_conversion, x1 = df_trip[PIPESP + '_best'], 
                     x2 = np.ones(df_trip.shape[0]) * xmax,
                     color = color_overperf, alpha = 0.1, label=None)
    ax.plot(df_trip[PIPESP + '_best'], df_trip[TIME]*units_conversion, lw=1, color=color_overperf, label='BEST')

    #average
    ax.plot(df_trip[PIPESP + '_avg'], df_trip[y_col]*units_conversion, 
            lw = 1.5, color = color_historic, label = 'AVG')

    #current
    ax.plot(df_trip[PIPESP], df_trip[y_col]*units_conversion, 
            marker='o', markersize=5, lw=0, 
            color=color_RT, label = 'RT')

    ax.legend(loc='center')

    #xmin, xmax = ax.get_xlim()
    ax.set(xlabel = '',
           xlim = (0 , xmax),
           ylim = ylims)

    #add highlights
    df_R = select_day_highlight(df, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(df, refs[1], 'R2', units_conversion, y_col, ax))

    ax.set_title('PIPE SPD (ft/hr)')
    ax.invert_yaxis()

    #########################
    ## PIPE MOVEMENT (min) ##
    #########################

    ax = axes[2]

    xmax = df_trip[PIPEMV].quantile(0.95)

    #worst boundary
    ax.fill_betweenx(y=df_trip[y_col]*units_conversion, x1 = np.ones(df_trip.shape[0]) * xmax,
                    x2 = df_trip[PIPEMV + '_worst'],
                    color = color_underperf, alpha=0.1, label=None)
    ax.plot(df_trip[PIPEMV + '_worst'], df_trip[y_col]*units_conversion, 
            lw=1, color = color_underperf, label = 'WORST')

    #best boundary
    ax.fill_betweenx(y=df_trip[y_col]*units_conversion, x1 = np.zeros(df_trip.shape[0]), 
                     x2 = df_trip[PIPEMV + '_best'],
                     color = color_overperf, alpha = 0.1, label=None)
    ax.plot(df_trip[PIPEMV + '_best'], df_trip[y_col]*units_conversion, lw=1, 
            color=color_overperf, label='BEST')

    #average
    ax.plot(df_trip[PIPEMV + '_avg'], df_trip[y_col]*units_conversion, 
            lw = 1.5, color = color_historic, label = 'AVG')

    #current
    ax.plot(df_trip[PIPEMV], df_trip[y_col]*units_conversion, 
            marker='o', markersize=5, lw=0, 
            color=color_RT, label = 'RT')

    #xmin, xmax = ax.get_xlim()
    ax.set(xlabel = '',
           xlim = (0 , xmax),
           ylim = ylims)
    
    #add highlights
    df_R = select_day_highlight(df, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(df, refs[1], 'R2', units_conversion, y_col, ax))

    ax.set_title('PIPE MOVEMENT (min)')
    ax.invert_yaxis()

    ######################
    ## SLP to SLP (min) ##
    ######################
    ax = axes[3]
    
    #connection trip time
    for _, row in df_conn_trip_rt_kpi.iterrows():
    
        ax.plot(np.array([0, row[CONNTTIME]]), np.array([row[y_col]]*2) * units_conversion, 
                lw=4, color=kpi_dict[row[KPIC]])

    ax.set(xlabel = '', ylabel = '', 
           xlim = (0 , df_conn_trip_rt_kpi[CONNTTIME].quantile(1.00)),
           ylim = ylims)
    
    #add highlights
    df_R = select_day_highlight(df, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(df, refs[1], 'R2', units_conversion, y_col, ax))

    ax.set_title('SLP to SLP (min)')
    ax.invert_yaxis()

    plt.tight_layout()

    if save:
        plt.savefig(f'{save_folder}plot/trip_performance.png',dpi=150)

    plt.close()

    return None

#function compute trip activity data
def compute_trip_activity(df_input: pd.DataFrame,
                          compute_col: str, activities: list,
                          activity_dict = sub_activity_state_swap,
                          TIME = TIME, RSUBACT = RSUBACT, 
                          LABEL = LABEL, DUR = DUR,
                          round_ndigits = round_ndigits,
                          save_folder = save_folder, save = True) -> pd.DataFrame:
    """This function computes trip activity duration data.
    Here df_input is time-based drilling data,
    compute_col is a compute column name (e.g. CIRCTIME, WASHTIME, REAMTIME),
    activities is a list of subactivity names (e.g. ['WASH IN', 'WASH OUT']),
    activity_dict is a subactivity dictionary,
    TIME, RSUBACT, LABEL, DUR represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,
    round_ndigits = round_ndigits,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function returns computed trip activity duration dataframe."""

    cond = np.zeros(df_input.shape[0], dtype=bool)
    
    for col in activities:
        cond |= (df_input[RSUBACT] == activity_dict[col])

    df = df_input.loc[cond].groupby(LABEL)[[DATETIME, TIME, DUR]].min()

    #compute in duration in minutes
    df[compute_col] = df[DUR].dt.total_seconds()/60
    #drop duration column
    df.drop(columns=[DUR],inplace=True)

    #*** fix on Aug 14: add activity in case they are missing ***#
    cols = [DATETIME, TIME, compute_col]
    if df.empty:
        df = pd.DataFrame([[0]*len(cols)], columns = cols)

    if save:
        df.round(round_ndigits).to_csv(f'{save_folder}csv/{compute_col[:-5]}_time.csv', index = False)

    return df

#function plot trip activity graphs
def plot_trip_activity(df_input: pd.DataFrame,
                       df_circulate: pd.DataFrame,
                       df_wash: pd.DataFrame,
                       df_ream: pd.DataFrame,
                       y_col: str, y_unit: str, 
                       units_conversion: float, ylims: np.array,
                       refs: list,
                       CIRCTIME = CIRCTIME, WASHTIME = WASHTIME, REAMTIME = REAMTIME,
                       color_historic = color_historic,
                       color_underperf = color_underperf,
                       save_folder = save_folder, save = True) -> None:
    """This function plots computed trip activity duration.
    Here df_input is time-based drilling data,
    df_circulate is a circulate duration data,
    df_wash is a washing duration data,
    df_ream is a reaming duration data,
    y_col is column name for y-axis (e.g. HD or TIME),
    y_unit is unit name for y-axis (e.g. 'ft' or 'hrs')
    y_unit is a unit name for y-axis (e.g. 'ft' or 'hrs'),
    units_conversion is a multiplier to convert y_col units,
    ylims is y-axis limits to show on the graph,
    refs is a list of reference days to highlight in MM.DD format,
    CIRCTIME, WASHTIME, REAMTIME represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,
    color_historic is color for average hitoric performance,
    color_underperf is color for underperformance,
    save_folder is a folder to save computed dataframe,
    save is a flag to save data.
    Function plots trip activity graphs and returns None."""

    df = df_input.copy()

    n_cols = 3

    #plot tripping activities
    _, axes = plt.subplots(1, n_cols, figsize = (2.5*1.5*n_cols,10), sharey=True)

    ##################
    ## CIR (static) ##
    ##################

    ax = axes[0]

    for _, row in df_circulate.iterrows():
            
        ax.plot([0, row[CIRCTIME]], np.array([row[TIME]]*2) * units_conversion, 
                lw = 4, color = color_historic)

    ax.set(xlabel = '', ylabel = 'TIME (hr)', 
           xlim = (0 , df_circulate[CIRCTIME].max()*1.2),
           ylim = ylims)

    #add highlights
    df_R = select_day_highlight(df, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(df, refs[1], 'R2', units_conversion, y_col, ax))

    ax.invert_yaxis()
    ax.set_title('CIR STATIC (min)')

    ##################
    #### WASHING #####
    ##################

    ax = axes[1]

    for _, row in df_wash.iterrows():
            
        ax.plot([0, row[WASHTIME]], np.array([row[TIME]]*2) * units_conversion, 
                lw = 4, color = color_historic)

    ax.set(xlabel = '', ylabel = '', 
           xlim = (0 , df_wash[WASHTIME].max()*1.2),
           ylim = ylims)

    #add highlights
    df_R = select_day_highlight(df, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(df, refs[1], 'R2', units_conversion, y_col, ax))

    ax.invert_yaxis()
    ax.set_title('WASHING (min)')

    ##################
    #### REAMING #####
    ##################

    ax = axes[2]

    for _, row in df_ream.iterrows():
            
        ax.plot([0, row[REAMTIME]], np.array([row[TIME]]*2) * units_conversion, 
                lw = 4, color = color_underperf)

    ax.set(xlabel = '', ylabel = '', 
           xlim = (0 , df_ream[REAMTIME].max()*1.2),
           ylim = ylims)

    #add highlights
    df_R = select_day_highlight(df, refs[0], 'R1', units_conversion, y_col, ax)
    df_R = df_R.append(select_day_highlight(df, refs[1], 'R2', units_conversion, y_col, ax))

    ax.invert_yaxis()
    ax.set_title('REAMING (min)')

    plt.tight_layout()


    if save:
        plt.savefig(f'{save_folder}plot/trip_activity.png',dpi=150)
    
    plt.close()

    return None

#*************************************************************************************************************#
#*******************************  dashboard data and plots: Report generation  *******************************#
#*************************************************************************************************************#

#function calculates rig activity kpi values
def rig_activity_kpi(df_input: pd.DataFrame, 
                     df_trip_input: pd.DataFrame, 
                     df_conn_drill_input: pd.DataFrame, 
                     df_conn_trip_input: pd.DataFrame, 
                     WCR: pd.DataFrame,
                     hole_diameter: str, 
                     wells_select: list, 
                     current_well: int, 
                     current_well_name: str,
                     rop_unit='ft/h', trip_speed_unit = 'ft/h', wcr_unit='ft/day',
                     WELL = WELLID, AVG = AVG,
                     csv_folder = 'csv', plot_folder = 'plot',
                     save = True, save_folder = save_folder):
    """This function takes dataframes grouped by different rig activities"""

    df = df_input.copy()
    df_trip = df_trip_input.copy()
    df_conn_drill = df_conn_drill_input.copy()
    df_conn_trip = df_conn_trip_input.copy()

    #select hole size section
    if hole_diameter != 'all':
        df = df.loc[df[BS] == hole_diameter]
        df_trip = df_trip.loc[df_trip[BS] == hole_diameter]
        df_conn_drill = df_conn_drill.loc[df_conn_drill[BS] == hole_diameter]
        df_conn_trip = df_conn_trip.loc[df_conn_trip[BS] == hole_diameter]
         
    #select rotary drilling and sliding activities
    df_rotate = df.loc[(df[RSUBACT] == sub_activity_state_swap['ROTATE'])]
    df_slide = df.loc[(df[RSUBACT] == sub_activity_state_swap['SLIDE'])]

    AVG = "AVERAGE OFFSET WELLS"
    
    #calculate median values for different well combinations
    median_values = []
    for wells in [wells_select, [current_well]]:
        median_values.append([df_rotate.loc[df_rotate[WELL].isin(wells), ROP].median(), 
                              df_slide.loc[df_slide[WELL].isin(wells), ROP].median(),
                              df_trip.loc[df_trip[WELL].isin(wells), PIPESP].median(),
                              df_conn_drill.loc[df_conn_drill[WELL].isin(wells), CONNDTIME].median(), 
                              df_conn_trip.loc[df_conn_trip[WELL].isin(wells), CONNTTIME].median(),
                              WCR[WCRC].loc[WCR.index.isin(wells)].median()])

    kpi_offset = pd.DataFrame({'': ['DRILLING', 'DRILLING', 'TRIPPING', 'CONNECTION', 'CONNECTION', 'PERFORMANCE'],
                               'ACTIVITY': [f'ROTATE ROP ({rop_unit})', f'SLIDE ROP ({rop_unit})',
                                            f'TRIP SPEED ({trip_speed_unit})',
                                            'DRILL CONNECTION (mm:ss)','TRIP CONNECTION (mm:ss)',
                                            f'WELL CONSTRUCTION RATE ({wcr_unit})'],
                                AVG: median_values[0],
                                current_well_name: median_values[1]                               
                          }).fillna(0).set_index(['','ACTIVITY'])
    
    #add multiindexing to show section size
    section = 'SECTION ' + decimal_fraction_str(hole_diameter) if (hole_diameter != 'all') else ''
    bs = str(hole_diameter).replace('.', replace_dot)

    kpi_offset.columns = pd.MultiIndex.from_tuples([(section, AVG), (section, current_well_name)])
    
    format_dict = {(section, AVG): '{0:.1f}',
                   (section, current_well_name): '{0:.1f}'}
    #.format(format_dict)
     
    if save:
        #x = x.format(formatter="{:.2f}", subset=pd.IndexSlice[['rar', 'beta', 'ir'], :])
        pd.set_option('precision', 1)
        df_styled = kpi_offset.style.apply(lambda x: [("background: " + color_underperf) 
                                                   if (((v < x.mean())&(x.name[0] != 'CONNECTION'))|
                                                       ((v >= x.mean())&(x.name[0] == 'CONNECTION')))                                       
                                                   else ("background: "+ color_overperf) for v in x], axis = 1, 
                                                         subset=[(section, AVG), (section, current_well_name)]) \
                                    .format(convert_time_decimal_mmss, subset = pd.IndexSlice[[('CONNECTION', 'DRILL CONNECTION (mm:ss)'),
                                                                                               ('CONNECTION', 'TRIP CONNECTION (mm:ss)')], :])
        display(df_styled)

        kpi_offset.round(round_ndigits).to_csv(f'{save_folder}{csv_folder}/kpi_offset_bs_{bs}.csv',index=True)
        dfi.export(df_styled, f"{save_folder}{plot_folder}/kpi_offset_bs_{bs}.png")

    return kpi_offset

#function that plots histogram and summary table
def plot_conn_time_hist(df_conn_input: pd.DataFrame, df_historic: pd.DataFrame,
                        super_activity: str, 
                        add_legend = True, convert_BS = True,
                        GROUPWELL = GROUPWELL, WELLID = WELLID, BS = BS,
                        CONNTTIME = CONNTTIME, CONNDTIME = CONNDTIME,
                        color_RT = color_RT, color_historic = color_historic,
                        current_well = CURRENT_WELL_ID, 
                        current_well_name = CURRENT_WELL_NAME,
                        csv_folder = 'csv', plot_folder = 'plot', 
                        save_folder = SAVE_DAILY_REPORT_FOLDER, save = True) -> None:

    df_conn = df_conn_input.copy()

    #define group column dictionary
    group_well_dict = {}
    for key in df_historic[WELLID].unique():
        if key == current_well:
            group_well_dict[key] = current_well_name
        else:
            group_well_dict[key] = 'AVERAGE OFFSET WELLS'

    use_col = CONNTTIME if super_activity == 'TRIP' else CONNDTIME

    ## plot
    _, ax = plt.subplots(1,1, figsize=(9,4))

    if not df_conn.empty:

        df_conn[GROUPWELL] = df_conn[WELLID].map(group_well_dict)
        #put and shift two histogramms manually
        hist_data = [df_conn.loc[(df_conn[WELLID] == current_well), use_col].values, 
                     df_conn.loc[(df_conn[WELLID] != current_well), use_col].values]

        if (len(hist_data[0]) > 0) | (len(hist_data[1]) > 0):
            
            hist_min = max([min(np.concatenate((hist_data[0],hist_data[1]))) - 1, 0])

            bins_offset = np.linspace(hist_min-0.25, 20.25-0.25, num=20)
            bins_rt = np.linspace(hist_min+0.25, 20.25+0.25, num=20)

            ax.hist(hist_data[1], bins=bins_offset, label='AVERAGE OFFSET WELLS', color=color_historic, alpha=1, rwidth=0.5)
            ax.hist(hist_data[0], bins=bins_rt, label=current_well_name, color=color_RT, alpha=1, rwidth=0.5)
            ax.legend(bbox_to_anchor=(1.08, 1)) #(loc='upper right')

            if not add_legend:
                ax.get_legend().remove()

        #change bitsize format
        if convert_BS:
            df_conn[BS] = df_conn[BS].apply(lambda x: decimal_fraction_str(x))
        #make table
        df_conn_summary = df_conn.groupby([GROUPWELL, BS])[use_col].agg(['median','count'])\
                                .sort_index(level=1, ascending=False)


        df_conn_summary.round(round_ndigits).to_csv(f'{save_folder}csv/conn_{super_activity.lower()}_time_dist_bs_current_offset.csv', index=True)
        df_conn_summary['median'] = df_conn_summary['median'].apply(lambda x: convert_time_decimal_mmss(x))
        df_conn_summary.rename(columns = {'median': f'Avg Connection Time {super_activity.title()} (mm:ss)', 'count': 'Count'}, inplace = True)
        df_conn_summary.index.names = ['Well Name', 'Hole Diameter']

    else:
        #create an empty dataframe with proper structure
        df_conn_summary = pd.DataFrame(columns = ['Well Name', 'Hole Diameter', 
                                                  f'Avg Connection Time {super_activity.title()} (mm:ss)', 
                                                  'Count']).set_index(['Well Name', 'Hole Diameter'])

    #original solution: behaves really strange
    # bins = np.linspace(0, 20, 5)
    # ax.hist([df_conn.loc[(df_conn[WELLID] == current_well), use_col].values, 
    #         df_conn.loc[(df_conn[WELLID] != current_well), use_col].values], 
    #         bins, label = [current_well_name, 'AVERAGE OFFSET WELLS'],
    #         color = [color_RT, color_historic],
    #         density = True)
    
    plt.xlabel(f'CONNECTION TIME {super_activity} (min)')
    plt.ylabel('COUNT')
    #plt.title(use_col.replace('_',' ').upper())
    plt.tight_layout()
    
    #save data
    if save:
        plt.savefig(f'{save_folder}{plot_folder}/conn_{super_activity.lower()}_time_dist.png', dpi=150)

        display(df_conn_summary)
        dfi.export(df_conn_summary, f"{save_folder}{plot_folder}/conn_{super_activity.lower()}_time_dist_bs_current_offset.png")

    plt.close()

    return None

#function to plot connection time versus datetime
def plot_conn_time_datetime(df_conn_kpi: pd.DataFrame,
                            super_activity: str,
                            DATETIME = DATETIME, KPIC = KPIC,
                            CONNTTIME = CONNTTIME, CONNDTIME = CONNDTIME,
                            kpi_dict = kpi_dict,
                            save_folder = SAVE_DAILY_REPORT_FOLDER, save = True) -> None:

    use_col = CONNTTIME if super_activity == 'TRIP' else CONNDTIME
    #plot
    _, ax = plt.subplots(1, 1, figsize = (max(0.42*df_conn_kpi.shape[0],3), 7))

    #add legend
    label_dict = {'BEST': color_overperf, 'HISTORIC AVG': color_historic, 'WORST': color_underperf}
    for key, value in label_dict.items(): #Loop over color dictionary
        ax.bar(df_conn_kpi[DATETIME], df_conn_kpi[use_col],width=0,color=value,label=key) #Plot invisible bar graph but have the legends specified

    ax.legend(bbox_to_anchor=(1.08, 1))

    df_conn_kpi.set_index(DATETIME)[use_col].plot.bar(ax=ax, 
                color=[kpi_dict[val] for val in df_conn_kpi[KPIC].values], label = None)

    ax.axhline(df_conn_kpi[use_col + '_best'].max(),lw=2, color=kpi_dict[0])
    ax.axhline(df_conn_kpi[use_col + '_worst'].min(),lw=2, color=kpi_dict[2])

    # ax2 = ax.twinx()
    # #add worst, best lines
    # df_conn_kpi.set_index(DATETIME)[use_col + '_best'].plot(ax=ax2, lw=3, color=color_overperf)
    # df_conn_kpi.set_index(DATETIME)[use_col + '_worst'].plot(ax=ax2, lw=3, color=color_underperf)
    # ax2.set_ylim(ax.get_ylim())

    ax.set(xlabel='\nDATE', 
           ylabel=use_col.upper().replace('_', ' ') + ' (min)',
           ylim = (0, 20))
    
    plt.tight_layout()
    

    if save:
        plt.savefig(f"{save_folder}plot/conn_{super_activity.lower()}_time_current_well.png")

    plt.close()

    return None

#function to plot connection time versus datetime
def plot_conn_time_datetime_nokpi(df_conn_input: pd.DataFrame,
                                  super_activity: str,
                                  table_size = 20, color_RT = color_RT,
                                  DATETIME = DATETIME, CONNTTIME = CONNTTIME, CONNDTIME = CONNDTIME,
                                  csv_folder = 'csv', plot_folder = 'plot', 
                                  save_folder = SAVE_DAILY_REPORT_FOLDER, save = True) -> None:

    df_conn = df_conn_input.copy()
    #reformat
    if not(df_conn.empty|(df_conn[DATETIME] == 0).all()):
        df_conn[DATETIME] = df_conn[DATETIME].dt.strftime('%H:%M:%S') #('%d/%m/%y)

    use_col = CONNTTIME if super_activity == 'TRIP' else CONNDTIME
    df_conn = df_conn.set_index(DATETIME)[[use_col]]

    #plot
    show_row_max = table_size*3

    for i in range(max([int(np.ceil(df_conn.shape[0]/ show_row_max)),1])):

        df_conn_ = df_conn.iloc[i*show_row_max:(i+1)*show_row_max] 

        _, ax = plt.subplots(1, 1, figsize = (max(0.44*df_conn_.shape[0],3), 5))#0.44*,5
        
        ymax = 0
        if not(df_conn.empty):
            df_conn_.plot.bar(ax=ax, color=color_RT)
            ymax = min([20, 1.02*max(df_conn[use_col])])
            ax.get_legend().remove()

        #(dd/mm/yy) 
        ax.set(xlabel="\nDATE (hh:mm:ss)",  
            ylabel=use_col.upper().replace('_', ' ') + ' (min)',
            ylim = (0, ymax))
        
        plt.tight_layout()

        if save:
            plt.savefig(f"{save_folder}{plot_folder}/conn_{super_activity.lower()}_time_nokpi_current_well_{i}.png",dpi=150)

        plt.close()
        
    if save:
        #save table
        df_conn.index.names = ['Connection Start Time']
        df_conn[use_col] = df_conn[use_col].apply(lambda x: convert_time_decimal_mmss(x))
        df_conn.columns = [use_col.replace('_',' ').title() + ' (mm:ss)']

        display(df_conn)
        #split images by table_size lines
        if not df_conn.empty:
            df_conn = df_conn.loc[df_conn.index != 0]
        df_conn.round(round_ndigits).to_csv(f'{save_folder}{csv_folder}/conn_{super_activity.lower()}_details_current_well.csv', index=True)

        df_conn_size = df_conn.shape[0]
        number_plots = max([int(np.ceil(df_conn_size/table_size)), 1])
        for i in range(number_plots):
            i_start = i * table_size
            i_end = min([i_start + table_size, df_conn_size])
            dfi.export(df_conn.iloc[i_start:i_end], f"{save_folder}{plot_folder}/conn_{super_activity.lower()}_details_current_well_{i}.png")

    return None

#function to highlight rig activity in a table
def highlight_rig_act(df_input: pd.DataFrame, c: str, 
                        add_diff_perform_color = True,
                        act_backgr_dict = act_backgr_dict,
                        color_underperf = color_underperf,
                        color_overperf = color_overperf) -> pd.DataFrame:
    #copy df to new - original data are not changed

    df = df_input.copy()
    #select all values to default value - red color
    df.loc[:,:] = 'background-color: None'

    #display(df_input[(c, '', 'ITEM')].map(act_backgr_dict))

    #overwrite values grey color    
    df[[(c, '', 'ITEM')]] = df_input[(c, '', 'ITEM')].map(act_backgr_dict)

    if add_diff_perform_color:
        df[[(c, DIFF, '(hh:mm)')]] = df_input[(c, DIFF, '(hh:mm)')].apply(lambda x: ("background: " + color_underperf) if x > 0
                                                                               else ("background: " + color_overperf))

    return df

#function to generate and save rig activity table
def generate_rig_activity_time_dist_table(total_time_sub_act,
                                          super_activity, hole_diameter,
                                          AVG = AVG, DIFF = DIFF,
                                          current_well = CURRENT_WELL_ID,
                                          current_well_name = CURRENT_WELL_NAME, 
                                          replace_dot = replace_dot, round_ndigits = round_ndigits,
                                          save_folder = SAVE_DAILY_REPORT_FOLDER, save = True):
    #save table
    drill_trip_act = total_time_sub_act.T[[current_well, AVG]]\
                                       .rename(columns = {current_well: current_well_name, 
                                                          AVG: "AVERAGE OFFSET WELLS"})
    AVG = "AVERAGE OFFSET WELLS"

    #add pct
    #current well
    drill_trip_act[current_well_name + ' (%)'] = drill_trip_act[current_well_name] \
                                                / drill_trip_act[current_well_name].sum() * 100
    
    drill_trip_act[AVG + ' (%)'] = drill_trip_act[AVG] / drill_trip_act[AVG].sum() * 100

    #all total
    drill_trip_act = drill_trip_act.append(pd.DataFrame(drill_trip_act.sum(axis=0))\
                                                .rename(columns = {0: 'TOTAL'}).T)

    #add difference in hours
    drill_trip_act[DIFF] = drill_trip_act[current_well_name] - drill_trip_act[AVG]


    #rename columns and reorder
    drill_trip_act.rename(columns = {current_well_name: current_well_name + ' (hh:mm)',
                                        AVG: AVG + ' (hh:mm)', 
                                        AVG + ' (%)': AVG + ' (%)'}, inplace=True)

    drill_trip_act.index.name = 'ITEM'


    #color columns
    drill_trip_act = drill_trip_act[[current_well_name + ' (hh:mm)', current_well_name + ' (%)',
                                        AVG + ' (hh:mm)', AVG + ' (%)', DIFF]]

    #display summary table
    drill_trip_act.reset_index(inplace = True)

    #fill missing values with zero
    drill_trip_act = drill_trip_act.fillna(0)

    #add multiindex for displaying purposes
    title = super_activity.upper() if (super_activity != 'all') else 'DRILL & TRIP'
    title_add = ': SECTION ' + decimal_fraction_str(hole_diameter) if (hole_diameter != 'all') else ''
    c = title + ' ACTIVITIES' + title_add

    #set multiindex
    drill_trip_act.columns = pd.MultiIndex.from_tuples([(c, '', 'ITEM'),
                                                        (c, current_well_name, '(hh:mm)'), (c, current_well_name, '(%)'),
                                                        (c, AVG,'(hh:mm)'), (c, AVG, '(%)'), (c , DIFF, '(hh:mm)')])

    #sort values
    drill_trip_act.sort_values(by=[(c, current_well_name, '(%)')], ascending=False, inplace=True)

    #define format for table image
    format_dict = {(c, current_well_name, '(hh:mm)'): convert_time_decimal_hhmm, 
                (c, current_well_name, '(%)'): '{0:.1f}%',
                (c, AVG, '(hh:mm)'): convert_time_decimal_hhmm, 
                (c, AVG, '(%)'): '{0:.1f}%',
                (c, DIFF, '(hh:mm)'): convert_time_decimal_hhmm}

    df_styled = drill_trip_act.style.format(format_dict).apply(highlight_rig_act, c=c, axis=None).hide_index()
    display(df_styled)

    if save:
        bs = str(hole_diameter).replace('.', replace_dot)

        drill_trip_act.round(round_ndigits).to_csv(f'{save_folder}csv/time_dist_act_{super_activity.lower()}_bs_{bs}.csv',index=True)
        dfi.export(df_styled, f"{save_folder}plot/time_dist_act_{super_activity.lower()}_bs_{bs}_table.png")

    return None

#function to generate and save rig activity table
def generate_rig_activity_time_dist_table_rt(total_time_sub_act,
                                             super_activity, hole_diameter,
                                             add_diff_perform_color = False,
                                             current_well = CURRENT_WELL_ID,
                                             current_well_name = CURRENT_WELL_NAME, 
                                             replace_dot = replace_dot, round_ndigits = round_ndigits,
                                             csv_folder = 'csv', plot_folder = 'plot',
                                             save_folder = SAVE_DAILY_REPORT_FOLDER, save = True):
    #save table
    drill_trip_act = total_time_sub_act.T[[current_well]].rename(columns = {current_well: current_well_name})

    #add pct
    #current well
    drill_trip_act[current_well_name + ' (%)'] = drill_trip_act[current_well_name] \
                                                / drill_trip_act[current_well_name].sum() * 100

    #all total
    drill_trip_act = drill_trip_act.append(pd.DataFrame(drill_trip_act.sum(axis=0))\
                                                .rename(columns = {0: 'TOTAL'}).T)

    #convert hrs to hh:mm
    drill_trip_act[current_well_name] = drill_trip_act[current_well_name].apply(lambda x: convert_time_decimal_hhmm(x))

    #rename columns and reorder
    drill_trip_act.rename(columns = {current_well_name: current_well_name + ' (hh:mm)'}, inplace=True)

    drill_trip_act.index.name = 'ITEM'


    #color columns
    drill_trip_act = drill_trip_act[[current_well_name + ' (hh:mm)', current_well_name + ' (%)']]


    #display summary table
    drill_trip_act.reset_index(inplace = True)

    #fill missing values with zero
    drill_trip_act = drill_trip_act.fillna(0)

    #add multiindex for displaying purposes
    title = super_activity.upper() if (super_activity != 'all') else 'DRILL & TRIP'
    title_add = ': SECTION ' + decimal_fraction_str(hole_diameter) if (hole_diameter != 'all') else ''
    c = title + ' ACTIVITIES' + title_add

    #set multiindex
    drill_trip_act.columns = pd.MultiIndex.from_tuples([(c, '', 'ITEM'),
                                                        (c, current_well_name, '(hh:mm)'), 
                                                        (c, current_well_name, '(%)')])

    #sort values
    drill_trip_act.sort_values(by=[(c, current_well_name, '(%)')], ascending=False, inplace=True)

    #define format for table image
    format_dict = {(c, current_well_name, '(%)'): '{0:.1f}%'}#(c, current_well_name, '(hrs)'): '{:,.3f}'

    df_styled = drill_trip_act.style.format(format_dict).apply(highlight_rig_act, c=c, 
                                     add_diff_perform_color=add_diff_perform_color, axis=None).hide_index()
    display(df_styled)

    if save:
        bs = str(hole_diameter).replace('.', replace_dot)

        drill_trip_act.round(round_ndigits).to_csv(f'{save_folder}{csv_folder}/time_dist_act_{super_activity.lower()}_bs_{bs}_rt.csv',index=False)
        dfi.export(df_styled, f"{save_folder}{plot_folder}/time_dist_act_{super_activity.lower()}_bs_{bs}_table_rt.png")

    return None

### rewrite conn_time_ilt_analysis and conn_time_overperf_analysis to one conn_time_analysis
#function to calculate and generate ILT tabel for connection time
def conn_time_ilt_analysis(df_conn_kpi: pd.DataFrame, 
                           super_activity: str, act_col: str,
                           DATETIME = DATETIME, BS = BS,
                           save_folder = SAVE_DAILY_REPORT_FOLDER, save = True) -> pd.DataFrame:

    df_ilt = df_conn_kpi.loc[df_conn_kpi[KPIC] == 2, [DATETIME, BS, act_col, act_col + '_avg']].set_index(DATETIME)
    #df_ilt.index = np.arange(1, df_ilt.shape[0] + 1)
    df_ilt.index.names = ['CONNECTION START TIME']
    df_ilt['IMPROVEMENT POTENTIAL (mm:ss)'] = df_ilt[act_col] - df_ilt[act_col + '_avg']
    #add total
    total = df_ilt.sum(numeric_only=True)
    total.name = 'TOTAL'
    df_ilt = df_ilt.append(total)

    #convert bit size
    df_ilt[BS] = df_ilt[BS].apply(lambda x: decimal_fraction_str(x))
    df_ilt.loc['TOTAL', BS] = np.nan

    #rename for report
    act_col_upper = act_col.replace('_',' ').upper() + ' (mm:ss)'
    df_ilt.rename(columns={act_col: act_col_upper,
                           act_col + '_avg': 'HISTORIC AVERAGE (mm:ss)',
                           BS: 'HOLE DIAMETER'}, inplace = True)
    if save:

        if df_ilt.shape[0] > 10:
            df_ilt_part = df_ilt.iloc[:10].append(df_ilt.iloc[-1])
        else:
            df_ilt_part = df_ilt

        df_style = df_ilt_part\
                         .style.applymap(lambda x: ('color: ' + color_underperf), subset=[act_col_upper])\
                         .applymap(lambda x: ('color: ' + '#F9A602'), subset=['HISTORIC AVERAGE (mm:ss)'])\
                         .set_caption("Underperformance analysis".upper())\
                         .format(formatter=None, na_rep = '', subset = ['HOLE DIAMETER'])\
                         .format(convert_time_decimal_mmss, subset = df_ilt.columns.tolist()[1:])
        display(df_style)
        dfi.export(df_style, f"{save_folder}plot/conn_{super_activity.lower()}_ilt.png")
        
    return df_ilt

#function to calculate and generate ILT tabel for connection time
def conn_time_overperf_analysis(df_conn_kpi: pd.DataFrame, 
                                super_activity: str, act_col: str,
                                save_folder = SAVE_DAILY_REPORT_FOLDER, save = True) -> pd.DataFrame:

    df_ilt = df_conn_kpi.loc[df_conn_kpi[KPIC] == 0, [DATETIME, BS, act_col, act_col + '_avg']].set_index(DATETIME)
    #df_ilt.index = np.arange(1, df_ilt.shape[0] + 1)
    df_ilt.index.names = ['CONNECTION START TIME']
    df_ilt['ACTUAL IMPROVEMENT (mm:ss)'] = -(df_ilt[act_col] - df_ilt[act_col + '_avg'])
    #add total
    total = df_ilt.sum(numeric_only=True)
    total.name = 'TOTAL'
    df_ilt = df_ilt.append(total)

    #convert bit size
    df_ilt[BS] = df_ilt[BS].apply(lambda x: decimal_fraction_str(x))
    df_ilt.loc['TOTAL', BS] = np.nan

    #rename for report
    act_col_upper = act_col.replace('_',' ').upper() + ' (mm:ss)'
    df_ilt.rename(columns={act_col: act_col_upper,
                           act_col + '_avg': 'HISTORIC AVERAGE (mm:ss)',
                           BS: 'HOLE DIAMETER'}, inplace = True)
    if save:

        if df_ilt.shape[0] > 10:
            df_ilt_part = df_ilt.iloc[:10].append(df_ilt.iloc[-1])
        else:
            df_ilt_part = df_ilt

        df_style = df_ilt_part\
                         .style.applymap(lambda x: ('color: ' + color_overperf), subset=[act_col_upper])\
                         .applymap(lambda x: ('color: ' + '#F9A602'), subset=['HISTORIC AVERAGE (mm:ss)'])\
                         .set_caption("Overperformance analysis".upper())\
                         .format(formatter=None, na_rep = '', subset = ['HOLE DIAMETER'])\
                         .format(convert_time_decimal_mmss, subset = df_ilt.columns.tolist()[1:])
        display(df_style)
        dfi.export(df_style, f"{save_folder}plot/conn_{super_activity.lower()}_overperf.png")
        
    return df_ilt

#execute generate_pdf_report.py
#subprocess.call("test1.py", shell=True)
#######################

def plot_well_trajectory(df_survey_input: pd.DataFrame,
                         xlims = None,
                         x_col = VS, y_col = TVD,
                         DLSR = DLSR, INCR = INCR,
                         depth_unit = 'ft', dls_unit = 'deg/100ft', 
                         inc_unit = 'deg',
                         well_name = CURRENT_WELL_NAME,
                         save_folder = SAVE_DAILY_REPORT_FOLDER, save = True):
    
    df_survey_rt = df_survey_input.copy()
    #add DLS category:
    df_survey_rt[DLSR] = pd.cut(df_survey_rt[DLS],[0,3,5,8,100],labels=['0-3','3-5','5-8','>8'])
    df_survey_rt[INCR] = pd.cut(df_survey_rt[INC], np.arange(0,105,20),labels=['0-20','20-40','40-60','60-80', '80-100'])

    palette_dls = ['#97FF8F','#FFF500','#F5AB24','#CE1140']
    dls_dict = dict(zip(['0-3','3-5','5-8','>8'], palette_dls))

    #plot
    fig, axes = plt.subplots(1,2,figsize=(12,6))

    ax = axes[0]
    sns.scatterplot(x = x_col, y = y_col, hue = DLSR, ax=ax,
                    data = df_survey_rt, palette = palette_dls)
    ax.invert_yaxis()
    ax.legend(title = f"DLS ranges ({dls_unit})")
    ax.set(xlabel = x_col.replace('vs', 'vertical section').upper() + f' ({depth_unit})', 
           ylabel = y_col.upper() + f' ({depth_unit})')
    if (xlims != None):
        ax.set_xlim(xlims)

    ax = axes[1]

    palette_inc = ['#97FF8F','#FFF500','#F5AB24','#ce5911', '#CE1140']
    inc_dict = dict(zip(['0-20','20-40','40-60','60-80','80-100'], palette_inc))


    sns.scatterplot(x = x_col, y = y_col, hue = INCR, ax = ax,
                    data = df_survey_rt, palette = palette_inc)
    ax.invert_yaxis()
    ax.legend(title = f'Inclination ranges ({inc_unit})')
    ax.set(xlabel = x_col.replace('vs', 'vertical section').upper() + f' ({depth_unit})', 
          ylabel = y_col.upper() + f' ({depth_unit})')
    if (xlims != None):
        ax.set_xlim(xlims)

    plt.suptitle(well_name, fontsize=14, y=0.97)

    plt.tight_layout()

    if save:
        plt.savefig(f'{save_folder}plot/current_well_profile_dls_inc.png',dpi=150)
    
    plt.close()
    
    return None

def full_rig_ops_summary(round_ndigits = round_ndigits,
                         save_folder = SAVE_DAILY_REPORT_FOLDER, save = True):

    df_assumptions = pd.DataFrame()

    df_assumptions['Service Daily Rate'] = ['Rig Rate',
                                            'Directional Service',
                                            'Mud Logging',
                                            'Mud',
                                            'Others',
                                            'Consumables']

    df_assumptions['1,000 USD/Day'] = [25,10,4,3.5,4,3.5]

    df_assumptions['%'] = [50,20,8,7,8,7]

    #Assumptions table
    #total daily cost
    total_daily = round(df_assumptions.iloc[0,1] / (df_assumptions.iloc[0,2]/100),2)
    #total hourly cost
    total_hrs = total_daily/24

    df_total = pd.DataFrame({'Service Rate': ['Total daily rate','Total hourly rate'],
                            '1,000 USD': [total_daily, total_hrs]})

    fig, ax = plt.subplots(1,1, figsize = (8,8))
    #plot assumptions summary
    df_assumptions.set_index('Service Daily Rate')['%'].plot.pie(figsize=(11, 6),autopct='%1.1f%%',fontsize=22, ax = ax,
                                                                    colors=['#FCE255','#C7B037','#F57524','#9CCC66','#80DDFF','#167A9D'])
    ax.set_ylabel('')

    try:
        ax.get_legend().remove()
    except:
        print('no legend')
        
    plt.tight_layout()

    if save:
        #save plot
        plt.savefig(f'{save_folder}plot/rig_services_dist.png',dpi=150)

        #save data
        df_assumptions.to_csv(f'{save_folder}csv/rig_services_dist_daily.csv',index = False)
        df_total.round(round_ndigits).to_csv(f'{save_folder}csv/rig_services_total.csv',index = False)

        #save data image for assumptions
        df_styled = df_assumptions.style.hide_index()
        display(df_styled)
        dfi.export(df_styled, f'{save_folder}plot/rig_services_dist_daily.png')

        #save data image for total estimate
        df_styled = df_total.round(round_ndigits).style.hide_index()
        display(df_styled)
        dfi.export(df_styled, f'{save_folder}plot/rig_services_total.png')
    
    plt.close()

    return total_hrs

### adjust style 
#calulate length intervals for current well
def find_act_depth(df_, activity, activity_dict, activity_col, depth_col, datetime_col):
    
    """activity = Rotary drilling, Sliding"""
    
    df = df_.loc[df_[activity_col] == activity_dict[activity]]
    df[depth_col + '_diff'] = df[depth_col].diff()
    df[datetime_col + '_diff'] = df[datetime_col].diff()
    time_step = df[datetime_col + '_diff'].value_counts().index[0]

    return df.loc[df[datetime_col + '_diff'] == time_step, depth_col + '_diff'].sum()

def find_act_time(df_, activity, activity_dict, activity_col, depth_col, datetime_col):
    
    """activity = Rotary drilling, Sliding"""
    
    df = df_.loc[df_[activity_col] == activity_dict[activity]]
    df[depth_col + '_diff'] = df[depth_col].diff()
    time_step = df[depth_col + '_diff'].value_counts().index[0]
    #df[datetime_col + '_diff'] = df[datetime_col].diff()
    #time_step = df[datetime_col + '_diff'].value_counts().index[0]

    return df[depth_col].nunique() * time_step

#generate executive summary table and plot
def executive_summary(df_rt,  WCR_summary, df_kpi, 
                      hole_diameter, total_hrs,
                      stand_length = STAND_LENGTH,
                      depth_unit = 'ft',
                      use_conn_table = False, 
                      number_connections = [None, None],
                      duration_connections = [None, None],
                      replace_dot = replace_dot, round_ndigits = round_ndigits,
                      save_folder = SAVE_DAILY_REPORT_FOLDER, save = True):

    s = f"SECTION {decimal_fraction_str(hole_diameter)}" if (hole_diameter != 'all') else ''

    bs = str(hole_diameter).replace('.', replace_dot)
    AVGOFF = 'AVERAGE OFFSET WELLS'

    _df = df_rt.copy()
    kpi = df_kpi.copy().drop('PERFORMANCE', axis = 0)

    #activity length
    if hole_diameter != 'all':
        _df = _df[_df[BS] == hole_diameter]
        
    length = []
    time = []
    for activity in ['ROTATE', 'SLIDE']:
        try:
            lg = round(find_act_depth(_df, activity = activity, activity_dict = sub_activity_state_swap, 
                          activity_col = RSUBACT, depth_col = HD, datetime_col = DATETIME))
        except:
            lg = 0 
        try:
             tm = (find_act_time(_df, activity = activity, activity_dict = sub_activity_state_swap, 
                          activity_col = RSUBACT, depth_col = TIME, datetime_col = DATETIME))
        except:
            tm = 0

        length.append(lg)
        time.append(tm)

    length_trip=[]
    time_trip=[]
    for activity in ['TRIP IN', 'TRIP OUT']:
        try :
            lg = round(find_act_depth(_df, activity = activity, activity_dict = sub_activity_state_swap, 
                                      activity_col = RSUBACT, depth_col = BD, datetime_col = DATETIME))
        except:
            lg=0
        try :
            tm = round(find_act_time(_df, activity = activity, activity_dict = sub_activity_state_swap, 
                                      activity_col = RSUBACT, depth_col = TIME, datetime_col = DATETIME))
        except:
            tm=0
        length_trip.append(lg)
        time_trip.append(tm)

    #number of conections
    length_trip_total = sum([abs(x) for x in length_trip])
    length_drill_total = sum([abs(x) for x in length])

    if use_conn_table:
        length_all = length + [length_trip_total] + number_connections
        time_all = time + [sum([abs(x) for x in time_trip])] + duration_connections
        #print(time_all)
    else:
        length_all = length + [length_trip_total] + [np.ceil(length_drill_total/stand_length), np.ceil(length_trip_total/stand_length)]
        time_all = time + [sum([abs(x) for x in time_trip])] + \
                   [kpi.loc[('CONNECTION','DRILL CONNECTION (mm:ss)'),(s,CURRENT_WELL_NAME)]*length_all[-2],
                    kpi.loc[('CONNECTION','TRIP CONNECTION (mm:ss)'),(s,CURRENT_WELL_NAME)]*length_all[-1]]
        #print('time_all, length_drill_total, length_trip_total', time_all, length_all[-2], length_all[-1])

    # #number of conections
    # #num_conns = round(sum(length)/stand_length)
    # length_trip_total = sum([abs(x) for x in length_trip])

    # length_all = length + [length_trip_total] + [round(sum(length)/stand_length), round(length_trip_total/stand_length)]
    # time_all = time + [sum([abs(x) for x in time_trip])] + [0,0]

    LENGTH = f'LENGTH ({depth_unit}/#)'
    kpi[LENGTH] = length_all
    TIMEHR = 'TIME (hh:mm)'
    SAVEHR = 'SAVINGS (hh:mm)'
    
    time_conversion = 60 #min to hours
    kpi[TIMEHR] = np.array(time_all)/time_conversion

    kpi['SAVINGS (%)'] = (kpi[(s,CURRENT_WELL_NAME)] - kpi[(s, AVGOFF)])/kpi[(s, AVGOFF)]*100

    kpi[SAVEHR] = kpi[LENGTH]/kpi[(s, AVGOFF)] - kpi[LENGTH]/kpi[(s,CURRENT_WELL_NAME)]
    
    #Different calculations for conection
    cond = kpi.index.isin(['CONNECTION'], level=0)
    kpi.loc[cond, 'SAVINGS (%)'] *= (-1)
    kpi.loc[cond, SAVEHR] = (kpi.loc[cond, (s, AVGOFF)]*kpi.loc[cond, LENGTH]
                                      - kpi.loc[cond, (s, CURRENT_WELL_NAME)]*kpi.loc[cond, LENGTH])/time_conversion

    kpi['SAVINGS (1,000 USD)'] = kpi[SAVEHR] * total_hrs

    #fill missing values with 0
    kpi.fillna(0, inplace=True)

    #ADD ILT
    kpi['ILT'] = (kpi['SAVINGS (%)'] >= 0).fillna(0).map({True:'NO', False:'YES'})

    #add total savings
    t = pd.DataFrame({'AREA': 'TOTAL', 'ACTIVITY':'OVERALL',
                      AVGOFF: '', CURRENT_WELL_NAME:''},index=[0]).set_index(['AREA', 'ACTIVITY'])

    #add multiindexing
    t.columns = pd.MultiIndex.from_tuples([(s, AVGOFF),(s, CURRENT_WELL_NAME)])
    t[LENGTH] = np.nan
    t[TIMEHR] = np.nan
    
    #add percentage with respect to total average time
    #average drilling time for all wells in days
    dur = (WCR_summary.loc[TIMEDAY,'avg'].round(2) * 24)
    
    if dur > 0.001:
        t['SAVINGS (%)'] = kpi[SAVEHR].sum() \
                            / dur*100
    else:
        t['SAVINGS (%)'] = 0

    t[SAVEHR] = kpi[SAVEHR].sum()

    t['SAVINGS (1,000 USD)'] = kpi['SAVINGS (1,000 USD)'].sum()
    t['ILT'] = ''

    #add total to kpi table
    kpi = kpi.append(t)
    
    kpi = kpi.replace([np.inf, -np.inf], 0)
    
    #calculate total connection time in hours
    # for n in ['DRILL CONNECTION (min/conn)', 'TRIP CONNECTION (min/conn)']:
    #     kpi.loc[('CONNECTION', n),TIMEHR] = kpi.loc[('CONNECTION',n),(s,CURRENT_WELL_NAME)] \
    #                                               * kpi.loc[('CONNECTION',n),'LENGTH'].values / time_conversion

    fig, axes = plt.subplots(1, 2, figsize=(15,8))
    
    ax = axes[0]
    kpi['SAVINGS (%)'].plot(kind = 'bar',
                            color=kpi['SAVINGS (%)'].apply(lambda x: color_underperf if x < 0 else color_overperf).values,
                            ax = ax)
    ax.set_ylabel('SAVINGS (%)')
    ax.set_xlabel('RIG ACTIVITY')
    ax.set_xticklabels([v[-1] for v in kpi.index] ,rotation=90)
    #ax.set_title('Executive summary ' + s)
    
    ax = axes[1]
    
    kpi[SAVEHR].plot(kind = 'bar',
                            color=kpi['SAVINGS (%)'].apply(lambda x: color_underperf if x < 0 else color_overperf).values,
                            ax = ax)
    ax.set_ylabel(SAVEHR)
    ax.set_xlabel('RIG ACTIVITY')
    ax.set_xticklabels([v[-1] for v in kpi.index] ,rotation=90)
    #plt.suptitle('Executive summary ' + s, y=0.99, fontsize=16)

    plt.tight_layout()
    
    if save:
        plt.savefig(f'{save_folder}plot/kpi_executive_summary_bs_{bs}.png', dpi=150)

        #display(kpi.info())
        #kpi_style_dict = {k:'{:,.1f}' for k in kpi.select_dtypes(include=np.number).columns.tolist()}
        pd.set_option('precision', 1)

        df_styled = kpi.style.apply(lambda x: [("background: " + color_underperf)
                                if (x.iloc[0] < 0)
                                else ("background: " + color_overperf) for v in x], axis = 1, 
                                subset=['SAVINGS (%)','ILT']) \
                             .format(formatter=None, na_rep = '')\
                             .format(convert_time_decimal_mmss, subset = pd.IndexSlice[[('CONNECTION', 'DRILL CONNECTION (mm:ss)'),
                                                                                        ('CONNECTION', 'TRIP CONNECTION (mm:ss)')], 
                                                                                        [(s, AVGOFF), (s, CURRENT_WELL_NAME)]])\
                             .format(convert_time_decimal_hhmm, na_rep = '', subset = [(TIMEHR,''), (SAVEHR,'')])\
                             .format(int, na_rep = '', subset = pd.IndexSlice[[('CONNECTION', 'DRILL CONNECTION (mm:ss)'),
                                                                               ('CONNECTION', 'TRIP CONNECTION (mm:ss)')], 
                                                                               [LENGTH]])
    
    display(df_styled)
    
    #save table
    dfi.export(df_styled, f"{save_folder}plot/kpi_executive_summary_table_bs_{bs}.png")
    kpi.round(round_ndigits).to_csv(f"{save_folder}csv/kpi_executive_summary_table_bs_{bs}.csv", index = True)

    plt.close()

    return kpi

#generate executive summary table and plot
def executive_summary_rt(df_rt, df_kpi, df_trip,
                         hole_diameter, 
                         stand_length = STAND_LENGTH,
                         depth_unit = 'ft',
                         rop_unit = 'ft/h',
                         trip_speed_unit = 'ft/h',
                         use_conn_table = False, 
                         number_connections = [None, None],
                         duration_connections = [None, None],
                         replace_dot = replace_dot, round_ndigits = round_ndigits,
                         current_well_name = CURRENT_WELL_NAME,
                         csv_folder = 'csv', plot_folder = 'plot',
                         save_folder = SAVE_DAILY_REPORT_FOLDER, save = True):

    s = f"SECTION {decimal_fraction_str(hole_diameter)}" if hole_diameter!='all' else ''
    bs = str(hole_diameter).replace('.', replace_dot)

    df_ = df_rt.copy()
    kpi = df_kpi.copy().drop('PERFORMANCE', axis = 0)
    df_trip_rt_bs = df_trip.copy()

    #activity length
    if hole_diameter != 'all':
        df_ = df_[df_[BS] == hole_diameter]
        
    length = []
    time = []
    for activity in ['ROTATE', 'SLIDE']:
        try:
            lg = round(find_act_depth(df_, activity = activity, activity_dict = sub_activity_state_swap, 
                          activity_col = RSUBACT, depth_col = HD, datetime_col = DATETIME),1)
        except:
            lg = 0 
        try:
            tm = find_act_time(df_, activity = activity, activity_dict = sub_activity_state_swap, 
                          activity_col = RSUBACT, depth_col = TIME, datetime_col = DATETIME)
        except:
            tm = 0
        length.append(lg)
        time.append(tm)

    length_trip=[]
    time_trip=[]
    for activity in ['TRIP IN', 'TRIP OUT']:
        try :
            lg = round(find_act_depth(df_, activity = activity, activity_dict = sub_activity_state_swap, 
                                      activity_col = RSUBACT, depth_col = BD, datetime_col = DATETIME),1)
        except:
            lg=0
        try :
             tm = find_act_time(df_, activity = activity, activity_dict = sub_activity_state_swap, 
                                      activity_col = RSUBACT, depth_col = TIME, datetime_col = DATETIME)
        except:
            tm=0
        length_trip.append(abs(lg))
        time_trip.append(tm)

    #number of conections
    length_trip_total = sum([abs(x) for x in length_trip])
    length_drill_total = sum([abs(x) for x in length])

    if use_conn_table:
        length_all = length + [length_trip_total] + number_connections
        time_all = time + [sum([abs(x) for x in time_trip])] + duration_connections
        #print(time_all)
    else:
        length_all = length + [length_trip_total] + [np.ceil(length_drill_total/stand_length), np.ceil(length_trip_total/stand_length)]
        time_all = time + [sum([abs(x) for x in time_trip])] + \
                   [kpi.loc[('CONNECTION','DRILL CONNECTION (mm:ss)'),(s,current_well_name)]*length_all[-2],
                    kpi.loc[('CONNECTION','TRIP CONNECTION (mm:ss)'),(s,current_well_name)]*length_all[-1]]
        #print('time_all, length_drill_total, length_trip_total', time_all, length_all[-2], length_all[-1])

    kpi['LENGTH'] = length_all
    
    time_conversion = 60 #min to hours
    kpi['TIME (hrs)'] = np.array(time_all)/time_conversion
   
    #fill missing values with 0
    kpi.fillna(0, inplace=True)
    kpi = kpi.replace([np.inf, -np.inf], 0)
    
    ##### drilling part
    drilling_kpi_summary = kpi.reset_index()
    drilling_kpi_summary = drilling_kpi_summary.iloc[[0,1,3],:]
    #renaming columns and content
    drilling_kpi_summary.columns = ['Activity', 'Sub-Activity', 'Performance', f'Length ({depth_unit})', 'Time (hh:mm)']
    drilling_kpi_summary['Activity'] = ['Drilling']*3
    drilling_kpi_summary['Sub-Activity'] = ['Rotate', 'Slide', 'Drill Connection']
    #add new column
    drilling_kpi_summary['# Connections'] = ['-', '-', int(round(drilling_kpi_summary.iloc[2,3]))]
    drilling_kpi_summary.iloc[2,3] = '-'
    
    #rearrange columns
    use_cols = ['Activity', 'Sub-Activity', f'Length ({depth_unit})', 'Performance',  '# Connections', 'Time (hh:mm)']
    drilling_kpi_summary = drilling_kpi_summary[use_cols]
    drilling_kpi_summary = drilling_kpi_summary.groupby(['Activity','Sub-Activity'])[use_cols[2:]].first()
    #set row order
    drilling_kpi_summary = drilling_kpi_summary.loc[[('Drilling', 'Rotate'), 
                                                     ('Drilling', 'Slide'),
                                                     ('Drilling', 'Drill Connection')]]

    #add units to performance column
    drilling_kpi_summary.loc[('Drilling', 'Rotate'),'Performance'] = str(round(drilling_kpi_summary.loc[('Drilling', 'Rotate'),'Performance'],1)) + f' ({rop_unit})'
    drilling_kpi_summary.loc[('Drilling', 'Slide'),'Performance'] = str(round(drilling_kpi_summary.loc[('Drilling', 'Slide'),'Performance'],1)) + f' ({rop_unit})'
    drilling_kpi_summary.loc[('Drilling', 'Drill Connection'),'Performance'] = str(convert_time_decimal_mmss(drilling_kpi_summary.loc[('Drilling', 'Drill Connection'),'Performance'])) \
                                                                             + ' (mm:ss)'

    drilling_kpi_summary['Time (hh:mm)'] = drilling_kpi_summary['Time (hh:mm)']\
                                         .apply(lambda x: convert_time_decimal_hhmm(x))
    ##### tripping part

    #compute trip-in, trip-out pipe_speed: PIPESP
    trip_in_trip_out_speed = [df_trip_rt_bs.loc[df_trip_rt_bs[RSUBACT] == 8, PIPESP].median(),
                              df_trip_rt_bs.loc[df_trip_rt_bs[RSUBACT] == 9, PIPESP].median()]

    tripping_kpi_summary = pd.DataFrame({'Activity': ['Tripping']*3,
                                         'Sub-Activity':['Trip-In', 'Trip-Out', 'Trip Connection'], 
                                         f'Length ({depth_unit})': [round(length,1) for length in length_trip] + ['-'], 
                                         'Performance': trip_in_trip_out_speed + [kpi.loc[('CONNECTION', 'TRIP CONNECTION (mm:ss)'),(s,current_well_name)]],
                                         '# Connection': ['-', '-', int(round(kpi.loc[('CONNECTION', 'TRIP CONNECTION (mm:ss)'),'LENGTH']))], 
                                         'Time (hh:mm)': [abs(x)/60 for x in time_trip] + [kpi.loc[('CONNECTION', 'TRIP CONNECTION (mm:ss)'),'TIME (hrs)']]}, 
                                         index = [0, 1, 2])

    tripping_kpi_summary = tripping_kpi_summary.groupby(['Activity','Sub-Activity']).first()
    tripping_kpi_summary = tripping_kpi_summary.loc[[('Tripping', 'Trip-In'), 
                                                     ('Tripping', 'Trip-Out'),
                                                     ('Tripping', 'Trip Connection')]]
 
    tripping_kpi_summary['Time (hh:mm)'] = tripping_kpi_summary['Time (hh:mm)']\
                                         .apply(lambda x: convert_time_decimal_hhmm(x))
               
    #add units to performance column
    tripping_kpi_summary.loc[('Tripping', 'Trip-In'),'Performance'] = str(round(tripping_kpi_summary.loc[('Tripping', 'Trip-In'),'Performance'],1)) + f' ({trip_speed_unit})'
    tripping_kpi_summary.loc[('Tripping', 'Trip-Out'),'Performance'] = str(round(tripping_kpi_summary.loc[('Tripping', 'Trip-Out'),'Performance'],1)) + f' ({trip_speed_unit})'
    tripping_kpi_summary.loc[('Tripping', 'Trip Connection'),'Performance'] = str(convert_time_decimal_mmss(tripping_kpi_summary.loc[('Tripping', 'Trip Connection'),'Performance'])) + ' (mm:ss)'

    #save table
    if save:
        tripping_kpi_summary.round(round_ndigits).to_csv(f"{save_folder}{csv_folder}/tripping_kpi_executive_summary_table_bs_{bs}_rt.csv", index = True)
        drilling_kpi_summary.round(round_ndigits).to_csv(f"{save_folder}{csv_folder}/drilling_kpi_executive_summary_table_bs_{bs}_rt.csv", index = True)
        dfi.export(drilling_kpi_summary, f"{save_folder}{plot_folder}/drilling_kpi_executive_summary_table_bs_{bs}_rt.png")
        dfi.export(tripping_kpi_summary, f"{save_folder}{plot_folder}/tripping_kpi_executive_summary_table_bs_{bs}_rt.png")

    return kpi, drilling_kpi_summary, tripping_kpi_summary

## savings potential functions
def compute_percentile(df_input, hole_diameter, wells_select, rig_activity, kpi_col, 
                       kpi_label, p_values = np.arange(50,100,10)/100, 
                       show_dist_plot = 1, WELL = WELLID,
                       replace_dot = replace_dot, round_ndigits = round_ndigits):
    
    """This function computes percentiles for drilling KPIs"""

    df = df_input.copy()

    #select specific wells
    df = df.loc[(df[WELL].isin(wells_select))]
    
    #select hole size
    if hole_diameter != 'all':
        df = df.loc[df[BS] == hole_diameter]
    #section name   
    s = f'SECTION {hole_diameter}"' if hole_diameter!='all' else ''
    bs = str(hole_diameter).replace('.', replace_dot)

    #for drilling mode select rotary or slide
    if (rig_activity == 'ROTATE')|(rig_activity == 'SLIDE'):
        df = df.loc[df[RSUBACT] == sub_activity_state_swap[rig_activity]]
                
    #flip p-values for connection activity
    p = p_values if ('CONNECTION' not in kpi_label) else 1 - p_values
    
    df_q = pd.DataFrame(df[kpi_col].quantile(q = p))
    
    #reindex for connection activity P10 = P90 (to keep all in one table)
    df_q.index = p_values
    #display(df_q)
    
    ######################
    ##plot distribution ##
    ######################
    #if not(df[kpi_col].empty):
    #try:
    if show_dist_plot:
        fig, ax = plt.subplots(1,1,figsize=(5,3))

        ax.set_xlabel(kpi_label)
        ax.set_title(kpi_label + ' distribution')

        sns.distplot(df[kpi_col].dropna(), ax = ax)    

        for ind, row in df_q.iterrows():
            #print(ind, ': ', round(val))
            ax.axvline(x = row[kpi_col])

    #except:
    #    print(kpi_col, 'did not plot')
    #    display(df[kpi_col])
    ########################

    #############################
    ##construct quantile table ##
    #############################
    
    #rename kpi_col to value
    df_q.rename(columns = {kpi_col: 'Value'}, inplace = True)
    
    #data density
    #data_portion = p_values if ('Connection' in kpi_label) else 
    df_q['Data'] = (df.shape[0] * (1 - p_values)).astype(int)
    
    #rename index
    if ('CONNECTION' not in kpi_label):
        df_q.index = df_q.index.map({val: 'P' + str(int(val*100)) for val in df_q.index.to_list()})
    else:
        df_q.index = df_q.index.map({val: 'P' + str(int(round((1 - val)*100))) for val in df_q.index.to_list()})

    #format dataframe
    df_q = df_q.T.unstack().to_frame().T

    #rename index to specified label
    df_q.index = [kpi_label]
        
    return df_q

def highlight_close_3(s, p_values, p_cols, p_value_colors, p_select_color):
    '''
    highlight the maximum in a Series yellow.
    '''

    if s.name in [('SELECT', ''), ('SAVINGS (%)', '')]:
        return ['background-color: ' + p_select_color] * s.shape[0]

    else:
        l_all = []

        for ii, row in p_values[p_cols].iterrows():
            
            l = []
            
            for ind, val in row.iteritems():
                #print(ind)
                if s.name[0] == val:
                    #print(i, ii, ind, val)
                    l_add = f'background-color: {p_value_colors[ind]}' 
                    #i += 1
                else:
                    l_add = ''

                l.append(l_add)
            
            l_all.append(l)
            
        #scan over all 
        v_all = np.array(l_all)
        
        l_combine = []

        for ii in range(v_all.shape[0]):

            elem = ''
            for el in v_all[ii,:]:
                if el != '':
                    elem = el 

            l_combine.append(elem)

        return l_combine

def highlight_cols(x, p_value_colors):
    #copy df to new - original data are not changed
    df = x.copy()
    #select all values to default value - red color
    df.loc[:,:] = 'background-color: None'
    #overwrite values grey color
    for col in [f'P-VALUE_{CURRENT_WELL_NAME}','P-VALUE_RECOMMENDED','P-VALUE_CLIENT']:
        df[[col]] = 'background-color: ' + p_value_colors[col]
    
    return df    

def highlight_cols_savings(x):
    #copy df to new - original data are not changed
    df = x.copy()
    #select all values to default value - red color
    df.loc[:,:] = 'background-color: None'
    #overwrite values grey color
    df[[('Current Well KPI','Value'),('Current Well KPI','Savings (%)')]] = ('background-color: ' + color_RT)
    #overwrite values grey color
    df[[('Recommended KPI','Value'),('Recommended KPI','Savings (%)')]] = ('background-color: ' + color_overperf)
    #overwrite values grey color
    df[[('Client Selected KPI','Value'),('Client Selected KPI','Savings (%)')]] = ('background-color: ' + color_historic)
    #return color df
    return df  

#generate savings potential table and plot
def savings_potential(dfs, df_kpi, hole_diameter, 
                      p_recom_vals, p_client_vals, p_select = None,
                      rop_unit = 'ft/h', trip_speed = 'ft/h',
                      current_well_name = CURRENT_WELL_NAME,
                      colors = [color_overperf, color_historic, color_RT],
                      save_folder = SAVE_DAILY_REPORT_FOLDER, save = True):

    p_select_color = colors[0] if p_select == p_recom_vals else colors[1]
    
    if p_select == None:
        p_select_color = colors[-1]

    kpi = df_kpi.copy().drop('PERFORMANCE', axis = 0)

    p_cols = ['P-VALUE_RECOMMENDED', 'P-VALUE_CLIENT', f'P-VALUE_{current_well_name}']
    p_value_colors = dict(zip(p_cols, colors))

    #KPI labels
    kpi_labels = [f'ROTATE ROP ({rop_unit})', f'SLIDE ROP ({rop_unit})',
                  f'TRIP SPEED ({trip_speed})',
                  'DRILL CONNECTION (mm:ss)', 'TRIP CONNECTION (mm:ss)']

    p_recom = dict(zip(kpi_labels, p_recom_vals))
    p_client = dict(zip(kpi_labels, p_client_vals))

    #rig activities
    rig_activities = ['ROTATE', 'SLIDE','','','']

    #KPI columns
    kpi_cols = [ROP, ROP, PIPESP, CONNDTIME, CONNTTIME]

    s = 'SECTION ' + decimal_fraction_str(hole_diameter) if (hole_diameter != 'all') else ''
    bs = str(hole_diameter).replace('.', replace_dot)

    #quantile table
    df_q = pd.DataFrame()

    for _df, rig_activity, kpi_col, kpi_label in zip(dfs[:-2], rig_activities[:-2], kpi_cols[:-2], kpi_labels[:-2]):

        df_q = df_q.append(compute_percentile(_df, hole_diameter, WELLS_SELECT_ID + [CURRENT_WELL_ID], rig_activity, 
                                                kpi_col, kpi_label, 
                                                p_values = np.arange(40,100,10)/100,
                                                show_dist_plot = 0)).fillna(0)

    df_q2 = pd.DataFrame()
    for _df, rig_activity, kpi_col, kpi_label in zip(dfs[-2:], rig_activities[-2:], kpi_cols[-2:], kpi_labels[-2:]):

        df_q2 = df_q2.append(compute_percentile(_df, hole_diameter, WELLS_SELECT_ID + [CURRENT_WELL_ID], rig_activity, 
                                              kpi_col, kpi_label, 
                                              p_values = np.arange(40,100,10)/100,
                                              show_dist_plot = 0)).fillna(0)

    d = df_q.unstack().to_frame().reset_index()

    #drop values
    d = d.loc[d['level_1'] == 'Value'].drop(columns=['level_1']).rename(columns={'level_0':'P-VALUE',
                                                                                 'level_2':'ACTIVITY',
                                                                                0:'Value'
                                                                                })

    d2 = df_q2.unstack().to_frame().reset_index()
    #drop values
    d2 = d2.loc[d2['level_1'] == 'Value'].drop(columns=['level_1']).rename(columns={'level_0':'P-VALUE',
                                                                                 'level_2':'ACTIVITY',
                                                                                0:'Value'
                                                                                })

    d = d.append(d2)

    kpi_r = kpi.reset_index().droplevel(level=0,axis=1).iloc[:,1:]
    kpi_r.columns = ['ACTIVITY', AVG, current_well_name]
    #display(kpi_r)

    d = d.merge(kpi_r, how = 'left', right_on = 'ACTIVITY', left_on = 'ACTIVITY')

    #calculate differenct between current well data and P50 Value for offset wells + current well
    d[current_well_name+'_diff'] = abs(d[current_well_name] - d['Value'])
    #display(d.groupby(['ACTIVITY'])[[current_well+'_diff']].min())
    d_ind = d.groupby('ACTIVITY')[current_well_name+'_diff'].idxmin()

    #remove nan values
    inds = d_ind.values
    inds = inds[~np.isnan(inds)]

    p_add = d.iloc[inds].loc[:,['ACTIVITY', 'P-VALUE']]

    _kpi = kpi.reset_index().droplevel(level=0,axis=1)
    _kpi.columns = ['AREA','ACTIVITY', AVG, current_well_name]

    _kpi = _kpi.merge(p_add, how='left', left_on = 'ACTIVITY', right_on = 'ACTIVITY').set_index(['AREA','ACTIVITY'])

    _kpi.rename(columns = {'P-VALUE': f'P-VALUE_{current_well_name}'}, inplace=True)


    _kpi[f'P-VALUE_RECOMMENDED'] = [val for key, val in p_recom.items()]
    _kpi[f'P-VALUE_CLIENT'] = [val for key, val in p_client.items()]

    p_current = _kpi[f'P-VALUE_{current_well_name}'].droplevel(level=0,axis=0)
    s_add = '' if s == '' else '_' + s.strip('\"').replace(' ','_').lower()
    
    # *** July 29 change ***: select recomended percentiles
    if p_select == None:
        df_q['SELECT'] = _kpi.droplevel(level=0,axis=0)[current_well_name] 
        df_q2['SELECT'] = _kpi.droplevel(level=0,axis=0)[current_well_name]
    else:
        df_q['SELECT'] = df_q[(p_select[0],'Value')]
        df_q2['SELECT'] = df_q2[(p_select[-1],'Value')]

    df_q['SAVINGS (%)'] = (df_q['SELECT'] - df_q[('P50','Value')])/df_q[('P50','Value')] *100
    df_q2['SAVINGS (%)'] = -(df_q2['SELECT'] - df_q2[('P50','Value')])/df_q2[('P50','Value')] *100

    if save:

        #save table
        df_styled =_kpi.style.apply(highlight_cols, axis=None, p_value_colors = p_value_colors) \
                      .set_caption(s).set_table_styles([STYLE]) \
                      .format(convert_time_decimal_mmss, subset = pd.IndexSlice[[('CONNECTION', 'DRILL CONNECTION (mm:ss)'),
                                                                                 ('CONNECTION', 'TRIP CONNECTION (mm:ss)')], 
                                                                                 ['AVG', CURRENT_WELL_NAME]])

        display(df_styled)

        dfi.export(df_styled, f'{save_folder}plot/kpi_p_values_table_bs_{bs}.png')
        _kpi.round(3).to_csv(f'{save_folder}csv/kpi_p_values_bs_{bs}.csv', index=True)

        subset_data = [col for col in df_q.columns if 'Data' in col]

        df_styled = df_q.style.format('{:,.1f}').apply(highlight_close_3, p_values=_kpi.iloc[:-2], p_cols = p_cols, 
                                    p_value_colors = p_value_colors, p_select_color = p_select_color)\
                              .set_caption(s).set_table_styles([STYLE])\
                              .format('{:.0f}', subset = subset_data)
        display(df_styled)
        #save table
        dfi.export(df_styled, f'{save_folder}plot/p_values_drill_trip_table_bs_{bs}.png')
        df_q.round(2).to_csv(f'{save_folder}csv/p_values_drill_trip_bs_{bs}.csv', index=True)

        subset_value = [col for col in df_q2.columns if 'Value' in col]
        subset_data = [col for col in df_q2.columns if 'Data' in col]

        df_styled = df_q2.style.format('{:,.1f}').apply(highlight_close_3, p_values=_kpi.iloc[-2:], p_cols = p_cols, 
                                                        p_value_colors = p_value_colors, p_select_color = p_select_color)\
                                                 .set_caption(s).set_table_styles([STYLE])\
                                                 .format(convert_time_decimal_mmss, subset = subset_value)\
                                                 .format('{:.0f}', subset = subset_data)
        display(df_styled)
        #save table
        dfi.export(df_styled, f'{save_folder}plot/p_values_conn_table_bs_{bs}.png')
        df_q2.round(round_ndigits).to_csv(f'{save_folder}csv/p_values_conn_bs_{bs}.csv', index=True)
        
    ## plot for overall saving
    df_savings_value = pd.DataFrame()

    df_savings_value[f'P-VALUE_{current_well_name}'] = _kpi.droplevel(level=0,axis=0)[current_well_name]

    for p_col in ['P-VALUE_RECOMMENDED', 'P-VALUE_CLIENT']:

        S = pd.Series()

        for ind, val in _kpi[p_col].droplevel(level=0,axis=0).iteritems():
            #print(ind, val)
            if 'CONNECTION' not in ind:

                S[ind] = (df_q.loc[ind, (val,'Value')])
            else:
                S[ind] = (df_q2.loc[ind, (val,'Value')])

        df_savings_value[p_col] = S
    #add reference P50
    S = pd.Series()

    for ind in df_savings_value.index:
        #print(ind, val)
        if 'CONNECTION' not in ind:
            S[ind] = (df_q.loc[ind, ('P50','Value')])
        else:
            S[ind] = (df_q2.loc[ind, ('P50','Value')])

    df_savings_value['P50'] = S
        
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for col in df_savings_value.columns:

        df1[col] = ((df_savings_value[col] - df_q[('P50','Value')])/df_q[('P50','Value')] *100).dropna()
        df2[col] = -((df_savings_value[col] - df_q2[('P50','Value')])/df_q2[('P50','Value')] *100).dropna()


    df_savings_pct = df1.append(df2)

    ax = df_savings_pct.iloc[:,:-1].plot(kind='bar',figsize=(15,8), color = [color_RT, color_overperf, color_historic])
    ax.set_xlabel('ACTIVITY')
    ax.set_ylabel('SAVINGS (%)')
    ax.legend(['Current Well KPI', 'Recommended KPI', 'Client Selected KPI'])
    ax.set_title(s)
    plt.tight_layout()

    if save:
        plt.savefig(f'{save_folder}plot/savings_potential_bs_{bs}.png',dpi=150)
        df_savings_pct.to_csv(f'{save_folder}csv/savings_potential_bs_{bs}.csv')
        
        #savings data in table
        _dict = {f'P-VALUE_{current_well_name}': 'Current Well KPI', 
                'P-VALUE_RECOMMENDED': 'Recommended KPI', 
                'P-VALUE_CLIENT': 'Client Selected KPI',
                'P50': 'Reference KPI'}

        df_savings = df_savings_value.merge(df_savings_pct, left_index = True, right_index = True, how = 'left')

        df_savings.columns = pd.MultiIndex.from_tuples([(_dict[v[:-2]], 'Value') if v.endswith('_x') 
                                                    else (_dict[v[:-2]], 'Savings (%)') for v in df_savings.columns])

        df_savings.sort_index(axis=1, level=0, ascending=False, inplace=True)
        df_savings.fillna(0, inplace = True)
        
        subset_value = [col for col in df_savings.columns if 'Value' in col]

        df_styled = df_savings.style.format('{:,.1f}').apply(highlight_cols_savings, axis = None)\
                              .set_caption(s).set_table_styles([STYLE])\
                              .format(convert_time_decimal_mmss, subset = pd.IndexSlice[['DRILL CONNECTION (mm:ss)',
                                                                                         'TRIP CONNECTION (mm:ss)'], 
                                                                                        subset_value])

        
        display(df_styled)
        dfi.export(df_styled, f'{save_folder}plot/savings_potential_table_bs_{bs}.png')
        df_savings.round(round_ndigits).to_csv(f'{save_folder}csv/savings_potential_bs_{bs}.csv', index=True)

    plt.close()

    return None

#for implementation
def post_process_wt_wt_conn_time(compute_col, table_name,
                                 HD = HD, DATETIME = DATETIME,
                                 TIME = TIME, BS = BS,
                                 SAVE_FOLDER = SAVE_FOLDER + 'csv/'):

    df_wt_wt_rt_kpi = pd.read_csv(SAVE_FOLDER + table_name, parse_dates=[DATETIME])

    #add datetime flag
    datetime_thr = pd.Timedelta(4, unit='hour')
    df_wt_wt_rt_kpi[DATETIME + '_flag'] = 0
    cond = (df_wt_wt_rt_kpi[DATETIME].diff().shift(-1) > datetime_thr)
    df_wt_wt_rt_kpi.loc[cond, DATETIME + '_flag'] = df_wt_wt_rt_kpi.loc[cond].index

    #add measure_depth flag
    measured_depth_thr = 5 #ft
    #add measured_depth smooth defined by measured_depth_thr
    df_wt_wt_rt_kpi[HD + '_smooth'] = 0
    df_wt_wt_rt_kpi[HD + '_smooth_next'] = df_wt_wt_rt_kpi[HD].copy()
    i = 1
    #run recursively until all small measured depths are merged
    while not((df_wt_wt_rt_kpi[HD + '_smooth'] == df_wt_wt_rt_kpi[HD + '_smooth_next']).all()):
        df_wt_wt_rt_kpi[HD + '_smooth'] = df_wt_wt_rt_kpi[HD + '_smooth_next'].copy()
        #print(f"*** ITERATION {i} ***")
        for index, value in df_wt_wt_rt_kpi[HD + '_smooth'].items():
            #initial conditions
            if index == 0:
                value_prev = value
            else:
                if abs(value - value_prev) < measured_depth_thr:
                    df_wt_wt_rt_kpi[HD + '_smooth_next'].iloc[index] = value_prev
                    #print('update from hd = ', value, 'update to hd = ', value_prev)
                #update previous value
                value_prev = value
        i += 1

    df_wt_wt_rt_kpi.drop(columns = [HD + '_smooth_next'], inplace = True)

    #fix datetime_flag
    df_wt_wt_rt_kpi[HD + '_diff'] = df_wt_wt_rt_kpi[HD + '_smooth'].diff().fillna(0)
    df_wt_wt_rt_kpi[DATETIME + '_flag'] = df_wt_wt_rt_kpi[DATETIME + '_flag']*df_wt_wt_rt_kpi[HD + '_diff']

    #show part of the data
    #use_cols = [DATETIME, HD, HD + '_smooth', BS, DATETIME + '_flag', compute_col]
    #display(df_wt_wt_rt_kpi[use_cols].head(20))

    #group
    df_wt_wt_rt_kpi_grouped = df_wt_wt_rt_kpi.groupby([HD + '_smooth', BS, DATETIME + '_flag'])\
                    [DATETIME, TIME, BD, compute_col, compute_col+'_worst', compute_col+'_avg', compute_col+'_best', KPIC]\
                    .agg({DATETIME:'min', BD: 'min', TIME: 'min', compute_col: 'sum', 
                    compute_col+'_worst': lambda x:x.value_counts().index[0],
                    compute_col+'_avg': lambda x:x.value_counts().index[0],
                    compute_col+'_best': lambda x:x.value_counts().index[0],
                    KPIC: 'max'})\
                    .reset_index().sort_values(by = DATETIME).rename(columns ={HD + '_smooth': HD})

    #use_cols = [DATETIME, HD, BS, DATETIME + '_flag', compute_col]
    #display(df_wt_wt_rt_kpi_grouped[use_cols].head(20))

    wtwt_cols = [DATETIME, HD, BD, TIME, BS, compute_col, compute_col+'_worst', compute_col+'_avg', compute_col+'_best', KPIC]
    df_wt_wt_rt_kpi_grouped[wtwt_cols].to_csv(SAVE_FOLDER + table_name.replace("old", "new"), index = False)

#process kpi wt_wt and conn_drill_time
def process_kpi(df_input: pd.DataFrame, measured_depth_thr,
                datetime_thr = pd.Timedelta(4, unit='hour'),
                DATETIME = DATETIME, HD = HD) -> pd.DataFrame:

    df = df_input.copy()

    #add datetime flag
    df[DATETIME + '_flag'] = 0
    cond = (df[DATETIME].diff().shift(-1) > datetime_thr)
    df.loc[cond, DATETIME + '_flag'] = df.loc[cond].index

    #add measured_depth smooth defined by measured_depth_thr
    df[HD + '_smooth'] = 0
    df[HD + '_smooth_next'] = df[HD].copy()
    i = 1
    #run recursively until all small measured depths are merged
    while not((df[HD + '_smooth'] == df[HD + '_smooth_next']).all()):
        df[HD + '_smooth'] = df[HD + '_smooth_next'].copy()
        #print(f"*** ITERATION {i} ***")
        for index, value in df[HD + '_smooth'].items():
            #initial conditions
            if index == 0:
                value_prev = value
            else:
                if abs(value - value_prev) < measured_depth_thr:
                    df[HD + '_smooth_next'].iloc[index] = value_prev
                    #print('update from hd = ', value, 'update to hd = ', value_prev)
                #update previous value
                value_prev = value
        i += 1

        if i > 10:
            # display(df[HD + '_smooth'])
            # display(df[HD + '_smooth_next'])
            # print((df[HD + '_smooth'] == df[HD + '_smooth_next']).sum())
            # print(df[HD + '_smooth'].shape)
            break

    df.drop(columns = [HD + '_smooth_next', HD], inplace = True)
    df.rename(columns ={HD + '_smooth': HD}, inplace = True)

    #fix datetime_flag
    df[HD + '_diff'] = df[HD].diff().fillna(0)
    df[DATETIME + '_flag'] = df[DATETIME + '_flag']*df[HD + '_diff']
                   
    return df 

#process conn_trip_time
def process_kpi_trip(df_conn_trip, stand_length, delta_stand_length, BD = BD):
    current_sum = 0
    prev_index = 0
    criteria = []
    values = df_conn_trip[BD].diff().shift(-1).abs().fillna(0).values
    df_conn_trip[BD + '_diff'] = values
    for ind, value in enumerate(values):
        current_sum += value
        if (current_sum > stand_length + delta_stand_length):
            prev_index = ind
            current_sum = value
            criteria.append(prev_index)
        elif (current_sum < stand_length + delta_stand_length)&(current_sum > stand_length - delta_stand_length)&(current_sum == value):
            prev_index = ind
            criteria.append(prev_index)
        else: 
            criteria.append(prev_index)

    df_conn_trip[BD + '_diff_flag'] = criteria

    return df_conn_trip

#*** updated functions Sep 24 ***#
#function to compute weight to weight table 
def compute_df_wt_wt(df_input: pd.DataFrame, measured_depth_thr = 5,
                     wt_wt_max = 200,
                     WELLID = WELLID, BS = BS, WTWT = WTWT,
                     LABELcd = LABELcd, TIME = TIME) -> pd.DataFrame:
    """This function computes weight to weight dataframe.
    Here df_input is time-based drilling data, 
    WELLID, BS, WTWT, LABELcd, TIME represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions.
    Function returns df_wt_wt dataframe."""

    df = df_input.copy()

    df_wt_wt = df.groupby([WELLID, LABELcd])[[TIME]].agg(['first', 'last'])
    #rename columns
    df_wt_wt.columns = ['_'.join(col) for col in df_wt_wt.columns]
    #delta time (minutes)
    df_wt_wt[WTWT] = (df_wt_wt[TIME + '_first'].shift(-1) - df_wt_wt[TIME + '_last'])
    #remove negative
    df_wt_wt = df_wt_wt[~((df_wt_wt[df_wt_wt < 0].any(axis=1)))]

    #add section size
    df_wt_wt[BS] = df.groupby([WELLID, LABELcd])[BS].apply(lambda x: max(x.unique()))

    df_aux = df.groupby([WELLID, LABELcd])[[HD, BD, DATETIME, TIME]].min().reset_index()
    df_wt_wt = df_wt_wt.reset_index().merge(df_aux, how = 'left', 
                            left_on=[WELLID, LABELcd], right_on =[WELLID, LABELcd])

    if df_wt_wt.shape[0] > 1:
        df_process = process_kpi(df_wt_wt, measured_depth_thr)

            #group
        df_grouped = df_process.groupby([HD, BS, DATETIME + '_flag'])[WTWT, 
                        DATETIME, WELLID, LABELcd, TIME + '_first', TIME + '_last']\
                    .agg({DATETIME:'min', WTWT: 'sum',
                            WELLID: 'max', LABELcd: 'min',
                            TIME + '_first': 'min',
                            TIME + '_last': 'max'})\
                    .reset_index().sort_values(by = DATETIME)
    else:
        df_grouped = df_wt_wt

    print('\n df_wt_wt size before filtering high weigh to weight times:', df_grouped.shape[0])
    df_grouped = df_grouped.loc[(df_grouped[WTWT] < wt_wt_max),:]
    print('\n df_wt_wt size after filtering high weigh to weight times:', df_grouped.shape[0])

    df_grouped = df_grouped.set_index(LABELcd)
    use_cols = [WELLID, TIME + '_first', TIME + '_last', WTWT, BS]

    return df_grouped[use_cols]

#function to compute connection time tables
def compute_df_conn(df_input: pd.DataFrame, 
                    compute_col: str, activity: str, dtime_dict: dict,
                    stand_length = STAND_LENGTH, delta_stand_length = 5,
                    measured_depth_thr = 5,
                    WELLID = WELLID, BS = BS, RSUBACT = RSUBACT, 
                    BD = BD, dBD = dBD,  
                    LABEL = LABEL, TIME = TIME,
                    activity_dict = sub_activity_state_swap) -> pd.DataFrame:
    """This function computes connection time dataframe.
    Here df_input is time-based drilling data, 
    compute_col is a column name for the computed connection time,
    activity is an activity name (conneciton while drilling or tripping),
    dtime_dict is a dictionary that define mapping from well ids to time step/data density.
    delta_stand_length is +/- stand_length at connection time post-process step,
    measured_depth_thr is measured_depth smoothing parameter at post-process step,
    WELLID, BS, RSUBACT, BD, dBD, TIME represent standard column names, 
    see standard naming and ada_naming_convention.xlsx file for descriptions,
    activity_dict is a dictionary that define mapping from names to number codes.
    Function returns df_conn for tripping or drilling dataframe."""

    df = df_input.copy()

    clip_max = 20 #max value is 5 min

    df_conn = pd.DataFrame()

    #compute connection time using specific time step for each well
    for well_id in df[WELLID].unique():

        df_ = df[df[WELLID] == well_id]

        if not(df_[df_[RSUBACT] == activity_dict[activity]].empty):
            df_cd = pd.DataFrame(df_[df_[RSUBACT] == activity_dict[activity]]\
                            .groupby([LABEL])[TIME].apply(lambda x: x.nunique() * dtime_dict[well_id])\
                            .dt.total_seconds()/60).reset_index()
            df_cd[WELLID] = well_id
            df_conn = df_conn.append(df_cd)

    if not(df_conn.empty):      
        df_conn.set_index([WELLID,LABEL], inplace = True)   
        df_conn[dBD] = df[df[RSUBACT] == activity_dict[activity]]\
                                        .groupby([WELLID, LABEL])[BD].apply(lambda x: x.max() - x.min())
        df_conn.rename(columns={TIME: compute_col},inplace=True)

        #add section size
        df_conn[BS] = df.groupby([WELLID, LABEL])[BS].apply(lambda x: max(x.unique()))
        df_conn = df_conn.reset_index().set_index(LABEL)

        #clip by 0
        df_conn[compute_col] = df_conn[compute_col].clip(lower=0, upper=clip_max)
    else:
        df_conn = pd.DataFrame(columns = [LABEL, WELLID, compute_col, dBD, BS],
             data = np.zeros([1,5])).set_index(LABEL)

    df_aux = df.groupby([WELLID, LABEL])[[HD, BD, DATETIME, TIME]].min().reset_index()
    df_conn = df_conn.reset_index().merge(df_aux, how = 'left', 
                            left_on=[WELLID, LABEL], right_on =[WELLID, LABEL])

    if compute_col == CONNDTIME:
        df_process = process_kpi(df_conn, measured_depth_thr)

        #group
        df_conn = df_process.groupby([HD, BS, DATETIME + '_flag'])[compute_col, 
                        DATETIME, WELLID, LABEL, dBD]\
                    .agg({DATETIME:'min', compute_col: 'sum',
                            WELLID: 'max', LABEL: 'min',
                            dBD: 'sum'})\
                    .reset_index().sort_values(by = DATETIME).set_index(LABEL)

    elif compute_col == CONNTTIME:
        df_process = process_kpi_trip(df_conn, stand_length, delta_stand_length)
        #group
        df_conn = df_process.groupby([BD + '_diff_flag'])[compute_col, 
                        DATETIME, WELLID, LABEL, dBD, BS]\
                    .agg({DATETIME:'min', compute_col: 'sum',
                            WELLID: 'max', LABEL: 'min',
                            dBD: 'sum', BS: 'min'})\
                    .reset_index().sort_values(by = DATETIME).set_index(LABEL)

    #clip 
    df_conn[compute_col] = df_conn[compute_col].clip(lower=0, upper=clip_max)
    use_cols = [WELLID, compute_col, dBD, BS]

    #filter trip connections faster than 1 minute
    if activity == 'CNX (trip)':
        df_conn = df_conn[df_conn[compute_col] >= 1]

    return df_conn[use_cols]

def compute_daily_depth_time_summary(df_input: pd.DataFrame,
                                     depth_unit = 'ft',
                                     DATETIME = DATETIME, HD = HD, TVD = TVD,
                                     current_well = CURRENT_WELL_ID,
                                     current_well_name = CURRENT_WELL_NAME,
                                     csv_folder = 'csv', plot_folder = 'plot',
                                     save = True, save_folder = SAVE_DAILY_REPORT_FOLDER):

    df_rt_bs = df_input.copy()
    #compute time boundaries per section
    report_time_start, report_time_end = pd.to_datetime(df_rt_bs.dropna(subset = [DATETIME])[DATETIME].iloc[[0,-1]].values)
    report_HD_start, report_TVD_start = df_rt_bs.dropna(subset = [HD, TVD])[[HD, TVD]].iloc[0].values.round(2)
    report_HD_end, report_TVD_end = df_rt_bs.dropna(subset = [HD, TVD])[[HD, TVD]].iloc[-1].values.round(2)

    total_time_diff = report_time_end - report_time_start
    total_HD_diff = round(report_HD_end - report_HD_start,2)
    total_TVD_diff = round(report_TVD_end - report_TVD_start,2)

    total_time_diff_hours = total_time_diff.components.hours
    total_time_diff_minutes = total_time_diff.components.minutes \
                            + (1 if total_time_diff.components.seconds >= 30 else 0)

    #account for 60 min 
    if total_time_diff_minutes == 60:
        total_time_diff_hours = str(total_time_diff_hours + 1).zfill(2)
        total_time_diff_minutes = '00' 
    else:
        total_time_diff_hours = str(total_time_diff_hours).zfill(2)
        total_time_diff_minutes = str(total_time_diff_minutes).zfill(2)

    #store start stop patameters in dataframe
    df_depth_time = pd.DataFrame({HD + '_start': report_HD_start,
                                TVD + '_start': report_TVD_start,
                                HD + '_end': report_HD_end,
                                TVD + '_end': report_TVD_end,
                                'total_HD_diff': total_HD_diff,
                                'total_TVD_diff': total_TVD_diff,
                                DATETIME + '_start': report_time_start,
                                DATETIME + '_end': report_time_end,
                                'total_time': total_time_diff_hours
                                                + ':' + total_time_diff_minutes}, 
                                    index = [' - '.join([str(int(current_well)), current_well_name])])

    #change datetime format
    df_depth_time[DATETIME + '_start'] = df_depth_time[DATETIME + '_start'].dt.strftime('%d/%m/%y %H:%M')
    df_depth_time[DATETIME + '_end'] = df_depth_time[DATETIME + '_end'].dt.strftime('%d/%m/%y %H:%M')

    print('\n' + '*'*15 + 'Current well intervals' + '*'*15 + '\n')
    print(f'\nMeasured depth interval for {current_well_name} is [{report_HD_start}, {report_HD_end}]')
    print(f'Time interval for {current_well_name} is [{report_time_start}, {report_time_end}]\n')
    print('\n' + '*'*52 + '\n')

    if save:
        #save csv
        df_depth_time.to_csv(f'{SAVE_DAILY_REPORT_FOLDER}{csv_folder}/drilled_depth_time_summary_rt.csv',index=True)
        #rename columns and make multiindexing
        df_styled = df_depth_time.copy()
        df_styled.columns = pd.MultiIndex(levels=[['Depth Drilled', 'Time Drilled'], 
                                                  [f'MD Start ({depth_unit})',f'TVD Start ({depth_unit})', 
                                                   f'MD End ({depth_unit})', f'TVD End ({depth_unit})', 
                                                   f'MD Length \nDrilled ({depth_unit})', f'TVD Length \nDrilled ({depth_unit})',
                                                   "Time Start \n(dd/mm/yy hh:mm)", "Time End \n(dd/mm/yy hh:mm)", 
                                                   "Total Time \nEvaluated on Report (hh:mm)"]], 
                                                    codes=[[0,0,0,0,0,0,1,1,1],[0,1,2,3,4,5,6,7,8]])
        display(df_styled)
        #save 2 table image: with depth and with time
        dfi.export(df_styled.iloc[:,:6].style.format('{:,.1f}').hide_index(), f"{SAVE_DAILY_REPORT_FOLDER}{plot_folder}/drilled_depth_summary_rt.png")
        dfi.export(df_styled.iloc[:,6:].style.hide_index(), f"{SAVE_DAILY_REPORT_FOLDER}{plot_folder}/drilled_time_summary_rt.png")

    return df_depth_time, report_time_start, report_time_end

def plot_performance_tendencies(df_input:pd.DataFrame, 
                                compute_col:str, compute_col_unit:str,
                                activity_map:dict, select_dates:list, table_size = 14,
                                sub_activity_state = sub_activity_state, add_year = False,
                                DATETIME = DATETIME, DATETIMED = DATETIMED,
                                RSUBACT = RSUBACT, CONNTIME = CONNTIME,
                                colors = [color_overperf, color_underperf],
                                csv_folder = 'csv', plot_folder = 'plot',
                                save_folder = SAVE_DAILY_REPORT_FOLDER, save = True):

    df_rt = df_input.copy()

    date_label = 'Date (dd/mm/yy)' if add_year else 'Date (dd/mm)'

    activity_codes = list(activity_map.keys())
    #add day column
    df_rt[DATETIMED] = df_rt[DATETIME].dt.strftime('%Y-%m-%d')
    #filter by activity codes
    df_rop_ = df_rt.loc[df_rt[RSUBACT].isin(activity_codes)].groupby([DATETIMED, RSUBACT])[[compute_col]].median().reset_index()
    #filter selected dates
    df_rop_ = df_rop_.loc[df_rop_[DATETIMED].isin(select_dates)]
    #check missing dates: at least current
    missing_dates = list(set(select_dates) - set(df_rop_[DATETIMED].unique()))
    missing_length = len(missing_dates)
    #add missing dates with nan values
    df_rop_ = df_rop_.append(pd.DataFrame({DATETIMED: missing_dates*2, 
                                           RSUBACT: [activity_codes[0]]*missing_length + [activity_codes[1]]*missing_length, 
                                           compute_col:[np.nan]*missing_length*2})).sort_values(DATETIMED)

    ##table
    df_rop = pd.pivot_table(df_rop_, values = compute_col, columns = DATETIMED, index = RSUBACT, dropna=False)
    #to keep nan values reassing df_rop_
    df_rop_ = df_rop.stack(dropna=False).reset_index().sort_values(by = DATETIMED).rename(columns={0:compute_col})
    #make proper table
    df_rop.index = df_rop.index.map(activity_map)
    df_rop.columns.names = [date_label]
    df_rop.index.names = ['']
    #df_rop.fillna('-', inplace = True) #for styled dataframe only
    #convert ('%Y-%m-%d') to ('%d/%m/%y')
    if add_year:
        df_rop.columns = ['/'.join([col[-2:], col[5:7], col[2:4]]) for col in df_rop.columns] 
        df_rop_[DATETIMED] = pd.to_datetime(df_rop_[DATETIMED], format='%Y-%m-%d').dt.strftime('%d/%m/%y')

    else:
        df_rop.columns = ['/'.join([col[-2:], col[5:7]]) for col in df_rop.columns]
        df_rop_[DATETIMED] = pd.to_datetime(df_rop_[DATETIMED], format='%Y-%m-%d').dt.strftime('%d/%m')

    ##plot
    #convert ('%Y-%m-%d') to ('%d/%m/%y')
    rescale_plot = max([df_rop_.shape[0],6])/table_size
    #print('rescale_plot',rescale_plot)
    #replace 0 values with nan
    df_rop_.loc[df_rop_[compute_col] == 0, compute_col] = np.nan

    ax = df_rop_.loc[df_rop_[RSUBACT]==activity_codes[0], [DATETIMED, compute_col]]\
                .set_index(DATETIMED).plot(kind='line',figsize=(10*rescale_plot,5), #12,5
                                           color = colors[0], marker = 'o',lw=2.0, markersize = 7)
    df_rop_.loc[df_rop_[RSUBACT]==activity_codes[1], [DATETIMED, compute_col]].set_index(DATETIMED)\
            .plot(kind='line',color = colors[1], marker = 'o', ax=ax, lw=2.0, markersize = 7)
    ax.set_xlabel('\n' + date_label)
    ax.set_ylabel(compute_col.replace('_', ' ').upper() + ' ' + compute_col_unit)
    ax.legend([sub_activity_state[activity_codes[0]],sub_activity_state[activity_codes[1]]])
    ax.set_title(compute_col.replace('_', ' ').upper())
    x_labels = list(df_rop_[DATETIMED].unique())
    plt.xticks(ticks = list(range(len(x_labels))), labels = x_labels, rotation = 90) #
    plt.tight_layout()

    if compute_col == CONNTIME:
        df_rop = df_rop.fillna(0).applymap(lambda x: convert_time_decimal_mmss(x))
        df_rop = df_rop.applymap(lambda x: x if x != '00:00' else np.nan)
        df_styled_format = None
    else:
        df_styled_format = '{:,.1f}'
        df_rop = df_rop.round(2)
        #df_styled = df_rop.style.format('{:,.1f}', na_rep='-')

    if save:
        #save csv
        df_rop.to_csv(f"{save_folder}{csv_folder}/performance_tendency_{compute_col}.csv", index = True)
        #save table
        df_rop_size = df_rop.shape[1]
        number_plots = max([int(np.ceil(df_rop_size/table_size)), 1])
        for i in range(number_plots):
            i_start = i * table_size
            i_end = min([i_start + table_size, df_rop_size])
            #print('i_start, i_end, df_rop_size', i_start, i_end, df_rop_size)
            dfi.export(df_rop.iloc[:, i_start:i_end].style.format(df_styled_format, na_rep='-'), 
                              f"{save_folder}{plot_folder}/performance_tendency_{compute_col}_table_{i}.png")

        #dfi.export(df_styled, f"{save_folder}plot/performance_tendency_{compute_col}_table.png")
        #save plot
        plt.savefig(f"{save_folder}{plot_folder}/performance_tendency_{compute_col}.png", dpi=150)

    plt.close()

    return df_rop, df_rop_

def save_dataframe_by_chunks(df: pd.DataFrame, chunk_size_Mb: int,
                             save_name:str, save_folder:str):
    total_memory_Mb = df.memory_usage().sum()/10**6
    chunk_number = int(np.ceil(total_memory_Mb/chunk_size_Mb))
    df_size = df.shape[0]
    chunk_size_df = int(np.ceil(df_size/chunk_number))
    check_size = 0
    for i in range(chunk_number):
        i_start = i * chunk_size_df
        i_end = min([i_start + chunk_size_df, df_size])
        check_size += df.iloc[i_start:i_end, :].shape[0]
        df.iloc[i_start:i_end, :].to_csv(save_folder + save_name + f'_part{i}.csv', index=False)

    #check that all data is splitted by chunks
    assert(check_size == df_size)
    
def unit_conversion(df_convert: pd.DataFrame, 
                    df_ada: pd.DataFrame, 
                    factor_col = None):
    """
    Convert standard unit to metric unit. Return DataFrame with new conversion unit.
    ---Parameters:
    df_convert: DataFrame. Data with standard unit and standard naming
    df_ada: DataFrame from 'ada_naming_convention.xlsx' sheet_name = 'naming'
    factor_col: str, Default: None (Multiply everything by 1)
    ---Output:
    df_new: DataFrame. Converted data
    summary: DataFrame. Summary that has conversion factor for each features
    """
    #test: factor_col=standard_metric_conversion

    df = df_convert.copy()
    summary = pd.DataFrame({})

    # ### Convert column type
    # if BS in df.columns:
    #     df[BS] = df[BS].astype(float, errors = 'ignore')
    # ###

    for feature in df.columns:
        #Some features such as consecutive_label might not present here
        if feature in df_ada['standard_naming'].values:
            if factor_col == None:
                factor = 1
            else:
                factor = df_ada[df_ada['standard_naming']==feature][factor_col].values[0]

            if factor!=1:
                if isinstance(factor, numbers.Number): #Check if numeric
                    df[feature] = df[feature]*factor
                else: #Check if string, example '(F-32)/1.8'
                    F = df[feature]
                    df[feature] = eval(F)
        else:
            factor = 1
        summary[feature] = [factor]

    print('\n*** Unit conversion summary ***\n')
    display(summary)

    return df

def compute_postjob_depth_time_summary(df_all: pd.DataFrame,
                                       depth_unit = 'ft',
                                       WELLID = WELLID, HD = HD, TVD = TVD,
                                       DATETIME = DATETIME,
                                       save_folder = SAVE_DAILY_REPORT_FOLDER, save = True):

    df_filter = pd.DataFrame()
    well_ids = df_all[WELLID].unique()
    well_ids = [x for x in well_ids if x == x]

    for well_id in well_ids:
        
        #select one well
        dfh = df_all.loc[df_all[WELLID] == well_id]

        #calculate start measured_depth/tvd
        report_HD_start, report_TVD_start, report_time_start = dfh[[HD, TVD, DATETIME]].min().values
        #calculate end measured_depth/tvd
        report_HD_end, report_TVD_end, report_time_end = dfh[[HD, TVD, DATETIME]].max().values

        df_filter_ = pd.DataFrame({HD + '_start': report_HD_start,
                                HD + '_end': report_HD_end,
                                TVD + '_start': report_TVD_start,
                                TVD + '_end': report_TVD_end,
                                DATETIME + '_start': report_time_start,
                                DATETIME + '_end': report_time_end}, 
                                index = [' - '.join([str(int(well_id)), well_name_dict[well_id]])])


        #add data
        df_filter = df_filter.append(df_filter_)

    if save:
        #save csv
        df_filter.to_csv(f'{save_folder}csv/historic_well_filtering.csv', index=True)
        #save table image
        df_styled = df_filter.rename(columns = {col: col.replace('_', ' ').upper() + f' ({depth_unit})'*((TVD in col)|(HD in col)) 
                                                    for col in df_filter.columns})
        df_styled.index.name = 'WELL'
        df_styled = df_styled.style.format({df_styled.columns[0]: '{0:.1f}',
                                            df_styled.columns[1]: '{0:.1f}',
                                            df_styled.columns[2]: '{0:.1f}',
                                            df_styled.columns[3]: '{0:.1f}'})\
                                    .apply(lambda x: [("background: " + color_RT) if CURRENT_WELL_NAME in x.name
                                                    else '' for v in x], axis=1)                        
        display(df_styled)
        dfi.export(df_styled, f"{save_folder}plot/historic_well_filtering.png")

    return df_filter

def build_WCR_summary_table(WCR_summary_well: pd.DataFrame, 
                            hole_diameter,
                            depth_unit = 'ft', wcr_unit = 'ft/day',
                            current_well_name = CURRENT_WELL_NAME,
                            STYLE = STYLE, color_RT = color_RT,
                            TIMEDAY =TIMEDAY, WELLIDN = WELLIDN,
                            DD = DD, WCRC = WCRC,
                            replace_dot = replace_dot,
                            save_folder = SAVE_DAILY_REPORT_FOLDER, save = True):

    #save output table as an image
    bs = str(hole_diameter).replace('.', replace_dot)

    #rename dictionary #DRILL
    rename_dict = {DD: f'DEPTH DRILLED ({depth_unit})', TIMEDAY: 'TOTAL TIME (days)', 
                   WCRC: f'WCR ({wcr_unit})', WELLIDN: 'WELL'}

    #define format for table image #DRILL
    format_dict = {f'DEPTH DRILLED ({depth_unit})': '{:,.1f}', 
                   'TOTAL TIME (days)': '{0:.3f}',
                   f'WCR ({wcr_unit})': '{0:.1f}'}

    #table title
    table_title = 'SECTION ' + decimal_fraction_str(hole_diameter) if (hole_diameter != 'all') else ''

    df_styled = WCR_summary_well.rename(columns = rename_dict)\
                                .set_index('WELL')\
                                .style\
                                .format(format_dict)\
                                .apply(lambda x: [("background: " + color_RT) if current_well_name in x.name
                                       else '' for v in x], axis=1)\
                                .set_caption(table_title)\
                                .set_table_styles([STYLE])

    display(df_styled)

    if save:
        dfi.export(df_styled, f"{save_folder}plot/wcr_summary_well_bs_{bs}.png")

    return df_styled
