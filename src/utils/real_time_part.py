#%%
#########################
#### Import Libraries ###
#########################

#linear algebra
import pandas as pd
import numpy as np

#visualization
import matplotlib.pyplot as plt

#standard naming and colors
from config import *
#custom functions
from bi_drill_utility import *

from datetime import datetime

#change directory
import os
import shutil
import time

#track time it took to execute
start_time = time.time()

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

#change location (current folder)
os.chdir(CODE_PATH)

#remove folders if exist and make folders to save data
for folder in ['plot', 'csv']:
    #remove existing
    try:
        shutil.rmtree(f'{SAVE_REAL_TIME_FOLDER}{folder}')
    except:
        print(f'{SAVE_REAL_TIME_FOLDER} folder does not exists')

    #make new folder
    os.makedirs(f'{SAVE_REAL_TIME_FOLDER}{folder}')

#########################################################################################

# Remove the leading "./" if present
cleaned_path = INPUT_FOLDER.lstrip("./")

# Split the path into components
path_components = cleaned_path.rstrip("/").split("/")

# Extract the values
if len(path_components) >= 6 and path_components[0] == "input_data":
    client, project, stream, username, scenario = path_components[1:6]
    print(f"\n========== RUN 2. MACHINE LEARNING CODE EXECUTION {formatted_datetime} | INFO | PROCESS | 100002 | {client} | {project} | {stream} | {username} | {scenario} | real_time_part_tablesupdate.sh\n")
else:
    print("The INPUT_FOLDER structure doesn't match the expected format.\n")

#########################################################################################

#------ copy to real-time separate script ------#
########################################################
######### Load standard current well drilling data #####
########################################################

#load time-depth drilling plan
df_plan = pd.read_csv(f'{INPUT_FOLDER}plan/time_depth_plan.csv')
#load directional plan
df_direct_plan = pd.read_csv(f'{INPUT_FOLDER}plan/directional_plan.csv')
#load formation_tops table: move to plan
df_form_plan = pd.read_csv(f'{INPUT_FOLDER}plan/formation_tops_plan.csv')

#build bit_sizes for dummy data
bit_sizes_max = [df_plan[BS].unique().max()]

#define formation colors
formation_palette = sns.color_palette("husl", df_form_plan[FORM].nunique())
form_colors_dict = dict(zip(df_form_plan[FORM].unique(), formation_palette))

if NO_RT:
    df_rt, df_survey_rt = make_dummy_data(bit_sizes_max)
else:
    df_rt = pd.read_csv(f'{INPUT_FOLDER}real_time/time_based_drill_current_well.csv', parse_dates=[DATETIME])
    df_survey_rt = pd.read_csv(f'{INPUT_FOLDER}real_time/official_survey_current_well.csv')
    if df_survey_rt.shape[0] <= 1:
        #add 2 rows full of zeros
        df_aux = pd.DataFrame(columns = survey_raw_cols,
                              data = np.zeros([2, len(survey_raw_cols)]),
                              index= np.arange(1,3))
        df_survey_rt = df_survey_rt.append(df_aux)

df_survey_rt.drop_duplicates(inplace=True) ### ASM changed to avoid duplocates in df_survey_rt

#load filtered historic data
df = pd.read_csv(f'{INPUT_FOLDER}database/time_based_drill.csv', parse_dates=[DATETIME])

##########################################################################
## Additional calculations for raw RT official_survey_current_well data ##
##########################################################################

######### well_id #########
df_survey_rt[WELLID] = CURRENT_WELL_ID #assign unique number

######### tvd, ns, ew, dls #########
df_survey_rt = calculate_tvd_ns_ew_dls(df_survey_rt)

######### unwrap_departure #########
df_survey_rt[UWDEP] = np.sqrt(df_survey_rt[NS]**2 + df_survey_rt[EW]**2)

############## well_section ###############
df_direct_plan[HD + '_max'] = df_direct_plan[HD].shift(-1).fillna(df_survey_rt[HD].max())
for index, row in df_direct_plan.iterrows():
    cond = (df_survey_rt[HD] >= row[HD])&(df_survey_rt[HD] <= row[HD + '_max'])
    df_survey_rt.loc[cond, WSEC] = row[WSEC]
df_direct_plan.drop(columns = [HD + '_max'], inplace = True)

############## vs ###############

##calculate VS = 'vs': vertical section: skip for now

############## ddi ###############

##calculate DDI = 'ddi': drilling difficulty index: skip for now

############## ddi_range ###############

##uncomment if ddi is computed: skip for now
# df_survey_rt_test[DDIR] = pd.cut(df_survey_rt_test[DDI],
#                                  bins = [0, 6.0, 6.4, 6.8, 10],
#                                  labels = ['0-6', '6-6.4', '6.4-6.8', '>6.8'])

### reformat in accordance with standard historic database format###
#add missing cols
miss_cols = [col for col in survey_cols if col not in df_survey_rt.columns]
for col in miss_cols:
    df_survey_rt[col] = np.nan
#reorder
df_survey_rt = df_survey_rt[survey_cols]

###########################################################################
## Additional calculations for raw RT time_based_drill_current_well data ##
###########################################################################

######### fix streamed datetime #########
df_rt[DATETIME] = pd.to_datetime(df_rt[DATETIME].apply(lambda x: str(x)[:19]), format = '%Y-%m-%d %H:%M:%S')
df_rt.drop_duplicates(DATETIME, inplace = True)
df_rt.sort_values(by = [DATETIME], inplace = True)

######### surface_rpm #########
#df_rt[RPM] = df_rt[BRPM] - df_rt[MRPM] 

######### well_id #########
df_rt[WELLID] = CURRENT_WELL_ID #assign unique number

######### cumulative_time #########
df_rt[TIME] = (df_rt[DATETIME].diff().dt.seconds/60).cumsum().fillna(0)

######### day_number #########
time_bins = np.arange(0,int(np.ceil(df_rt[TIME].max()/60/24))+1,1)
df_rt[DAYN] = pd.cut(df_rt[TIME]/60/24, 
                     bins = time_bins, 
                     labels = time_bins[1:], 
                     include_lowest = True).astype(int)

######### tvd and well_section #########
df_rt = add_tvd_well_section(df_rt, df_survey_rt)
df_rt.sort_values(by = [DATETIME], inplace = True)

######### incl, azm, dls #########
df_rt = add_survey(df_rt, df_survey_rt, feature = INC)
df_rt = add_survey(df_rt, df_survey_rt, feature = AZM)
df_rt = add_survey(df_rt, df_survey_rt, feature = DLS)

######### hole_diameter & mud_motor#########
if not(READ_BS_MM_CASING):
    df_rt[BS] = np.nan
    df_rt[MM] = np.nan

    group_col = TIME #TIME or HD
    df_plan_group = df_plan.groupby([BS,MM])[group_col].agg(['min','max'])
    for index, row in df_plan_group.iterrows():
        cond = (df_rt[group_col] >= row['min'])&(df_rt[group_col] <= row['max'])
        df_rt.loc[cond, [BS, MM]] = index

    #fill forward
    df_rt[BS].fillna(method = 'ffill', inplace = True)
    df_rt[MM].fillna(method = 'ffill', inplace = True)

    ######### casing #########
    df_rt[CASING] = np.nan

df_plan[TIME + '_max'] = df_plan[TIME].shift(-1).fillna(df_rt[TIME].max())
for index, row in df_plan.iterrows():
    cond = (df_rt[TIME] >= row[TIME])&(df_rt[TIME] <= row[TIME + '_max'])
    df_rt.loc[cond, CASING] = row[CASING]
df_plan.drop(columns = [TIME + '_max'], inplace = True)

######### mse #########
df_rt = calculate_mse(df_rt)

######### formation #########
df_rt[FORM] = np.nan

#get formation bottom by depth
df_form_plan[FORMBOT] = df_form_plan[FORMTOP].shift(-1).fillna(df_rt[HD].max())
#get formation bottom by time
df_form_plan[FORMBOTTIME] = df_form_plan[FORMTOPTIME].shift(-1).fillna(df_rt[TIME].max()) 
#for now define formation by measured_depth top
for _, row in df_form_plan.iterrows():
    #bit size boundaries
    d_min, d_max = (row[FORMTOP], row[FORMBOT])
    #add formation value
    cond = (df_rt[HD] >= d_min)&(df_rt[HD] < d_max)
    df_rt.loc[cond, FORM] = row[FORM]

######################## rig activity labels: #####################
####### rig_sub_activity rig_super_state consecutive_labels #######
#compute typical sampling time step
if DTIME_RT == None:
    dtime_rt = (df_rt[TIME].diff()*60).round().dropna().astype(int).mode()[0]
else:
    dtime_rt = DTIME_RT
#check typical data frequencies
if dtime_rt not in [1,5,10,15]:
    warnings.warn(f"Check the data frequence of real-time data: atypical value of {dtime_rt} s automatically identified")
#add value to the time step dictionary
dtime_dict = {}
dtime_dict[CURRENT_WELL_ID] = pd.Timedelta(seconds = dtime_rt)

#read rig_design parameters
df_rig = pd.read_csv(f'{INPUT_FOLDER}database/rig_design.csv')
#select rig corresponding to current well
df_rig_rt = df_rig.loc[df_rig[RIG] == RIG_NAME].squeeze()

if not(READ_CURRENT_RIG_LABELS):
    #df_rt = rig_labels_hocol(df_rt, dtime_rt, real_time = False)#old function
    df_rt = rig_labels(df_rt, df_rig_rt, RIG_NAME, dtime_rt, real_time = False)

### reformat in accordance with standard historic database format###
#add missing cols
miss_cols = [col for col in time_based_drill_cols if col not in df_rt.columns]
for col in miss_cols:
    df_rt[col] = np.nan
#reorder
df_rt = df_rt[time_based_drill_cols]

##merge_with dummy data to make work RT with limited data
df_rt_dummy, df_survey_rt_dummy = make_dummy_data(bit_sizes_max, day = df_rt[DATETIME].min())

df_rt = df_rt.append(df_rt_dummy[time_based_drill_cols]).sort_values(by=[DATETIME])
df_survey_rt = df_survey_rt.append(df_survey_rt_dummy[survey_cols]).sort_values(by=[HD])
df_rt[WELLID] = CURRENT_WELL_ID
df_survey_rt[WELLID] = CURRENT_WELL_ID

df_rt.drop_duplicates(DATETIME, inplace = True)

# df_rt[DATETIME] = pd.to_datetime(df['datetime'].apply(lambda x: str(x)[:19]), format = '%Y-%m-%d %H:%M:%S')
# df_rt = df_rt.append(df_rt_dummy[time_based_drill_cols]).sort_values(by=[DATETIME])
# df_survey_rt = df_survey_rt.append(df_survey_rt_dummy[survey_cols]).sort_values(by=[HD])
# df_rt[WELLID] = current_well
# df_survey_rt[WELLID] = current_well

########################################################
######### Add consequtive labels for current well  #####
########################################################

df_rt = conn_activity_label(df_rt, activity = 'CNX (trip)', LABEL = LABELct)
df_rt = conn_activity_label(df_rt, activity = 'CNX (drill)', LABEL = LABELcd)
df_rt = conn_activity_label_between(df_rt, activity = 'CNX (trip)', LABEL = LABELbtwn)

#####################
### add month_day ###
#####################

#add MMDD column to real-time data
df_rt[MMDD] = df_rt[DATETIME].dt.month.astype('str') \
              + '.' + df_rt[DATETIME].dt.day.astype('str').apply(lambda x: x.zfill(2))

########################
### add stand_number ###
########################

#stand number calculations
bins = np.arange(0, df_rt[HD].max() + STAND_LENGTH, STAND_LENGTH)
labels = np.arange(1,len(bins))
#define {stand labels/number: measured depth (midpoint of a stand)} dict
bins_labels_dict = dict(zip(labels, bins + STAND_LENGTH/2))
#add stand number to current well data
df_rt[STAND] = pd.cut(df_rt[HD], bins = bins, labels = labels)

########################
### add rop_mse class ##
########################
df_rt = add_rop_mse(df, df_rt)

####################################
####### add DUR = 'duration' #######
####################################

min_time = pd.Timedelta(seconds=dtime_rt)
duration_map=df_rt.groupby(LABEL).apply(lambda x: x[TIME].nunique() * min_time)
df_rt[DUR]=df_rt[LABEL].map(duration_map)         

#*** real-time data ***
print('\n*** real-time data ***\n')
display(df_rt.head())
# #test: add missing values in meassured_depth
# df_rt.loc[np.random.choice(df_rt.loc[df_rt[MMDD] == '10.22'].index,size=100), HD] = NULL_VALUE
# #test: add missing values in bit_depth
# df_rt.loc[np.random.choice(df_rt.loc[df_rt[MMDD] == '10.22'].index,size=100), BD] = NULL_VALUE
####################################################################
### Compute tripping and connection time tables for current well ###
####################################################################

#weight to weight table for current well
df_wt_wt_rt = compute_df_wt_wt(df_rt)
#tripping table for current well
df_trip_rt = compute_df_trip(df_rt)
#connection time drill table for current well
df_conn_drill_rt = compute_df_conn(df_rt, compute_col = CONNDTIME, activity = 'CNX (drill)', dtime_dict = dtime_dict)
#connection time trip table for current well
df_conn_trip_rt = compute_df_conn(df_rt, compute_col = CONNTTIME, activity = 'CNX (trip)', dtime_dict = dtime_dict)

#save complete real-time data tables: append by row at a time
df_rt.to_csv(f"{SAVE_REAL_TIME_FOLDER}time_based_drill_current_well_out.csv")
df_survey_rt.to_csv(f"{SAVE_REAL_TIME_FOLDER}official_survey_current_well_out.csv")

#add read from the plan and save tables for all bit sizes
current_bit_size = df_rt[BS].iloc[-1] 
include_bits = df_rt[BS].unique()
sorted(include_bits, reverse = True)
bit_sizes = ['all'] + list(include_bits)
#remove nan in bit_sizes:
bit_sizes = [bit for bit in bit_sizes if str(bit) != 'nan']

#***********************************************************************************#
#**********************************  Overview tab  *********************************#
#***********************************************************************************#

########## ACTIVITY DISTRIBUTION: TIME PLOT ######

for super_activity in ['all', 'DRILL', 'TRIP']:
    for hole_diameter in bit_sizes:
        _ = rig_activity_summary(df_rt, WELLID,
                                 super_activity, hole_diameter,
                                 WELLS_SELECT_ID, CURRENT_WELL_ID, dtime_dict, add_naming = '_rt',
                                 save_folder = SAVE_REAL_TIME_FOLDER)

########## WELL PERFORMANCE: WCR PLOT ############

df_wcr_plan = pd.read_csv(f'{SAVE_FOLDER}csv/wcr_plan.csv')

for hole_diameter in bit_sizes:
    compute_WCR(df_rt, df_wcr_plan, hole_diameter, dtime_dict, well_name_dict,
                save_folder = SAVE_REAL_TIME_FOLDER)


#***********************************************************************************#
#**********************************  Drilling tab  *********************************#
#***********************************************************************************#

super_activity = 'DRILL'

#between connection and connection times with kpi colors
df_wt_wt_rt_kpi = get_kpi_color(df_rt, df_wt_wt_rt, LABELcd, WTWT, WTWT,
                                name = 'weight_weight', kpi_ascend = False, 
                                read_folder = SAVE_FOLDER,
                                save_folder = SAVE_REAL_TIME_FOLDER)

df_conn_drill_rt_kpi =  get_kpi_color(df_rt, df_conn_drill_rt, LABEL, CONNDTIME, CONNDTIME,
                                      name = 'conn_drill', kpi_ascend = False,
                                      read_folder = SAVE_FOLDER,
                                      save_folder = SAVE_REAL_TIME_FOLDER)

df_conn_trip_rt_kpi = get_kpi_color(df_rt, df_conn_trip_rt, LABEL, CONNTTIME, CONNTTIME,
                                    name = 'conn_trip', kpi_ascend = False,
                                    read_folder = SAVE_FOLDER,
                                    save_folder = SAVE_REAL_TIME_FOLDER)

df_plot_lims_drill = pd.read_csv(f'{SAVE_FOLDER}csv/{super_activity.lower()}_plot_lims.csv')

y_col = df_plot_lims_drill['y_col'].iloc[0]
ylims = df_plot_lims_drill['y_lims'].values
units_conversion = 1/60 if (y_col == TIME) else 1
y_unit = ' (ft)' if y_col == HD else ' (hr)'

#read selected R1 and R2
df_R = pd.read_csv(f'{SAVE_FOLDER}csv/ref1_ref2_highlight_{y_col}.csv')
R1, R2 = df_R[MMDD].astype(str).values

#plot drilling activity per stand
df_per_stand = drill_per_stand(df_rt, STAND_LENGTH,
                           y_col, y_unit, units_conversion, 
                           ylims, [R1, R2], bins_labels_dict,
                           save_folder = SAVE_REAL_TIME_FOLDER)

### update every day:
####################################
#####  ACTIVITY DISTRIBUTION  ######
####################################

df_time_dist_drill = rig_activity_summary(df_rt, MMDD,
                                    super_activity, 'all',
                                    WELLS_SELECT_ID, CURRENT_WELL_ID, dtime_dict, 
                                    refs = [R1, R2], add_naming = '_per_day',
                                    save_folder = SAVE_REAL_TIME_FOLDER)

###################################
########  DEPTH DRILLED  ##########
###################################

#compute performance table
df_perform_drill = compute_plot_depth(df_rt[(df_rt[HD] != NULL_VALUE)], ['ROTATE', 'SLIDE'], super_activity, 
                                      dtime_dict, [R1, R2], save_folder = SAVE_REAL_TIME_FOLDER)

#***********************************************************************************#
#**********************************  Tripping tab  *********************************#
#***********************************************************************************#

super_activity = 'TRIP'

df_plot_lims_drill = pd.read_csv(f'{SAVE_FOLDER}csv/{super_activity.lower()}_plot_lims.csv')

y_col = df_plot_lims_drill['y_col'].iloc[0]
ylims = df_plot_lims_drill['y_lims'].values
units_conversion = 1/60 if (y_col == TIME) else 1
y_unit = ' (ft)' if y_col == HD else ' (hr)'


df_R = pd.read_csv(f'{SAVE_FOLDER}csv/ref1_ref2_highlight_{y_col}.csv')
R1, R2 = df_R[MMDD].astype(str).values

#compute pipe speed and pipe movement
df_trip_rt_kpi = compute_pipe_speed_envelope(df_rt, df_trip_rt, LABELct, 'pipe_speed', 
                                             save_folder = SAVE_REAL_TIME_FOLDER)

#####################################
###########  ACTIVITY  ##############
#####################################

#compute circulating time
df_circulate = compute_trip_activity(df_rt, CIRCTIME, ['CIR (static)'], save_folder = SAVE_REAL_TIME_FOLDER)

#compute washing time
df_wash = compute_trip_activity(df_rt, WASHTIME, ['WASH IN', 'WASH OUT'], save_folder = SAVE_REAL_TIME_FOLDER)

#compute reaming time
df_ream = compute_trip_activity(df_rt, REAMTIME, ['REAM UP', 'REAM DOWN'], save_folder = SAVE_REAL_TIME_FOLDER)

### update every day:
####################################
#####  ACTIVITY DISTRIBUTION  ######
####################################

df_time_dist_trip = rig_activity_summary(df_rt, MMDD,
                                    super_activity, 'all',
                                    WELLS_SELECT_ID, CURRENT_WELL_ID, dtime_dict, 
                                    refs = [R1, R2], add_naming = '_per_day', save_folder = SAVE_REAL_TIME_FOLDER)

####################################
#########  DEPTH DRILLED  ##########
####################################

#merge time-based drill data and trip calculations
df_rt_pipe = df_rt.merge(df_trip_rt.reset_index()[[LABELct, PIPESP, RSUBACT]]\
                                   .rename(columns={RSUBACT:RSUBACT+'_trip'}),
                         how = 'left', left_on = LABELct, right_on = LABELct)

#compute performance table 
df_perform_trip = compute_plot_depth(df_rt_pipe[df_rt_pipe[BD] != NULL_VALUE], ['TRIP IN', 'TRIP OUT'], super_activity, 
                                dtime_dict, [R1, R2], save_folder = SAVE_REAL_TIME_FOLDER)

###########################################
#########  UPDATE RT well KPIs  ###########
###########################################

for hole_diameter in bit_sizes:
    kpi_boxplot(df_rt, df_wt_wt_rt, df_trip_rt, 
                df_conn_drill_rt, df_conn_trip_rt, 
                hole_diameter, [CURRENT_WELL_ID], CURRENT_WELL_ID, 
                save_folder = SAVE_REAL_TIME_FOLDER)

end_time = time.time()
print(f'\n\n*** Time it took to run bi_drill_main.py is {end_time-start_time} s.***')

######################################
#########  INPUT -> OUTPUT  ##########
#########  TABLES MAPPING   ##########
######################################

# input: df_rt, df_survey_rt (WITSML format)

## Overview update output
# df_rt = time_based_drill_current_well.csv
# df_survey_rt = official_survey_current_well.csv
# rig_activity_summary() update all time_dist_act_hours_all_bs_{bs}_rt.csv and 
# time_dist_act_pct_all_bs_{bs}_rt.csv for current well only(one row)
# compute_WCR() update all wcr_summary_well_bs_{bs}.csv

## Drilling update output
# df_bound_rop = rop_best_worst_avg_bs_all.csv
# df_wt_wt_rt_kpi = weight_weight_time_current_well.csv
# df_conn_drill_rt_kpi = conn_drill_time_current_well.csv
# df_conn_trip_rt_kpi = conn_trip_time_current_well.csv
# df_per_stand = rotate_slide_depth_per_stand.csv
# df_time_dist_drill: rig_activity_summary() saves time_dist_act_hours_drill_bs_all_per_day.csv and time_dist_act_pct_drill_bs_all_per_day.csv
# df_perform_drill = drill_performance_per_day.csv

## Tripping update output
# df_trip_rt_kpi = pipe_speed_time_current_well.csv
# df_circulate = circulating_time.csv
# df_wash = washing_time.csv
# df_ream = reaming_time.csv
# df_time_dist_trip: rig_activity_summary() saves time_dist_act_hours_trip_bs_all_per_day.csv and time_dist_act_pct_trip_bs_all_per_day.csv
# df_perform_trip = trip_performance_per_day.csv
# %%