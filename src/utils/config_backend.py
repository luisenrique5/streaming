import pandas as pd
import os

# ==== select postjob reporting option ===================

#set to False for daily report generation/dashboard execution
POSTJOB = False; from utils_backend import * #Set True to build postjob report

# ==== operator, current well and historic wells ===================

# set current operator name
OPERATOR = 'SIERRACOL' #THIS IS ONLY FOR FOLDER NAMING PURPOSES

#RT/candidate well
# CURRENT_WELL_NAME = 'RT'

HOLE_DIAMETERS = [12.25,8.5,6.125]#[12.25, 8.75, 6.75] #[12.25, 8.75, 6.75] #[12.25, 8.75] #[12.25, 8.5, 6.125] #connect with dashboard selection
WELL_TYPE = 'H' #connect with dashboard selection: ['J', 'S', 'H', 'V', 'any']
DLS_RANGE = ["any","any","any","0"] #connect to dashboard
INC_RANGE = ["any","any","any","0","0"] #connect to dashboard

#complete wells
# WELLS_SELECT_NAME = ['La Yuca-249','La Yuca-297', 'La Yuca-298'] #TO BE UPDATED WITH SELECTED WELLS CONFIGURATION FROM DASHBOARD

#fix wells_select_name
WELLS_SELECT_NAME = [well for well in WELLS_SELECT_NAME if well != CURRENT_WELL_NAME]

# ====  PATHS ===================

# Get the current working directory
CODE_PATH = os.getcwd() + '/'

# set output and input paths
# SAVE_FOLDER = CODE_PATH + f'../data/bi_drill_data/output_data/{OPERATOR}/'
# INPUT_FOLDER = CODE_PATH + f'../data/bi_drill_data/input_data/{OPERATOR}/'

# set output path for real time calculations update
SAVE_REAL_TIME_FOLDER = SAVE_FOLDER + 'real_time_update/'

# ====  DAILY REPORT PATHS AND PARAMETERS ===================

if POSTJOB:
    report_folder = 'postjob_report/'
    #enter wheather save the xlsx or not
    SAVE_XLSX = 1 #input('Enter 1 (save xlsx data) or 0 (do not save xlsx data):')
    KEEP_RT_IN_HISTORIC = True # if True keeps current RT well in historic_drill.xlsx
    #otherwise if False renames current well to RT and reassigns well id to 999.
else:
    report_folder = 'daily_report/'
    #enter wheather save the xlsx or not
    SAVE_XLSX = 0 #input('Enter 1 (save xlsx data) or 0 (do not save xlsx data):')
    KEEP_RT_IN_HISTORIC = False # if True keeps current RT well in historic_drill.xlsx
    #otherwise if False renames current well to RT and reassigns well id to 999.

# set output paths for report data
SAVE_DAILY_REPORT_FOLDER = SAVE_FOLDER + report_folder
DAILY_REPORT_PLOTS_PATH =  SAVE_DAILY_REPORT_FOLDER.strip('.') + 'plot/'

# ==== additional parameters: RT related ===================

NO_RT = False #set False when RT data is present, True - when missing for testing
DTIME_RT = None #data density of RT well in seconds, None - auto detects data frequency

NULL_VALUE = -999.25

READ_CURRENT_RIG_LABELS = False  # if False (default) - computes from plan for RT, True - reads from processed data
READ_BS_MM_CASING = False #if False (default) - computes from plan for RT, True - reads from processed data

if POSTJOB | NO_RT:
    READ_CURRENT_RIG_LABELS = True
    READ_BS_MM_CASING = True

# ==== round and save parameters ===================

ROUND_NDIGITS = 3
REPLACE_DOT = 'p'
#set table style
STYLE = {'selector': 'caption','props': [('font-weight', 'bold'),('color', 'k')]}

# ======= STANDARD FEATURE NAMES =============

#Well info
WELLID = 'well_id'
WELL = 'well_name'
WELLTYPE = 'well_type'
WELLIDN = 'well_id_name'
RIG = 'rig'
FIELD = 'field'
SPUD = 'spud_date'

#Drilling parameters
BS = 'hole_diameter'
HD = 'measured_depth'
BD = 'bit_depth'
BH = 'block_height'
WOB = 'wob'
HL = 'hook_load'
GPM = 'flow_rate'
RPM = 'surface_rpm'
MRPM = 'motor_rpm'
BRPM = 'bit_rpm'
TQ = 'torque'
ROP = 'rop'
SPP =  'spp'
DIFFP = 'diff_pressure'
PVT = 'pit_volume'
APRESS = 'annular_pressure'
HSI = 'hsi'

#Logging curves
GR = 'gamma'
GRD = 'gamma_depth'

#Formation tops
FORM = 'formation'
FORMTOP = 'formation_top_depth'
FORMBOT = 'formation_bottom_depth'
FORMTOPTIME = 'formation_top_time'
FORMBOTTIME = 'formation_bottom_time'

#Time
DATETIME = 'datetime'
DATETIMED = DATETIME + '_day'
TIME = 'cumulative_time' #minutes
TIMEDAY = 'cumulative_time_days' #days
DAYN = 'day_number'
MMDD = 'month_day'
BITIMEFROM = 'bi_datetime_from'
BITIMETO = 'bi_datetime_to'
BISECFROM = 'bi_sections_datetime_from'
BISECTO = 'bi_sections_datetime_to'
CASINGTIMEFROM = 'casing_datetime_from'
CASINGTIMETO = 'casing_datetime_to'

#Consecutive labels
LABEL = 'consecutive_labels'
LABELct = LABEL + '_conn_trip'
LABELcd = LABEL + '_conn_drill'
LABELbtwn = LABEL + '_btwn'

#Surveys
DDI = 'ddi' #drilling difficulty index
TVD = 'tvd'
AZM = 'azm'
INC = 'incl'
NS = 'ns'
EW = 'ew'
VS = 'vs'
DLS = 'dls'#(deg/100ft)
WSEC = 'well_section'
DDIR = 'ddi_range'
UWDEP = 'unwrap_departure'
TORT = 'tortuosity'
ATORT = 'abs_tortuosity'
DLSR = 'dls_range'
INCR = 'inc_range'

#Casing and motor
CASING = 'casing'
MM = 'mud_motor'

#Calculate
MSE = 'mse'
WTWT = 'weight_to_weight' #min
PIPESP = 'pipe_speed' #(ft/hr)
PIPEMV = 'pipe_movement' #min
TRIPSPEED = 'trip_speed' #(ft/hr)

CONNDTIME = 'connection_time_drill' #min
CONNTTIME = 'connection_time_trip' #min
CONNTIME = 'connection_time' #min
DD = 'depth_drilled'
WCRC = 'wcr'

#Enovate rig activities columns
RSUPER = 'rig_super_state' #Enovate Rig Super State code
RSUBACT = 'rig_sub_activity' #Enovate Rig Sub Activity code

#Auxiliary
dBD = 'delta_' + BD
dBH = 'delta_' + BH
dTIME = 'delta_' + TIME
DTIME = 'delta_' + TIME + '_days'
DUR = 'duration' #s

PLAN = 'plan'

DEPTHTYPE = 'depth_analysis_type'
DEPTHTVDTO = 'depth_analysis_tvd_to'
COLOR = 'color'

#drill tab
STAND = 'stand_number'
ROP_MSE = 'rop_mse'
KPIC = 'kpi_color'

#trip tab
CIRCTIME = 'circulating_time'
WASHTIME = 'washing_time'
REAMTIME = 'reaming_time'

#report
GROUPWELL = 'group_wells'
AVG = 'AVG'
DIFF = 'DIFFERENCE'
CLIENT = 'client'

# ======= STANDARD DATA COLUMN SETS =============

rig_activity_order = ['ROTATE','SLIDE','REAM UP', 'REAM DOWN',
                      'CNX (drill)','CNX (trip)','TRIP IN','TRIP OUT',
                      'WASH IN', 'WASH OUT','PUMP ON/OFF', 'CIR (static)',
                      'STATIC', 'OUT OF HOLE', 'OTHER', 'NULL']

rig_activity_drill_order = ['ROTATE','SLIDE','REAM UP', 'REAM DOWN', 'CNX (drill)',
                            'WASH IN', 'WASH OUT','PUMP ON/OFF', 'CIR (static)',
                            'STATIC', 'OTHER', 'NULL']#'OUT OF HOLE',

rig_activity_trip_order = ['REAM UP', 'REAM DOWN', 'CNX (trip)',
                           'TRIP IN','TRIP OUT',
                           'WASH IN', 'WASH OUT', 'CIR (static)',
                           'STATIC', 'OTHER', 'NULL']#'OUT OF HOLE',

#standard historic time_based_drill table columns
time_based_drill_cols = [WELLID, DATETIME, TIME, DAYN, HD, TVD, INC, AZM, DLS, WSEC, BD, BS, FORM,
                        BH, ROP, WOB, HL, GPM, PVT, DIFFP, SPP, APRESS, TQ,
                        RPM, MRPM, BRPM, MSE, MM, CASING, RSUPER, RSUBACT, LABEL]

#WITSML time_based_drill_current_well table columns
time_based_drill_rt_raw_cols = [DATETIME, HD, BD, BH, ROP, WOB, HL, GPM,
                                SPP, TQ, RPM, MRPM, BRPM]

#standatd historic official_survey table columns
survey_cols = [WELLID, HD, INC, AZM, TVD, DLS, NS, EW, TORT, ATORT, DDI, UWDEP, VS, WSEC, DDIR]

#WITSML official_survey_current_well table columns
survey_raw_cols = [HD, INC, AZM]

#directional plan columns
directional_plan_cols = [HD, TVD, INC, AZM, VS, NS, EW, DLS, UWDEP, FORM, WSEC]

well_general_cols = [WELLID, WELL, WELLTYPE, CLIENT, RIG, FIELD,
                     SPUD, BITIMEFROM, BITIMETO, 'total_time',
                     'total_depth', 'location', 'latitude', 'longitude']

rig_design_cols = ['rig_id', 'rig', 'client', 'stand_length', 'depth_onb_thr', 'depth_conn_thr',
                'depth_conn_start', 'bd_conn', 'depth_super_thr', 'depth_ooh_thr', 'depth_trip_thr',
                'hl_conn_drill_thr', 'hl_conn_drill1_thr', 'hl_conn_trip_thr', 'hl_conn_trip1_thr',
                'hl_null_thr', 'gpm_thr', 'rpm_thr', 'spp_thr', 'spp_stat_thr', 'wob_thr', 'rpm_rot_thr',
                'rpm_stat_thr', 'n_tseq_static', 'n_tseq_trip', 'n_tseq_circ', 'filter_dict',
                'filter_dict_1', 'filter_dict_5', 'filter_dict_10', 'filter_dict_15']

formation_tops_plan_cols =['well_name', 'formation', 'formation_top_depth', 'formation_top_tvd',
                        'formation_top_time', 'formation_bottom_depth', 'formation_bottom_time']

# ======= STANDARD COLORS =============

color_overperf = '#9F8AD9'
color_underperf = '#ce1140'
color_neutral = '#2A2D40'
color_historic = '#FCE255'
color_RT = '#97FF8F'
color_purple = '#9F8AD9'

daily_report_color_RT = '#558A2E' #dark green

# ======= STANDARD COLORS SETS =============

#kpi colors in needed sequence
kpi_colors = [color_underperf, color_underperf,
                color_neutral, color_overperf, color_overperf]

#kpi colors for drilling and trippign performance
kpi_dict = {0: color_overperf, 1: color_historic, 2: color_underperf}

#performance colors in needed sequence
perform_colors = [color_underperf, color_historic,
                  color_overperf, color_neutral, color_RT]

#mse rop performance colors
mse_rop_colors = ['#583BD4', '#9F8AD9','#F74B76', '#CE1140']
mse_rop_colors_map = dict(zip([0,1,2,3], mse_rop_colors))

#ddi range colors
dd_range_color_dict = {'0-6': '#97FF8F', '6-6.4': '#FFF500',
                        '6.4-6.8': '#F5AB24', '>6.8' :'#CE1140'}

#well difficulty reach and depth analysis background colors
well_diff_colors = ['#583BD4', '#E3CB44', '#841F38', '#2A2D40']

rig_activity_color_dict =  {'ROTATE': '#FCE255',
                            'SLIDE': '#FDA300',
                            'REAM UP': '#C7B037',
                            'REAM DOWN': '#8C6C31',
                            'CNX (drill)': '#F57524',
                            'CNX (trip)': '#E1C220',
                            'TRIP IN': '#9CCC66',
                            'TRIP OUT': '#558A2E',
                            'WASH IN': '#80DDFF',
                            'WASH OUT': '#1EA5D5',
                            'PUMP ON/OFF': '#167A9D',
                            'CIR (static)': '#305580',
                            'STATIC': '#9DAAB9',
                            'OUT OF HOLE': '#727272',
                            'OTHER': '#52587D',
                            'NULL': '#C8C8C8',
                            'TOTAL': None}

act_backgr_dict = {key: ('background-color: ' + str(value))
                   for key,value in rig_activity_color_dict.items()}

# ======= STANDARD RIG ACTIVITIES =============

### Super activities mapping
super_state = {0:'OTHER', 1:'DRILL', 7:'TRIP', 5:'OUT OF HOLE'}
#in aggrement with Pason, that also has {4:'Transition', 6:'Pre-spud'}
super_state_swap = {value:key for key, value in super_state.items()}

### Subactivities mapping
sub_activity_state = {0:'OTHER', 1:'ROTATE', 2: 'SLIDE', 3: 'PUMP ON/OFF',
                      5:'CNX (drill)', 6: 'CNX (trip)',
                      7: 'STATIC', 8: 'TRIP IN', 9: 'TRIP OUT', 4:'NULL',
                      11:'CIR (static)', 12: 'REAM UP', 13: 'REAM DOWN',
                      14: 'WASH IN', 15: 'WASH OUT', 16:'OUT OF HOLE'}
sub_activity_state_swap = {value:key for key, value in sub_activity_state.items()}

# ======= READ DATA AND BUILD WELL DICT =============

#load well_general table
df_wells = pd.read_csv(f'{INPUT_FOLDER}database/well_general.csv')

#construct well id dictionary {well_id: well_name}
well_name_dict = {row[WELLID]: row[WELL] for _, row in df_wells.iterrows()}
well_name_dict_swap = {well: wellid for wellid, well in well_name_dict.items()}

#current well name: need to connect!!
CURRENT_WELL_ID =  well_name_dict_swap[CURRENT_WELL_NAME]
WELLS_SELECT_ID = [well_name_dict_swap[well] for well in WELLS_SELECT_NAME]
RIG_NAME = df_wells.loc[df_wells[WELLID] == CURRENT_WELL_ID, RIG].values[0]

#load rig_design table
df_rig = pd.read_csv(f'{INPUT_FOLDER}database/rig_design.csv')
STAND_LENGTH = df_rig.loc[df_rig[RIG] == RIG_NAME, 'stand_length'].values[0]
# %%
