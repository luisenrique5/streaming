import pandas as pd
import numpy as np
import numbers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper", font_scale = 1.8)
from IPython.display import display
import re
import os
import sys
import ast
import subprocess
import time
from scipy.interpolate import interp1d
import dataframe_image as dfi
import warnings
from src.entities.config_backend import ConfigBackend
# Ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.8)

class BI_Drill_Utility:
    def __init__(self, base_key: str, redis_client=None):
        self.config_backend = ConfigBackend(base_key, redis_client)
        self.config_backend.load_well_general()
        self.config_backend.load_rig_design()
        pd.set_option('display.precision', self.config_backend.ROUND_NDIGITS)
        self.BS = self.config_backend.BS
        self.FORM = self.config_backend.FORM
        self.NO_RT = self.config_backend.NO_RT
        self.DATETIME = self.config_backend.DATETIME
        self.CURRENT_WELL_ID = self.config_backend.CURRENT_WELL_ID
        self.WELLID = self.config_backend.WELLID
        self.NS = self.config_backend.NS
        self.EW = self.config_backend.EW
        self.UWDEP = self.config_backend.UWDEP
        self.HD = self.config_backend.HD
        self.WSEC = self.config_backend.WSEC
        self.survey_cols = self.config_backend.survey_cols
        self.TIME = self.config_backend.TIME
        self.DATETIME = self.config_backend.DATETIME
        self.DAYN = self.config_backend.DAYN
        self.INC = self.config_backend.INC
        self.AZM = self.config_backend.AZM
        self.DLS = self.config_backend.DLS      
        self.READ_BS_MM_CASING = self.config_backend.READ_BS_MM_CASING
        self.MM = self.config_backend.MM
        self.MMDD = self.config_backend.MMDD
        self.CASING = self.config_backend.CASING
        self.FORMTOP = self.config_backend.FORMTOP
        self.FORMBOT = self.config_backend.FORMBOT
        self.FORMBOTTIME = self.config_backend.FORMBOTTIME
        self.FORMTOPTIME = self.config_backend.FORMTOPTIME
        self.DTIME_RT = self.config_backend.DTIME_RT
        self.RIG = self.config_backend.RIG
        self.RIG_NAME = self.config_backend.RIG_NAME
        self.READ_CURRENT_RIG_LABELS = self.config_backend.READ_CURRENT_RIG_LABELS
        self.time_based_drill_cols = self.config_backend.time_based_drill_cols
        self.LABEL = self.config_backend.LABEL
        self.LABELct = self.config_backend.LABELct
        self.LABELcd = self.config_backend.LABELcd
        self.LABELbtwn = self.config_backend.LABELbtwn
        self.STAND_LENGTH = self.config_backend.STAND_LENGTH
        self.STAND = self.config_backend.STAND
        self.DUR = self.config_backend.DUR  
        self.CONNDTIME = self.config_backend.CONNDTIME
        self.CONNTTIME = self.config_backend.CONNTTIME
        self.WELLS_SELECT_ID = self.config_backend.WELLS_SELECT_ID
        self.SAVE_FOLDER = self.config_backend.SAVE_FOLDER
        self.WTWT = self.config_backend.WTWT
     
    def decimal_fraction_str(self, num):

        if isinstance(num, numbers.Number):

            integer_part = int(num // 1)
            decimal_part = (num % 1)
            ratio = decimal_part.as_integer_ratio()

            return str(integer_part) + '-' + str(ratio[0]) + '/'  \
                + str(ratio[1]) + "\""

        else:

            return num
        
    
    def convert_time_decimal_mmss(self, time_decimal:float):

        if str(time_decimal) != 'nan':

            minutes = int(time_decimal)
            seconds = int(round((time_decimal*60) % 60, 0))

            return str(minutes).zfill(2) + ':' + str(seconds).zfill(2)

        else:
            return np.nan    
    
        
    def make_dummy_data(self, bit_sizes, 
                        day = None, 
                        current_well = None, 
                        dtime_rt = None,
                        WELLID = None, DATETIME = None,
                        BS = None, RSUPER = None, RSUBACT = None,
                        super_state_swap = None,
                        sub_activity_state_swap = None,
                        time_based_drill_cols = None,
                        survey_cols = None):
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

        if current_well is None:
            current_well = self.config_backend.CURRENT_WELL_ID  
        
        if dtime_rt is None:
            dtime_rt = self.config_backend.DTIME_RT    
        
        if WELLID is None:
            WELLID = self.config_backend.WELLID
            
        if DATETIME is None:
            DATETIME = self.config_backend.DATETIME    
            
        if BS is None:
            BS = self.config_backend.BS    
            
        if RSUPER is None:
            RSUPER = self.config_backend.RSUPER  
        
        if RSUBACT is None:
            RSUBACT = self.config_backend.RSUBACT     
        
        if super_state_swap is None:
            super_state_swap = self.config_backend.super_state_swap  
            
        if sub_activity_state_swap is None:
            sub_activity_state_swap = self.config_backend.sub_activity_state_swap
            
        if time_based_drill_cols is None:
            time_based_drill_cols = self.config_backend.time_based_drill_cols      
        
        if survey_cols is None:
            survey_cols = self.config_backend.survey_cols        
        
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

    #sub function to plot kpi
    def kpi_boxplot_sub(self, s: pd.Series, 
                        q50_current: float,
                        title: str, ax,  
                        kpi_colors = None, 
                        color_historic = None, 
                        color_RT = None) -> pd.DataFrame:
        
        if kpi_colors is None:
            kpi_colors = self.config_backend.kpi_colors
        
        if color_historic is None:
            color_historic = self.config_backend.color_historic
        
        if color_RT is None:
            color_RT = self.config_backend.color_RT        

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
                xticks_labels = [self.convert_time_decimal_mmss(label) for label in quantiles.values]
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

    
    # #function to compute TVD, NS, EW, DLS for official survey
    def calculate_tvd_ns_ew_dls(self, df_survey: pd.DataFrame,
                                HD = None, INC = None, AZM = None,
                                TVD = None, NS = None, EW = None, DLS = None) -> pd.DataFrame:
        """This function calculates tvd, ns, ew, dls using minimum curvature methon.
        Here df_survey is official survey data, 
        HD, INC, AZM, TVD, NS, EW, DLS represent standard column names, 
        see standard naming and ada_naming_convention.xlsx file for descriptions.
        This fuction requires presense of HD (ft), INC (deg), AZM (deg) 
        in passed dataframe and it returns dataframe with computed
        TVD (ft), NS (ft), EW (ft) and DLS (degree/100ft) columns (default units)."""
        
        if HD is None:
            HD = self.config_backend.HD
        
        if INC is None:
            INC = self.config_backend.INC
        
        if AZM is None:
            AZM = self.config_backend.AZM
        
        if TVD is None:
            TVD = self.config_backend.TVD
        
        if NS is None:
            NS = self.config_backend.NS
         
        if EW is None:
            EW = self.config_backend.EW
        
        if DLS is None:
            DLS = self.config_backend.DLS                       
        
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
    def calculate_mse(self, df_input: pd.DataFrame, 
                    drill_cols = None, 
                    MSE = None) -> pd.DataFrame:
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
        
        if drill_cols is None:
            drill_cols = [self.config_backend.TQ, self.config_backend.RPM, self.config_backend.ROP, self.config_backend.WOB, self.config_backend.BS]
        
        if MSE is None:
            MSE = self.config_backend.MSE    

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
    
    def add_tvd_well_section(self, df_input: pd.DataFrame, 
                df_survey_input: pd.DataFrame, 
                DATETIME = None, 
                HD = None, TVD = None,
                WSEC = None) -> pd.DataFrame:
        
        if DATETIME is None:
            DATETIME = self.config_backend.DATETIME
        
        if HD is None:
            HD = self.config_backend.HD
            
        if TVD is None:
            TVD = self.config_backend.TVD
        
        if WSEC is None:
            WSEC = self.config_backend.WSEC
                        
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
    
    def add_survey(self, df_input: pd.DataFrame, df_survey_input: pd.DataFrame, 
                feature: str, 
                HD = None, TVD = None, 
                DATETIME = None) -> pd.DataFrame:
        
        if HD is None:
            HD = self.config_backend.HD
        
        if TVD is None:
            TVD = self.config_backend.TVD
        
        if DATETIME is None:
            DATETIME = self.config_backend.DATETIME        
        
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
    
    def comp_duration(self, df_input: pd.DataFrame, 
                  activity_col: str, dtime: float, 
                  time_col = None, LABEL = None, DUR = None) -> pd.DataFrame:
        
        if time_col is None:
            time_col = self.config_backend.TIME
        
        if LABEL is None:
            LABEL = self.config_backend.LABEL
        
        if DUR is None:
            DUR = self.config_backend.DUR
                    
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
    
    def rig_labels(self, df_input: pd.DataFrame, df_rig: pd.Series,
                rig_name: str, dtime: float, real_time: bool,
                use_filter_dict = True,
                operator = None,
                hd_diff_apply = True, wob_thr_apply = False, show_activity_names = False,
                RSUPER = None, RSUBACT = None,
                TIME = None, HD = None, BD = None, HL = None, 
                RPM = None, MRPM = None, BRPM = None,
                GPM = None, SPP = None, WOB = None, 
                super_state = None, 
                sub_activity_state = None, 
                verbose = False) -> pd.DataFrame:
        
        if operator is None:
            operator = self.config_backend.OPERATOR
        
        if RSUPER is None:
            RSUPER = self.config_backend.RSUPER
            
        if RSUBACT is None:
            RSUBACT = self.config_backend.RSUBACT
        
        if TIME is None:
            TIME = self.config_backend.TIME
        
        if HD is None:
            HD = self.config_backend.HD
        
        if BD is None:
            BD = self.config_backend.BD
        
        if HL is None:
            HL = self.config_backend.HL
        
        if RPM is None:
            RPM = self.config_backend.RPM
        
        if MRPM is None:
            MRPM = self.config_backend.MRPM
        
        if BRPM is None:
            BRPM = self.config_backend.BRPM
        
        if GPM is None:
            GPM = self.config_backend.GPM
        
        if SPP is None:
            SPP = self.config_backend.SPP
            
        if WOB is None:
            WOB = self.config_backend.WOB
        
        if super_state is None:
            super_state = self.config_backend.super_state
        
        if sub_activity_state is None:
            sub_activity_state = self.config_backend.sub_activity_state                                                    
        
                
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
        BHch = self.config_backend.BH +'_change'
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
        df[BHch] = np.sign(df[self.config_backend.BH].diff(1).fillna(0))
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
        df.loc[df[self.config_backend.CASING] == 1, RSUBACT] = 0
        
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

        df.loc[(df[RSUB] == sub_state_swap['ROTATE']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['ROTATE']
        df.loc[(df[RSUB] == sub_state_swap['SLIDE']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['SLIDE']
        df.loc[(df[RSUB] == sub_state_swap['PUMP OFF']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['PUMP ON/OFF']
        df.loc[(df[RSUB] == sub_state_swap['PUMP ON']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['PUMP ON/OFF']
        df.loc[(df[RSUB] == sub_state_swap['CNX (trip)']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['CNX (trip)']
        df.loc[(df[RSUB] == sub_state_swap['CNX (drill)']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['CNX (drill)']
        df.loc[(df[RSUB] == sub_state_swap['STATIC']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['STATIC']
        df.loc[(df[RSUB] == sub_state_swap['NULL']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['NULL']
        df.loc[(df[RSUB] == sub_state_swap['TRIP IN']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['TRIP IN']
        df.loc[(df[RSUB] == sub_state_swap['TRIP OUT']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['TRIP OUT']

        df.loc[(df[RSUBCIR] == sub_cir_state_swap['REAM UP']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['REAM UP']
        df.loc[(df[RSUBCIR] == sub_cir_state_swap['REAM DOWN']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['REAM DOWN']
        df.loc[(df[RSUBCIR] == sub_cir_state_swap['WASH IN']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['WASH IN']
        df.loc[(df[RSUBCIR] == sub_cir_state_swap['WASH OUT']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['WASH OUT']
        df.loc[(df[RSUBCIRST] == sub_cir_static_state_swap['CIR (static)']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['CIR (static)']

        d = df[RSUBACTIVITY].value_counts(normalize=True)
        d.index = d.index.map(sub_activity_state) 

        #display data distribution
        if verbose:
            print('\nSub States final data distribution:\n', d)

        ####################
        ### Super States ###
        ####################
        
        #Super States: Drilling
        df.loc[(df[HD]-df[BD]) <= depth_super_thr, RSUPER] = self.config_backend.super_state_swap['DRILL']
        #Super States: Tripping
        df.loc[(df[HD]-df[BD]) > depth_super_thr, RSUPER] = self.config_backend.super_state_swap['TRIP']
        #Super States: Out of Hole
        df.loc[(df[BD] < depth_ooh_thr), RSUPER] = self.config_backend.super_state_swap['OUT OF HOLE']

        #fix mislabeled Sub states for Out of Hole super state
        df.loc[df[RSUPER]==self.config_backend.super_state_swap['OUT OF HOLE'], RSUB] = sub_state_swap['OTHER']

        for col in ['TRIP IN','TRIP OUT']:
            df.loc[(df[RSUPER] == self.config_backend.super_state_swap['DRILL'])&
                (df[RSUB] == sub_state_swap[col]), RSUB] = sub_state_swap['OTHER']

        for col in ['TRIP IN','TRIP OUT']:
            df.loc[(df[RSUPER] == self.config_backend.super_state_swap['DRILL'])
                &(df[RSUB] == sub_state_swap['OTHER'])
                &(df[RSUBACTIVITY] == self.config_backend.sub_activity_state_swap[col]), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['OTHER']
        
        #connection trip while drilling 'Connection Drilling': 'CNX (drill)','Connection Tripping': 'CNX (trip)'
        df.loc[(df[RSUPER]==self.config_backend.super_state_swap['DRILL'])
            &(df[RSUB] == sub_state_swap['CNX (trip)']), RSUB] = sub_state_swap['NULL']

        #connection drill while tripping
        df.loc[(df[RSUPER]==self.config_backend.super_state_swap['TRIP'])
            &(df[RSUB] == sub_state_swap['CNX (drill)']),RSUB] = sub_state_swap['NULL']
        
        #####################################
        ### out of hole to sub activities ###
        #####################################
        df.loc[(df[RSUPER] == self.config_backend.super_state_swap['OUT OF HOLE']), RSUBACTIVITY] = self.config_backend.sub_activity_state_swap['OUT OF HOLE']
        
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
    
        df = self.comp_duration(df, activity_col = RSUBACTIVITY, dtime = dtime)
        
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
            &(df[self.config_backend.MM] == 0), RSUBACT] = sub_state_swap['ROTATE']
            #*** replace 'ROTATE' with 'CIR (static)' Ecopetrol March 22, 2021 ***
        
        ##################################
        ### smooth subactivity column ####
        ##################################
        if smooth:

            for key, value in filter_dict.items():
                df.loc[(df[DUR] < pd.Timedelta(seconds=value))&
                    (df[RSUBACTIVITY] == self.config_backend.sub_activity_state_swap[key]), RSUBACT] = np.nan

            #fill forwars and change variable type
            df[RSUBACT].fillna(method = 'ffill',inplace=True) 
        
        ##recompute labels and duration for smooth labels
        df = self.comp_duration(df, activity_col = RSUBACT, dtime = dtime)

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
    
    
    def conn_activity_label(self, df_input: pd.DataFrame, 
                            activity: str, LABEL: str,
                            activity_col = None, 
                            activity_dict = None) -> pd.DataFrame:
        
        if activity_col is None:
            activity_col = self.config_backend.RSUBACT
        
        if activity_dict is None:  
            activity_dict = self.config_backend.sub_activity_state_swap 
        
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
    
    def conn_activity_label_between(self, df_input: pd.DataFrame, 
                                    activity: str, LABEL: str,
                                    activity_col = None, activity_dict = None) -> pd.DataFrame:
        
        if activity_col is None:
            activity_col = self.config_backend.RSUBACT
        
        if activity_dict is None:  
            activity_dict = self.config_backend.sub_activity_state_swap    
        
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
    
    def add_rop_mse(self, df_input: pd.DataFrame,
                    df_input_rt: pd.DataFrame,
                    ROP = None, MSE = None, ROP_MSE = None, RSUBACT = None, 
                    activity_dict = None) -> pd.DataFrame:
        
        if ROP is None:
            ROP = self.config_backend.ROP
        
        if MSE is None:
            MSE = self.config_backend.MSE
        
        if ROP_MSE is None:
            ROP_MSE = self.config_backend.ROP_MSE
        
        if RSUBACT is None:
            RSUBACT = self.config_backend.RSUBACT
        
        if activity_dict is None:
            activity_dict = self.config_backend.sub_activity_state_swap                
        
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

        model_consts = (
            df.loc[
                (df[RSUBACT] == activity_dict['ROTATE']) 
                | (df[RSUBACT] == activity_dict['SLIDE'])
            ]
            .groupby([self.config_backend.BS, self.config_backend.FORM])[[ROP, MSE]]
            .median()
            .reset_index()
            .rename(columns={ROP: ROP+'_median', MSE: MSE+'_median'})
        )

        #display(model_consts)
        
        #in case of multiple execution
        try:
            df_rt.drop(columns = [ROP + '_median', MSE + '_median', ROP_MSE], inplace = True)
        except:
            pass

        df_rt = df_rt.merge(model_consts, how = 'left', left_on = [self.config_backend.BS, self.config_backend.FORM], right_on = [self.config_backend.BS, self.config_backend.FORM])
        
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
    
    #process kpi wt_wt and conn_drill_time
    def process_kpi(self,df_input: pd.DataFrame, measured_depth_thr,
                    datetime_thr = pd.Timedelta(4, unit='hour'),
                    DATETIME = None, HD = None) -> pd.DataFrame:

        if DATETIME is None:
            DATETIME = self.config_backend.DATETIME
        
        if HD is None:
            HD = self.config_backend.HD
                
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

    def compute_df_wt_wt(self, df_input: pd.DataFrame, measured_depth_thr = 5,
                        wt_wt_max = 200,
                        WELLID = None, BS = None, WTWT = None,
                        LABELcd = None, TIME = None) -> pd.DataFrame:
        
        if WELLID is None:
            WELLID = self.config_backend.WELLID
        
        if BS is None:
            BS = self.config_backend.BS   
        
        if WTWT is None:
            WTWT = self.config_backend.WTWT
        
        if LABELcd is None:
            LABELcd = self.config_backend.LABELcd
            
        if TIME is None:
            TIME = self.config_backend.TIME             
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

        df_aux = df.groupby([WELLID, LABELcd])[[self.config_backend.HD, self.config_backend.BD, self.config_backend.DATETIME, TIME]].min().reset_index()
        df_wt_wt = df_wt_wt.reset_index().merge(df_aux, how = 'left', 
                                left_on=[WELLID, LABELcd], right_on =[WELLID, LABELcd])

        if df_wt_wt.shape[0] > 1:
            df_process = self.process_kpi(df_wt_wt, measured_depth_thr)

                #group
            df_grouped = df_process.groupby([self.config_backend.HD, BS, self.config_backend.DATETIME + '_flag'])[[WTWT, self.config_backend.DATETIME, WELLID, LABELcd, TIME + '_first', TIME + '_last']].agg({
                    self.config_backend.DATETIME: 'min', 
                    WTWT: 'sum',
                    WELLID: 'max', 
                    LABELcd: 'min',
                    TIME + '_first': 'min',
                    TIME + '_last': 'max'
                }).reset_index() \
                .sort_values(by=self.config_backend.DATETIME)

        else:
            df_grouped = df_wt_wt

        print('\n df_wt_wt size before filtering high weigh to weight times:', df_grouped.shape[0])
        df_grouped = df_grouped.loc[(df_grouped[WTWT] < wt_wt_max),:]
        print('\n df_wt_wt size after filtering high weigh to weight times:', df_grouped.shape[0])

        df_grouped = df_grouped.set_index(LABELcd)
        use_cols = [WELLID, TIME + '_first', TIME + '_last', WTWT, BS]

        return df_grouped[use_cols]
    
    #mapping function to identify trip in/trip out activities
    def get_trip_in_trip_out(self,s: pd.Series, 
                            activity_dict: dict) -> int:
        """This is a function to identify prevailing activity between 
        connections: trip in, trip out or nan.
        Here s is a Series,
        activity_dict is a dictionary that define mapping from names to number codes.
        Function returns trip in, trip out or nan."""
        
        try:
            output = s.loc[s.isin([self.config_backend.sub_activity_state_swap['TRIP IN'],
                                self.config_backend.sub_activity_state_swap['TRIP OUT']])].mode().values[0]
        except:
            output = np.nan
        
        return output
    
    #function to compute tripping table that includes pipe speed
    def compute_df_trip(self, df_input: pd.DataFrame, add_stand = False,
                        trip_speed_max = 5000,
                        WELLID = None, BS = None, PIPESP = None,
                        LABELct = None, LABELbtwn = None,
                        BD = None, dBD = None, TIME = None, dTIME = None, 
                        RSUBACT = None, STAND = None,
                        activity_dict = None) -> pd.DataFrame:
        
        if WELLID is None:
            WELLID = self.config_backend.WELLID
        
        if BS is None:
            BS = self.config_backend.BS
        
        if PIPESP is None:
            PIPESP = self.config_backend.PIPESP
        
        if LABELct is None:
            LABELct = self.config_backend.LABELct
            
        if LABELbtwn is None:
            LABELbtwn = self.config_backend.LABELbtwn
        
        if BD is None:
            BD = self.config_backend.BD
        
        if dBD is None:
            dBD = self.config_backend.dBD
        
        if TIME is None:
            TIME = self.config_backend.TIME
        
        if dTIME is None:
            dTIME = self.config_backend.dTIME
        
        if RSUBACT is None:
            RSUBACT = self.config_backend.RSUBACT
        
        if STAND is None:
            STAND = self.config_backend.STAND
        
        if activity_dict is None:
            activity_dict = self.config_backend.sub_activity_state_swap                                        
        
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
                    TIME + '_first', TIME + '_last', dBD, dTIME, PIPESP, self.config_backend.TRIPSPEED, BS, RSUBACT]

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
        df_trip[self.config_backend.TRIPSPEED] = (df_trip[dBD]/df_trip[dTIME + '_conn']).clip(lower=0) * 60
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
                .apply(self.get_trip_in_trip_out, activity_dict = self.config_backend.sub_activity_state_swap).to_frame().dropna().reset_index()

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
        df_trip = df_trip.loc[(df_trip[self.config_backend.TRIPSPEED] < trip_speed_max),:]
        print('\ndf_trip size after filtering low trip speed:', df_trip.shape[0])

        return df_trip
    
    #process conn_trip_time
    def process_kpi_trip(self,df_conn_trip, stand_length, delta_stand_length, BD = None):
        
        if BD is None:
            BD = self.config_backend.BD
            
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
   
    def compute_df_conn(self, df_input: pd.DataFrame, 
                        compute_col: str, activity: str, dtime_dict: dict,
                        stand_length = None, delta_stand_length = 5,
                        measured_depth_thr = 5,
                        WELLID = None, BS = None, RSUBACT = None, 
                        BD = None, dBD = None,  
                        LABEL = None, TIME = None,
                        activity_dict = None) -> pd.DataFrame:
        
        if stand_length is None:
            stand_length = self.config_backend.STAND_LENGTH
        
        if WELLID is None:
            WELLID = self.config_backend.WELLID
        
        if BS is None:
            BS = self.config_backend.BS
        
        if RSUBACT is None:
            RSUBACT = self.config_backend.RSUBACT
        
        if BD is None:
            BD = self.config_backend.BD
        
        if dBD is None:
            dBD = self.config_backend.dBD
        
        if LABEL is None:
            LABEL = self.config_backend.LABEL
        
        if TIME is None:
            TIME = self.config_backend.TIME
        
        if activity_dict is None:
            activity_dict = self.config_backend.sub_activity_state_swap                                
        
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
                df_conn = pd.concat([df_conn, df_cd])

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

        df_aux = df.groupby([WELLID, LABEL])[[self.config_backend.HD, BD, self.config_backend.DATETIME, TIME]].min().reset_index()
        df_conn = df_conn.reset_index().merge(df_aux, how = 'left', 
                                left_on=[WELLID, LABEL], right_on =[WELLID, LABEL])

        if compute_col == self.config_backend.CONNDTIME:
            df_process = self.process_kpi(df_conn, measured_depth_thr)

            #group
            df_conn = (df_process.groupby([self.config_backend.HD, BS, self.config_backend.DATETIME + '_flag'])[[compute_col, self.config_backend.DATETIME, WELLID, LABEL, dBD]]
                .agg({
                    self.config_backend.DATETIME: 'min',
                    compute_col: 'sum',
                    WELLID: 'max',
                    LABEL: 'min',
                    dBD: 'sum'
                })
                .reset_index()
                .sort_values(by=self.config_backend.DATETIME)
                .set_index(LABEL)
        )

        elif compute_col == self.config_backend.CONNTTIME:
            df_process = self.process_kpi_trip(df_conn, stand_length, delta_stand_length)
            #group
            df_conn = (df_process.groupby([BD + '_diff_flag'])[[compute_col, self.config_backend.DATETIME, WELLID, LABEL, dBD, BS]]
                .agg({
                    self.config_backend.DATETIME: 'min',
                    compute_col: 'sum',
                    WELLID: 'max',
                    LABEL: 'min',
                    dBD: 'sum',
                    BS: 'min'
                })
                .reset_index()
                .sort_values(by=self.config_backend.DATETIME)
                .set_index(LABEL)
            )   

        #clip 
        df_conn[compute_col] = df_conn[compute_col].clip(lower=0, upper=clip_max)
        use_cols = [WELLID, compute_col, dBD, BS]

        #filter trip connections faster than 1 minute
        if activity == 'CNX (trip)':
            df_conn = df_conn[df_conn[compute_col] >= 1]

        return df_conn[use_cols]
    
    def rig_activity_summary(self, df_input: pd.DataFrame, group_col: str,
                            super_activity: str, hole_diameter: float,
                            wells_select: list, current_well: int,
                            dtime_dict: dict, refs = None,
                            add_naming = '', MMDD = None,
                            WELL = None, TIME = None, AVG = None, 
                            RSUPER = None, RSUBACT = None,
                            super_activity_dict = None,
                            sub_activity_dict = None,
                            color_dict = None, 
                            rig_activity_order = None,
                            round_ndigits = None, replace_dot = None,
                            csv_folder = 'csv', plot_folder = 'plot',
                            save_folder = None,  save = True):
        
        if MMDD is None:
            MMDD = self.config_backend.MMDD
        
        if WELL is None:
            WELL = self.config_backend.WELLID
        
        if TIME is None:
            TIME = self.config_backend.TIME
        
        if AVG is None:
            AVG = self.config_backend.AVG
        
        if RSUPER is None:
            RSUPER = self.config_backend.RSUPER
        
        if RSUBACT is None:
            RSUBACT = self.config_backend.RSUBACT
        
        if super_activity_dict is None:
            super_activity_dict = self.config_backend.super_state_swap
        
        if sub_activity_dict is None:
            sub_activity_dict = self.config_backend.sub_activity_state
        
        if color_dict is None:
            color_dict = self.config_backend.rig_activity_color_dict
        
        if rig_activity_order is None:
            rig_activity_order = self.config_backend.rig_activity_order
        
        if round_ndigits is None:
            round_ndigits = self.config_backend.ROUND_NDIGITS
        
        if replace_dot is None:
            replace_dot = self.config_backend.REPLACE_DOT
        
        if save_folder is None: 
            save_folder = self.config_backend.SAVE_FOLDER                                                   

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
            cond &= (df[self.config_backend.BS] == hole_diameter)
            
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
            total_time_sub_act = total_time_sub_act[self.config_backend.rig_activity_drill_order]
        elif super_activity == 'TRIP':
            total_time_sub_act = total_time_sub_act[self.config_backend.rig_activity_trip_order]

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
                df_RS.plot.barh(y = 'xmax', width=0.8, ax = ax2, color = self.config_backend.color_neutral, alpha=0.2)
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
                df_RS.plot.barh(y = 'xmax', width=0.8, ax = ax2, color = self.config_backend.color_neutral, alpha=0.2)
                ax2.set_xlim(xmin, xmax)
                ax2.set_ylim(ax.get_ylim())
                ax2.set_yticklabels(df_RS['label'].replace(np.nan, '').values)
                ax2.grid(None)
                ax2.set_ylabel(None)
                ax2.get_legend().remove()

            #add multiindex for displaying purposes
            if add_naming != '_rt':
                title = super_activity.upper() + 'ING' if (super_activity != 'all') else 'DRILLING & TRIPPING'
                title_add = ': SECTION ' + self.config_backend.decimal_fraction_str(hole_diameter) if (hole_diameter != 'all') else ''
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

    def kpi_boxplot(self, df_, df_wt_wt_, df_trip_, df_conn_drill_, df_conn_trip_, 
                        hole_diameter: float,
                        wells_select: list, current_well: int, 
                        WELL = None, ROP = None, 
                        CONNTTIME = None, CONNDTIME = None,
                        PIPESP = None, WTWT = None, dTIME = None, 
                        rop_unit = 'ft/h', trip_speed_unit = 'ft/h',
                        activity_dict = None,
                        kpi_colors = None, 
                        replace_dot = None, round_ndigits = None, 
                        save_folder = None,  save = True) -> pd.DataFrame:
        
        if WELL is None:
            WELL = self.config_backend.WELLID
        
        if ROP is None:
            ROP = self.config_backend.ROP
        
        if CONNTTIME is None:
            CONNTTIME = self.config_backend.CONNTTIME
        
        if CONNDTIME is None:
            CONNDTIME = self.config_backend.CONNDTIME
        
        if PIPESP is None:
            PIPESP = self.config_backend.TRIPSPEED
        
        if WTWT is None:
            WTWT = self.config_backend.WTWT
        
        if dTIME is None:
            dTIME = self.config_backend.dTIME
        
        if activity_dict is None:
            activity_dict = self.config_backend.sub_activity_state_swap
        
        if kpi_colors is None:
            kpi_colors = self.config_backend.kpi_colors
        
        if replace_dot is None:
            replace_dot = self.config_backend.REPLACE_DOT
        
        if round_ndigits is None:
            round_ndigits = self.config_backend.ROUND_NDIGITS
        
        if save_folder is None: 
            save_folder = self.config_backend.SAVE_FOLDER                                            

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
        
        s = 'SECTION ' + self.decimal_fraction_str(hole_diameter) if (hole_diameter != 'all') else ''

        bs = str(hole_diameter).replace('.', replace_dot)
        
        #select hole size section
        if hole_diameter != 'all':

            df = df.loc[df[self.config_backend.BS] == hole_diameter]
            df_trip = df_trip.loc[df_trip[self.config_backend.BS] == hole_diameter]
            df_conn_drill = df_conn_drill.loc[df_conn_drill[self.config_backend.BS] == hole_diameter]
            df_conn_trip = df_conn_trip.loc[df_conn_trip[self.config_backend.BS] == hole_diameter]
            df_wt_wt = df_wt_wt.loc[df_wt_wt[self.config_backend.BS] == hole_diameter]
            
        #store data quantiles together
        df_box = pd.DataFrame()
        
        _, axes = plt.subplots(11,1,figsize=(10,14))
        
        title = f'Drilling ROP ({rop_unit})'
        ax = axes[0]
        s_drill = df.loc[df[WELL].isin(wells_select)&
                        (((df[self.config_backend.RSUBACT] == activity_dict['ROTATE']))
                        |((df[self.config_backend.RSUBACT] == activity_dict['SLIDE']))), ROP]
        q50_current = df.loc[df[WELL].isin([current_well])&
                        (((df[self.config_backend.RSUBACT] == activity_dict['ROTATE']))
                        |((df[self.config_backend.RSUBACT] == activity_dict['SLIDE']))), ROP].median()

        df_box = self.kpi_boxplot_sub(s_drill, q50_current, title, ax).fillna(0) 
        ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))
        
        title = f'Rotating ROP ({rop_unit})'
        ax = axes[1]
        s_drill = df.loc[df[WELL].isin(wells_select)&
                        (df[self.config_backend.RSUBACT] == activity_dict['ROTATE']), ROP]
        q50_current = df.loc[df[WELL].isin([current_well])&
                        (df[self.config_backend.RSUBACT] == activity_dict['ROTATE']), ROP].median()

        df_box[title] = self.kpi_boxplot_sub(s_drill, q50_current, title, ax).fillna(0)
        ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))
        
        title = f'Sliding ROP ({rop_unit})'
        ax = axes[2]
        s_drill = df.loc[df[WELL].isin(wells_select)&
                        (df[self.config_backend.RSUBACT] == activity_dict['SLIDE']), ROP]
        q50_current = df.loc[df[WELL].isin([current_well])&
                        (df[self.config_backend.RSUBACT] == activity_dict['SLIDE']), ROP].median()
        
        if not(s_drill.empty):
            df_box[title] = self.kpi_boxplot_sub(s_drill, q50_current, title, ax).fillna(0)  
        else:
            df_box[title] = 0 
        ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))
        
        title = 'Connection Time (mm:ss)'
        ax = axes[3]
        if not(df_conn_drill.empty):
            s_conn = pd.concat([df_conn_drill.loc[df_conn_drill[WELL].isin(wells_select), CONNDTIME], df_conn_trip.loc[df_conn_trip[WELL].isin(wells_select), CONNTTIME]])

            q50_current = pd.concat([df_conn_drill.loc[df_conn_drill[WELL].isin([current_well]), CONNDTIME],df_conn_trip.loc[df_conn_trip[WELL].isin([current_well]), CONNTTIME]]).median()

            df_box[title] = self.kpi_boxplot_sub(s_conn, q50_current, title, ax, kpi_colors = kpi_colors[-1::-1]).fillna(0) 
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
            df_box[title] = self.kpi_boxplot_sub(s_conn, q50_current, title, ax, kpi_colors = kpi_colors[-1::-1]).fillna(0) 
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
            df_box[title] = self.kpi_boxplot_sub(s_conn, q50_current, title, ax, kpi_colors = kpi_colors[-1::-1]).fillna(0) 
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
            df_box[title] = self.kpi_boxplot_sub(s_trip, q50_current, title, ax).fillna(0) 
        else:
            df_box[title] = 0
        ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))
        
        title = f'Tripping In Speed ({trip_speed_unit})'
        ax = axes[7]
        s_trip = df_trip.loc[(df_trip[WELL].isin(wells_select))&
                            (df_trip[self.config_backend.RSUBACT] == activity_dict['TRIP IN']), PIPESP]
        q50_current = df_trip.loc[(df_trip[WELL].isin([current_well]))&
                                (df_trip[self.config_backend.RSUBACT] == activity_dict['TRIP IN']), PIPESP].median()
        
        if not(s_trip.empty):
            df_box[title] = self.kpi_boxplot_sub(s_trip, q50_current, title, ax).fillna(0) 
        else:
            df_box[title] = 0
        ax.set_xlim(round(df_box.loc['q1',title],1), round(df_box.loc['q99',title],1))

        title = f'Tripping Out Speed ({trip_speed_unit})'
        ax = axes[8]
        s_trip = df_trip.loc[(df_trip[WELL].isin(wells_select))&
                            (df_trip[self.config_backend.RSUBACT] == activity_dict['TRIP OUT']), PIPESP]
        q50_current = df_trip.loc[(df_trip[WELL].isin([current_well]))&
                                (df_trip[self.config_backend.RSUBACT] == activity_dict['TRIP OUT']), PIPESP].median()
        
        if not(s_trip.empty):
            df_box[title] = self.kpi_boxplot_sub(s_trip, q50_current, title, ax).fillna(0) 
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
            df_box[title] = self.kpi_boxplot_sub(s_trip, q50_current, title, ax, kpi_colors = kpi_colors[-1::-1]).fillna(0) 
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
            df_box[title] = self.kpi_boxplot_sub(s_wtwt, q50_current, title, ax, kpi_colors = kpi_colors[-1::-1]).fillna(0) 
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

