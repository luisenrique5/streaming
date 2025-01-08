import pandas as pd
import numpy as np
from datetime import datetime
import os
import shutil
import time
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from src.entities.bi_drill_utility import BI_Drill_Utility

class DrillDataProcessor:
    def __init__(self, input_folder: str):
        self.input_folder = input_folder
        self._setup_paths()
        self.bi_drill_utility = BI_Drill_Utility(self.input_folder)
        
        self.start_time = time.time()
        self.current_datetime = datetime.now()
        self.formatted_datetime = self.current_datetime.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

    def _setup_paths(self):
        cleaned_path = self.input_folder.lstrip("./")
        path_components = cleaned_path.rstrip("/").split("/")
        
        if len(path_components) >= 6 and path_components[0] == "input_data":
            self.client, self.project, self.stream, self.username, self.scenario = path_components[1:6]
            self.save_real_time_folder = self.input_folder.replace("input_data", "output_data") + "/real_time_update/"
            print(f"\n========== RUN 2. MACHINE LEARNING CODE EXECUTION  | INFO | PROCESS | "
                  f"100002 | {self.client} | {self.project} | {self.stream} | {self.username} | {self.scenario} | "
                  f"real_time_part_tablesupdate.sh\n")
        else:
            raise ValueError("La estructura del input_folder no coincide con el formato esperado")
        
        for folder in ['plot', 'csv']:
            folder_path = f'{self.save_real_time_folder}{folder}'
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)

    def process(self):
        start_time = time.time()
        # Cargar datos del plan
        df_plan = pd.read_csv(f'{self.input_folder}plan/time_depth_plan.csv')
        df_direct_plan = pd.read_csv(f'{self.input_folder}plan/directional_plan.csv')
        df_form_plan = pd.read_csv(f'{self.input_folder}plan/formation_tops_plan.csv')
        
        # Datos históricos
        df = pd.read_csv(f'{self.input_folder}database/time_based_drill.csv', parse_dates=[self.bi_drill_utility.DATETIME])
        
        # Configuración de bit sizes y formaciones
        bit_sizes_max = [df_plan[self.bi_drill_utility.BS].unique().max()]
        formation_palette = sns.color_palette("husl", df_form_plan[self.bi_drill_utility.FORM].nunique())
        form_colors_dict = dict(zip(df_form_plan[self.bi_drill_utility.FORM].unique(), formation_palette))
        
        # Cargar datos en tiempo real
        if self.bi_drill_utility.NO_RT:
            df_rt, df_survey_rt = self.bi_drill_utility.make_dummy_data(bit_sizes_max)
        else:
            df_rt = pd.read_csv(f'{self.input_folder}real_time/time_based_drill_current_well.csv', 
                              parse_dates=[self.bi_drill_utility.DATETIME])
            df_survey_rt = pd.read_csv(f'{self.input_folder}real_time/official_survey_current_well.csv')
            if df_survey_rt.shape[0] <= 1:
                df_aux = pd.DataFrame(
                    columns=self.bi_drill_utility.survey_raw_cols,
                    data=np.zeros([2, len(self.bi_drill_utility.survey_raw_cols)]),
                    index=np.arange(1,3)
                )
                df_survey_rt = df_survey_rt.append(df_aux)
        
        df_survey_rt.drop_duplicates(inplace=True)
        
        # Procesar datos del survey
        df_survey_rt[self.bi_drill_utility.WELLID] = self.bi_drill_utility.CURRENT_WELL_ID
        df_survey_rt = self.bi_drill_utility.calculate_tvd_ns_ew_dls(df_survey_rt)
        df_survey_rt[self.bi_drill_utility.UWDEP] = np.sqrt(df_survey_rt[self.bi_drill_utility.NS]**2 + 
                                                           df_survey_rt[self.bi_drill_utility.EW]**2)
        
        df_direct_plan[self.bi_drill_utility.HD + '_max'] = df_direct_plan[self.bi_drill_utility.HD].shift(-1).fillna(df_survey_rt[self.bi_drill_utility.HD].max())
        for index, row in df_direct_plan.iterrows():
            cond = (df_survey_rt[self.bi_drill_utility.HD] >= row[self.bi_drill_utility.HD]) & \
                  (df_survey_rt[self.bi_drill_utility.HD] <= row[self.bi_drill_utility.HD + '_max'])
            df_survey_rt.loc[cond, self.bi_drill_utility.WSEC] = row[self.bi_drill_utility.WSEC]
        df_direct_plan.drop(columns=[self.bi_drill_utility.HD + '_max'], inplace=True)
        
        # Reformatear survey data
        miss_cols = [col for col in self.bi_drill_utility.survey_cols if col not in df_survey_rt.columns]
        for col in miss_cols:
            df_survey_rt[col] = np.nan
        df_survey_rt = df_survey_rt[self.bi_drill_utility.survey_cols]
        
        # Procesar datos en tiempo real
        df_rt[self.bi_drill_utility.DATETIME] = pd.to_datetime(df_rt[self.bi_drill_utility.DATETIME].apply(lambda x: str(x)[:19]),
                                                             format='%Y-%m-%d %H:%M:%S')
        df_rt.drop_duplicates(self.bi_drill_utility.DATETIME, inplace=True)
        df_rt.sort_values(by=[self.bi_drill_utility.DATETIME], inplace=True)
        
        df_rt[self.bi_drill_utility.WELLID] = self.bi_drill_utility.CURRENT_WELL_ID
        df_rt[self.bi_drill_utility.TIME] = (df_rt[self.bi_drill_utility.DATETIME].diff().dt.seconds/60).cumsum().fillna(0)
        
        time_bins = np.arange(0, int(np.ceil(df_rt[self.bi_drill_utility.TIME].max()/60/24))+1, 1)
        df_rt[self.bi_drill_utility.DAYN] = pd.cut(df_rt[self.bi_drill_utility.TIME]/60/24,
                                                  bins=time_bins,
                                                  labels=time_bins[1:],
                                                  include_lowest=True).astype(int)
        
        df_rt = self.bi_drill_utility.add_tvd_well_section(df_rt, df_survey_rt)
        df_rt.sort_values(by=[self.bi_drill_utility.DATETIME], inplace=True)
        
        # Añadir features del survey
        for feature in [self.bi_drill_utility.INC, self.bi_drill_utility.AZM, self.bi_drill_utility.DLS]:
            df_rt = self.bi_drill_utility.add_survey(df_rt, df_survey_rt, feature)
        
        # Procesar hole diameter y mud motor
        if not self.bi_drill_utility.READ_BS_MM_CASING:
            df_rt[self.bi_drill_utility.BS] = np.nan
            df_rt[self.bi_drill_utility.MM] = np.nan
            
            group_col = self.bi_drill_utility.TIME
            df_plan_group = df_plan.groupby([self.bi_drill_utility.BS, self.bi_drill_utility.MM])[group_col].agg(['min','max'])
            for index, row in df_plan_group.iterrows():
                cond = (df_rt[group_col] >= row['min']) & (df_rt[group_col] <= row['max'])
                df_rt.loc[cond, [self.bi_drill_utility.BS, self.bi_drill_utility.MM]] = index
            
            df_rt[self.bi_drill_utility.BS].fillna(method='ffill', inplace=True)
            df_rt[self.bi_drill_utility.MM].fillna(method='ffill', inplace=True)
            df_rt[self.bi_drill_utility.CASING] = np.nan
        
        # Procesar casing
        df_plan[self.bi_drill_utility.TIME + '_max'] = df_plan[self.bi_drill_utility.TIME].shift(-1).fillna(df_rt[self.bi_drill_utility.TIME].max())
        for index, row in df_plan.iterrows():
            cond = (df_rt[self.bi_drill_utility.TIME] >= row[self.bi_drill_utility.TIME]) & \
                  (df_rt[self.bi_drill_utility.TIME] <= row[self.bi_drill_utility.TIME + '_max'])
            df_rt.loc[cond, self.bi_drill_utility.CASING] = row[self.bi_drill_utility.CASING]
        df_plan.drop(columns=[self.bi_drill_utility.TIME + '_max'], inplace=True)
        
        # Calcular MSE y formation
        df_rt = self.bi_drill_utility.calculate_mse(df_rt)
        df_rt[self.bi_drill_utility.FORM] = np.nan
        
        df_form_plan[self.bi_drill_utility.FORMBOT] = df_form_plan[self.bi_drill_utility.FORMTOP].shift(-1).fillna(df_rt[self.bi_drill_utility.HD].max())
        df_form_plan[self.bi_drill_utility.FORMBOTTIME] = df_form_plan[self.bi_drill_utility.FORMTOPTIME].shift(-1).fillna(df_rt[self.bi_drill_utility.TIME].max())
        
        for _, row in df_form_plan.iterrows():
            d_min, d_max = (row[self.bi_drill_utility.FORMTOP], row[self.bi_drill_utility.FORMBOT])
            cond = (df_rt[self.bi_drill_utility.HD] >= d_min) & (df_rt[self.bi_drill_utility.HD] < d_max)
            df_rt.loc[cond, self.bi_drill_utility.FORM] = row[self.bi_drill_utility.FORM]
        
        # Procesar rig activity labels
        if self.bi_drill_utility.DTIME_RT is None:
            dtime_rt = (df_rt[self.bi_drill_utility.TIME].diff()*60).round().dropna().astype(int).mode()[0]
        else:
            dtime_rt = self.bi_drill_utility.DTIME_RT
            
        if dtime_rt not in [1,5,10,15]:
            warnings.warn(f"Check the data frequence of real-time data: atypical value of {dtime_rt} s automatically identified")
            
        dtime_dict = {self.bi_drill_utility.CURRENT_WELL_ID: pd.Timedelta(seconds=dtime_rt)}
        
        # Leer rig design y procesar labels
        df_rig = pd.read_csv(f'{self.input_folder}database/rig_design.csv')
        df_rig_rt = df_rig.loc[df_rig[self.bi_drill_utility.RIG] == self.bi_drill_utility.RIG_NAME].squeeze()
        
        if not self.bi_drill_utility.READ_CURRENT_RIG_LABELS:
            df_rt = self.bi_drill_utility.rig_labels(df_rt, df_rig_rt, self.bi_drill_utility.RIG_NAME, dtime_rt, real_time=False)
        
        # Reformatear data en tiempo real
        miss_cols = [col for col in self.bi_drill_utility.time_based_drill_cols if col not in df_rt.columns]
        for col in miss_cols:
            df_rt[col] = np.nan
        df_rt = df_rt[self.bi_drill_utility.time_based_drill_cols]
        
        # Merge con dummy data
        df_rt_dummy, df_survey_rt_dummy = self.bi_drill_utility.make_dummy_data(bit_sizes_max, day=df_rt[self.bi_drill_utility.DATETIME].min())
        df_rt = pd.concat([df_rt,df_rt_dummy[self.bi_drill_utility.time_based_drill_cols]]).sort_values(by=[self.bi_drill_utility.DATETIME])
        df_survey_rt = pd.concat([df_survey_rt,df_survey_rt_dummy[self.bi_drill_utility.survey_cols]]).sort_values(by=[self.bi_drill_utility.HD])
        
        df_rt[self.bi_drill_utility.WELLID] = self.bi_drill_utility.CURRENT_WELL_ID
        df_survey_rt[self.bi_drill_utility.WELLID] = self.bi_drill_utility.CURRENT_WELL_ID
        df_rt.drop_duplicates(self.bi_drill_utility.DATETIME, inplace=True)
        
        # Procesar labels consecutivos
        df_rt = self.bi_drill_utility.conn_activity_label(df_rt, activity='CNX (trip)', 
                                                        LABEL=self.bi_drill_utility.LABELct)
        df_rt = self.bi_drill_utility.conn_activity_label(df_rt, activity='CNX (drill)', 
                                                        LABEL=self.bi_drill_utility.LABELcd)
        df_rt = self.bi_drill_utility.conn_activity_label_between(df_rt, activity='CNX (trip)', 
                                                                LABEL=self.bi_drill_utility.LABELbtwn)
        
        # Añadir month_day
        df_rt[self.bi_drill_utility.MMDD] = df_rt[self.bi_drill_utility.DATETIME].dt.month.astype('str') + \
                                           '.' + df_rt[self.bi_drill_utility.DATETIME].dt.day.astype('str').apply(lambda x: x.zfill(2))
        
        # Calcular stand number
        bins = np.arange(0, df_rt[self.bi_drill_utility.HD].max() + self.bi_drill_utility.STAND_LENGTH, self.bi_drill_utility.STAND_LENGTH)
        labels = np.arange(1, len(bins))
        bins_labels_dict = dict(zip(labels, bins + self.bi_drill_utility.STAND_LENGTH/2))
        df_rt[self.bi_drill_utility.STAND] = pd.cut(df_rt[self.bi_drill_utility.HD], bins=bins, labels=labels)

        # Añadir rop_mse class
        df_rt = self.bi_drill_utility.add_rop_mse(df, df_rt)

        # Añadir duration
        min_time = pd.Timedelta(seconds=dtime_rt)
        duration_map = df_rt.groupby(self.bi_drill_utility.LABEL).apply(
            lambda x: x[self.bi_drill_utility.TIME].nunique() * min_time)
        df_rt[self.bi_drill_utility.DUR] = df_rt[self.bi_drill_utility.LABEL].map(duration_map)

        # Mostrar datos real-time
        print('\n*** real-time data ***\n')
        print(df_rt.head())

        # Computar tablas de tripping y connection time
        df_wt_wt_rt = self.bi_drill_utility.compute_df_wt_wt(df_rt)
        df_trip_rt = self.bi_drill_utility.compute_df_trip(df_rt)
        df_conn_drill_rt = self.bi_drill_utility.compute_df_conn(
            df_rt, 
            compute_col=self.bi_drill_utility.CONNDTIME, 
            activity='CNX (drill)', 
            dtime_dict=dtime_dict
        )
        df_conn_trip_rt = self.bi_drill_utility.compute_df_conn(
            df_rt, 
            compute_col=self.bi_drill_utility.CONNTTIME, 
            activity='CNX (trip)', 
            dtime_dict=dtime_dict
        )

        # Guardar datos real-time
        df_rt.to_csv(f"{self.save_real_time_folder}time_based_drill_current_well_out.csv")
        df_survey_rt.to_csv(f"{self.save_real_time_folder}official_survey_current_well_out.csv")

        # Procesar bit sizes
        current_bit_size = df_rt[self.bi_drill_utility.BS].iloc[-1]
        include_bits = df_rt[self.bi_drill_utility.BS].unique()
        bit_sizes = ['all'] + list(include_bits)
        bit_sizes = [bit for bit in bit_sizes if str(bit) != 'nan']

        # Overview tab
        for super_activity in ['all', 'DRILL', 'TRIP']:
            for hole_diameter in bit_sizes:
                _ = self.bi_drill_utility.rig_activity_summary(
                    df_rt, 
                    self.bi_drill_utility.WELLID,
                    super_activity, 
                    hole_diameter,
                    self.bi_drill_utility.WELLS_SELECT_ID,  
                    self.bi_drill_utility.CURRENT_WELL_ID, 
                    dtime_dict, 
                    add_naming='_rt',
                    save_folder=self.save_real_time_folder
                )
        
         # Update KPIs
        for hole_diameter in bit_sizes:
            self.bi_drill_utility.kpi_boxplot(
                df_rt,
                df_wt_wt_rt,
                df_trip_rt,
                df_conn_drill_rt,
                df_conn_trip_rt,
                hole_diameter,
                [self.bi_drill_utility.CURRENT_WELL_ID],
                self.bi_drill_utility.CURRENT_WELL_ID,
                save_folder=self.save_real_time_folder
            )        

        # # Well performance (WCR plot)
        # df_wcr_plan = pd.read_csv(f'{self.bi_drill_utility.SAVE_FOLDER}csv/wcr_plan.csv')
        # for hole_diameter in bit_sizes:
        #     self.bi_drill_utility.compute_WCR(
        #         df_rt, 
        #         df_wcr_plan, 
        #         hole_diameter, 
        #         dtime_dict, 
        #         self.bi_drill_utility.well_name_dict,
        #         save_folder=self.save_real_time_folder
        #     )

        # Drilling tab
        super_activity = 'DRILL'
        
        # # KPI colors para conexiones
        # df_wt_wt_rt_kpi = self.bi_drill_utility.get_kpi_color(
        #     df_rt, 
        #     df_wt_wt_rt, 
        #     self.bi_drill_utility.LABELcd, 
        #     self.bi_drill_utility.WTWT, 
        #     self.bi_drill_utility.WTWT,
        #     name='weight_weight', 
        #     kpi_ascend=False,
        #     read_folder=self.bi_drill_utility.SAVE_FOLDER,
        #     save_folder=self.save_real_time_folder
        # )

        # df_conn_drill_rt_kpi = self.bi_drill_utility.get_kpi_color(
        #     df_rt, 
        #     df_conn_drill_rt, 
        #     self.bi_drill_utility.LABEL, 
        #     self.bi_drill_utility.CONNDTIME, 
        #     self.bi_drill_utility.CONNDTIME,
        #     name='conn_drill', 
        #     kpi_ascend=False,
        #     read_folder=self.bi_drill_utility.SAVE_FOLDER,
        #     save_folder=self.save_real_time_folder
        # )

        # df_conn_trip_rt_kpi = self.bi_drill_utility.get_kpi_color(
        #     df_rt, 
        #     df_conn_trip_rt, 
        #     self.bi_drill_utility.LABEL, 
        #     self.bi_drill_utility.CONNTTIME, 
        #     self.bi_drill_utility.CONNTTIME,
        #     name='conn_trip', 
        #     kpi_ascend=False,
        #     read_folder=self.bi_drill_utility.SAVE_FOLDER,
        #     save_folder=self.save_real_time_folder
        # )

        # # Plot limits para drilling
        # df_plot_lims_drill = pd.read_csv(
        #     f'{self.bi_drill_utility.SAVE_FOLDER}csv/{super_activity.lower()}_plot_lims.csv'
        # )
        # y_col = df_plot_lims_drill['y_col'].iloc[0]
        # ylims = df_plot_lims_drill['y_lims'].values
        # units_conversion = 1/60 if (y_col == self.bi_drill_utility.TIME) else 1
        # y_unit = ' (ft)' if y_col == self.bi_drill_utility.HD else ' (hr)'

        # # Referencias R1 y R2
        # df_R = pd.read_csv(
        #     f'{self.bi_drill_utility.SAVE_FOLDER}csv/ref1_ref2_highlight_{y_col}.csv'
        # )
        # R1, R2 = df_R[self.bi_drill_utility.MMDD].astype(str).values

        # # Plot drilling activity per stand
        # df_per_stand = self.bi_drill_utility.drill_per_stand(
        #     df_rt, 
        #     self.bi_drill_utility.STAND_LENGTH,
        #     y_col, 
        #     y_unit, 
        #     units_conversion,
        #     ylims, 
        #     [R1, R2], 
        #     bins_labels_dict,
        #     save_folder=self.save_real_time_folder
        # )

        # # Activity distribution drilling
        # df_time_dist_drill = self.bi_drill_utility.rig_activity_summary(
        #     df_rt, 
        #     self.bi_drill_utility.MMDD,
        #     super_activity, 
        #     'all',
        #     self.bi_drill_utility.WELLS_SELECT_ID, 
        #     self.bi_drill_utility.CURRENT_WELL_ID, 
        #     dtime_dict,
        #     refs=[R1, R2], 
        #     add_naming='_per_day',
        #     save_folder=self.save_real_time_folder
        # )

        # # Depth drilled
        # df_perform_drill = self.bi_drill_utility.compute_plot_depth(
        #     df_rt[(df_rt[self.bi_drill_utility.HD] != self.bi_drill_utility.NULL_VALUE)],
        #     ['ROTATE', 'SLIDE'],
        #     super_activity,
        #     dtime_dict,
        #     [R1, R2],
        #     save_folder=self.save_real_time_folder
        # )

        # # Tripping tab
        # super_activity = 'TRIP'

        # # Plot limits para tripping
        # df_plot_lims_drill = pd.read_csv(
        #     f'{self.bi_drill_utility.SAVE_FOLDER}csv/{super_activity.lower()}_plot_lims.csv'
        # )
        # y_col = df_plot_lims_drill['y_col'].iloc[0]
        # ylims = df_plot_lims_drill['y_lims'].values
        # units_conversion = 1/60 if (y_col == self.bi_drill_utility.TIME) else 1
        # y_unit = ' (ft)' if y_col == self.bi_drill_utility.HD else ' (hr)'

        # # Referencias R1 y R2
        # df_R = pd.read_csv(
        #     f'{self.bi_drill_utility.SAVE_FOLDER}csv/ref1_ref2_highlight_{y_col}.csv'
        # )
        # R1, R2 = df_R[self.bi_drill_utility.MMDD].astype(str).values

        # # Pipe speed y movimiento
        # df_trip_rt_kpi = self.bi_drill_utility.compute_pipe_speed_envelope(
        #     df_rt,
        #     df_trip_rt,
        #     self.bi_drill_utility.LABELct,
        #     'pipe_speed',
        #     save_folder=self.save_real_time_folder
        # )

        # # Actividades de tripping
        # df_circulate = self.bi_drill_utility.compute_trip_activity(
        #     df_rt,
        #     self.bi_drill_utility.CIRCTIME,
        #     ['CIR (static)'],
        #     save_folder=self.save_real_time_folder
        # )

        # df_wash = self.bi_drill_utility.compute_trip_activity(
        #     df_rt,
        #     self.bi_drill_utility.WASHTIME,
        #     ['WASH IN', 'WASH OUT'],
        #     save_folder=self.save_real_time_folder
        # )

        # df_ream = self.bi_drill_utility.compute_trip_activity(
        #     df_rt,
        #     self.bi_drill_utility.REAMTIME,
        #     ['REAM UP', 'REAM DOWN'],
        #     save_folder=self.save_real_time_folder
        # )

        # # Activity distribution tripping
        # df_time_dist_trip = self.bi_drill_utility.rig_activity_summary(
        #     df_rt,
        #     self.bi_drill_utility.MMDD,
        #     super_activity,
        #     'all',
        #     self.bi_drill_utility.WELLS_SELECT_ID,
        #     self.bi_drill_utility.CURRENT_WELL_ID,
        #     dtime_dict,
        #     refs=[R1, R2],
        #     add_naming='_per_day',
        #     save_folder=self.save_real_time_folder
        # )

        # # Depth drilled tripping
        # df_rt_pipe = df_rt.merge(
        #     df_trip_rt.reset_index()[
        #         [self.bi_drill_utility.LABELct, 
        #          self.bi_drill_utility.PIPESP, 
        #          self.bi_drill_utility.RSUBACT]
        #     ].rename(columns={self.bi_drill_utility.RSUBACT: self.bi_drill_utility.RSUBACT+'_trip'}),
        #     how='left',
        #     left_on=self.bi_drill_utility.LABELct,
        #     right_on=self.bi_drill_utility.LABELct
        # )

        # df_perform_trip = self.bi_drill_utility.compute_plot_depth(
        #     df_rt_pipe[df_rt_pipe[self.bi_drill_utility.BD] != self.bi_drill_utility.NULL_VALUE],
        #     ['TRIP IN', 'TRIP OUT'],
        #     super_activity,
        #     dtime_dict,
        #     [R1, R2],
        #     save_folder=self.save_real_time_folder
        # )

        end_time = time.time()
        print(f'\n\n*** Time it took to run bi_drill_main.py is {end_time-start_time} s.***')
        
        return self.bi_drill_utility.SAVE_FOLDER