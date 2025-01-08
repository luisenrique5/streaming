import pandas as pd
import numpy as np
from datetime import datetime
import os
import shutil
import time
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.entities.bi_drill_utility import BI_Drill_Utility

# -------------------------------------------------------------------------
# 1) Definimos un pool de hilos global para la lectura de CSV
# -------------------------------------------------------------------------
csv_thread_pool = ThreadPoolExecutor(max_workers=8)

# Lock para proteger escritura en los CSV de salida y evitar colisiones.
_output_write_lock = Lock()

def _read_csv_async(path, parse_dates=None):
    """Función auxiliar para leer CSV con pandas en un hilo del ThreadPool."""
    if parse_dates:
        return pd.read_csv(path, parse_dates=parse_dates)
    else:
        return pd.read_csv(path)

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
            print(
                f"\n========== RUN 2. MACHINE LEARNING CODE EXECUTION  | INFO | PROCESS | "
                f"100002 | {self.client} | {self.project} | {self.stream} | {self.username} | {self.scenario} | "
                f"real_time_part_tablesupdate.sh\n"
            )
        else:
            raise ValueError("La estructura del input_folder no coincide con el formato esperado")

        for folder in ['plot', 'csv']:
            folder_path = f'{self.save_real_time_folder}{folder}'
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)

    def process(self):
        start_time = time.time()

        # ---------------------------------------------------------------------
        # 2) Lectura concurrente de CSV
        # ---------------------------------------------------------------------
        # Definimos las rutas y parse_dates de cada archivo que leeremos.
        # Nota: Ajusta las rutas si tu estructura es distinta.
        files_to_read = {
            "df_plan": {
                "path": f'{self.input_folder}plan/time_depth_plan.csv',
                "parse_dates": None
            },
            "df_direct_plan": {
                "path": f'{self.input_folder}plan/directional_plan.csv',
                "parse_dates": None
            },
            "df_form_plan": {
                "path": f'{self.input_folder}plan/formation_tops_plan.csv',
                "parse_dates": None
            },
            "df": {
                "path": f'{self.input_folder}database/time_based_drill.csv',
                "parse_dates": [self.bi_drill_utility.DATETIME]
            },
            "df_rt": {
                "path": f'{self.input_folder}real_time/time_based_drill_current_well.csv',
                "parse_dates": [self.bi_drill_utility.DATETIME]
            },
            "df_survey_rt": {
                "path": f'{self.input_folder}real_time/official_survey_current_well.csv',
                "parse_dates": None
            },
        }

        # Disparamos las lecturas en paralelo
        futures = {}
        for key, info in files_to_read.items():
            future = csv_thread_pool.submit(_read_csv_async, info["path"], info["parse_dates"])
            futures[future] = key

        # Inicializamos variables en None, se completarán cuando termine cada future
        df_plan = df_direct_plan = df_form_plan = df = df_rt = df_survey_rt = None

        # Esperamos a que terminen las lecturas
        for future in as_completed(futures):
            key = futures[future]
            result = future.result()
            if key == "df_plan":
                df_plan = result
            elif key == "df_direct_plan":
                df_direct_plan = result
            elif key == "df_form_plan":
                df_form_plan = result
            elif key == "df":
                df = result
            elif key == "df_rt":
                df_rt = result
            elif key == "df_survey_rt":
                df_survey_rt = result

        # ---------------------------------------------------------------------
        # 3) Ahora seguimos con la misma lógica de tu código original
        # ---------------------------------------------------------------------

        # Datos históricos (df) ya están en variables
        # 'df': time_based_drill con parse_dates
        # 'df_plan': time_depth_plan
        # 'df_direct_plan': directional_plan
        # 'df_form_plan': formation_tops_plan
        # 'df_rt': time_based_drill_current_well con parse_dates
        # 'df_survey_rt': official_survey_current_well (sin parse_dates)

        bit_sizes_max = [df_plan[self.bi_drill_utility.BS].unique().max()]
        formation_palette = sns.color_palette("husl", df_form_plan[self.bi_drill_utility.FORM].nunique())
        form_colors_dict = dict(zip(df_form_plan[self.bi_drill_utility.FORM].unique(), formation_palette))

        # Manejo de dummy data si NO_RT
        if self.bi_drill_utility.NO_RT:
            df_rt, df_survey_rt = self.bi_drill_utility.make_dummy_data(bit_sizes_max)
        else:
            # Ajuste si df_survey_rt es muy pequeño
            if df_survey_rt.shape[0] <= 1:
                df_aux = pd.DataFrame(
                    columns=self.bi_drill_utility.survey_raw_cols,
                    data=np.zeros([2, len(self.bi_drill_utility.survey_raw_cols)]),
                    index=np.arange(1, 3)
                )
                df_survey_rt = df_survey_rt.append(df_aux)

        df_survey_rt.drop_duplicates(inplace=True)

        # Procesar df_survey_rt
        df_survey_rt[self.bi_drill_utility.WELLID] = self.bi_drill_utility.CURRENT_WELL_ID
        df_survey_rt = self.bi_drill_utility.calculate_tvd_ns_ew_dls(df_survey_rt)
        df_survey_rt[self.bi_drill_utility.UWDEP] = np.sqrt(
            df_survey_rt[self.bi_drill_utility.NS]**2 + df_survey_rt[self.bi_drill_utility.EW]**2
        )

        # Ajuste a df_direct_plan
        df_direct_plan[self.bi_drill_utility.HD + '_max'] = df_direct_plan[self.bi_drill_utility.HD].shift(-1).fillna(
            df_survey_rt[self.bi_drill_utility.HD].max()
        )
        for index, row in df_direct_plan.iterrows():
            cond = (
                (df_survey_rt[self.bi_drill_utility.HD] >= row[self.bi_drill_utility.HD]) &
                (df_survey_rt[self.bi_drill_utility.HD] <= row[self.bi_drill_utility.HD + '_max'])
            )
            df_survey_rt.loc[cond, self.bi_drill_utility.WSEC] = row[self.bi_drill_utility.WSEC]
        df_direct_plan.drop(columns=[self.bi_drill_utility.HD + '_max'], inplace=True)

        # Reformatear df_survey_rt
        miss_cols = [col for col in self.bi_drill_utility.survey_cols if col not in df_survey_rt.columns]
        for col in miss_cols:
            df_survey_rt[col] = np.nan
        df_survey_rt = df_survey_rt[self.bi_drill_utility.survey_cols]

        # Procesar df_rt
        df_rt[self.bi_drill_utility.DATETIME] = pd.to_datetime(
            df_rt[self.bi_drill_utility.DATETIME].apply(lambda x: str(x)[:19]),
            format='%Y-%m-%d %H:%M:%S'
        )
        df_rt.drop_duplicates(self.bi_drill_utility.DATETIME, inplace=True)
        df_rt.sort_values(by=[self.bi_drill_utility.DATETIME], inplace=True)

        df_rt[self.bi_drill_utility.WELLID] = self.bi_drill_utility.CURRENT_WELL_ID
        df_rt[self.bi_drill_utility.TIME] = (
            df_rt[self.bi_drill_utility.DATETIME].diff().dt.seconds / 60
        ).cumsum().fillna(0)

        time_bins = np.arange(0, int(np.ceil(df_rt[self.bi_drill_utility.TIME].max() / 60 / 24)) + 1, 1)
        df_rt[self.bi_drill_utility.DAYN] = pd.cut(
            df_rt[self.bi_drill_utility.TIME] / 60 / 24,
            bins=time_bins,
            labels=time_bins[1:],
            include_lowest=True
        ).astype(int)

        df_rt = self.bi_drill_utility.add_tvd_well_section(df_rt, df_survey_rt)
        df_rt.sort_values(by=[self.bi_drill_utility.DATETIME], inplace=True)

        # Añadir features del survey
        for feature in [self.bi_drill_utility.INC, self.bi_drill_utility.AZM, self.bi_drill_utility.DLS]:
            df_rt = self.bi_drill_utility.add_survey(df_rt, df_survey_rt, feature)

        # Procesar bit size y mud motor (si READ_BS_MM_CASING es False)
        if not self.bi_drill_utility.READ_BS_MM_CASING:
            df_rt[self.bi_drill_utility.BS] = np.nan
            df_rt[self.bi_drill_utility.MM] = np.nan

            group_col = self.bi_drill_utility.TIME
            df_plan_group = df_plan.groupby(
                [self.bi_drill_utility.BS, self.bi_drill_utility.MM]
            )[group_col].agg(['min', 'max'])
            for index, row in df_plan_group.iterrows():
                cond = (df_rt[group_col] >= row['min']) & (df_rt[group_col] <= row['max'])
                df_rt.loc[cond, [self.bi_drill_utility.BS, self.bi_drill_utility.MM]] = index

            df_rt[self.bi_drill_utility.BS].fillna(method='ffill', inplace=True)
            df_rt[self.bi_drill_utility.MM].fillna(method='ffill', inplace=True)
            df_rt[self.bi_drill_utility.CASING] = np.nan

        # Procesar casing
        df_plan[self.bi_drill_utility.TIME + '_max'] = df_plan[self.bi_drill_utility.TIME].shift(-1).fillna(
            df_rt[self.bi_drill_utility.TIME].max()
        )
        for index, row in df_plan.iterrows():
            cond = (
                (df_rt[self.bi_drill_utility.TIME] >= row[self.bi_drill_utility.TIME]) &
                (df_rt[self.bi_drill_utility.TIME] <= row[self.bi_drill_utility.TIME + '_max'])
            )
            df_rt.loc[cond, self.bi_drill_utility.CASING] = row[self.bi_drill_utility.CASING]
        df_plan.drop(columns=[self.bi_drill_utility.TIME + '_max'], inplace=True)

        # Calcular MSE y formaciones
        df_rt = self.bi_drill_utility.calculate_mse(df_rt)
        df_rt[self.bi_drill_utility.FORM] = np.nan

        df_form_plan[self.bi_drill_utility.FORMBOT] = df_form_plan[self.bi_drill_utility.FORMTOP].shift(-1).fillna(
            df_rt[self.bi_drill_utility.HD].max()
        )
        df_form_plan[self.bi_drill_utility.FORMBOTTIME] = df_form_plan[self.bi_drill_utility.FORMTOPTIME].shift(-1).fillna(
            df_rt[self.bi_drill_utility.TIME].max()
        )

        for _, row in df_form_plan.iterrows():
            d_min, d_max = row[self.bi_drill_utility.FORMTOP], row[self.bi_drill_utility.FORMBOT]
            cond = (df_rt[self.bi_drill_utility.HD] >= d_min) & (df_rt[self.bi_drill_utility.HD] < d_max)
            df_rt.loc[cond, self.bi_drill_utility.FORM] = row[self.bi_drill_utility.FORM]

        # Procesar rig activity labels
        if self.bi_drill_utility.DTIME_RT is None:
            dtime_rt = (df_rt[self.bi_drill_utility.TIME].diff() * 60).round().dropna().astype(int).mode()[0]
        else:
            dtime_rt = self.bi_drill_utility.DTIME_RT

        if dtime_rt not in [1, 5, 10, 15]:
            warnings.warn(
                f"Check the data frequence of real-time data: atypical value of {dtime_rt} s automatically identified"
            )

        dtime_dict = {self.bi_drill_utility.CURRENT_WELL_ID: pd.Timedelta(seconds=dtime_rt)}

        # Leer rig_design
        df_rig = pd.read_csv(f'{self.input_folder}database/rig_design.csv')
        df_rig_rt = df_rig.loc[df_rig[self.bi_drill_utility.RIG] == self.bi_drill_utility.RIG_NAME].squeeze()

        if not self.bi_drill_utility.READ_CURRENT_RIG_LABELS:
            df_rt = self.bi_drill_utility.rig_labels(
                df_rt, df_rig_rt, self.bi_drill_utility.RIG_NAME, dtime_rt, real_time=False
            )

        # Reformatear datos en tiempo real
        miss_cols = [
            col for col in self.bi_drill_utility.time_based_drill_cols if col not in df_rt.columns
        ]
        for col in miss_cols:
            df_rt[col] = np.nan
        df_rt = df_rt[self.bi_drill_utility.time_based_drill_cols]

        # Merge con dummy data
        df_rt_dummy, df_survey_rt_dummy = self.bi_drill_utility.make_dummy_data(
            bit_sizes_max, day=df_rt[self.bi_drill_utility.DATETIME].min()
        )
        df_rt = pd.concat([df_rt, df_rt_dummy[self.bi_drill_utility.time_based_drill_cols]]).sort_values(
            by=[self.bi_drill_utility.DATETIME]
        )
        df_survey_rt = pd.concat([df_survey_rt, df_survey_rt_dummy[self.bi_drill_utility.survey_cols]]).sort_values(
            by=[self.bi_drill_utility.HD]
        )

        df_rt[self.bi_drill_utility.WELLID] = self.bi_drill_utility.CURRENT_WELL_ID
        df_survey_rt[self.bi_drill_utility.WELLID] = self.bi_drill_utility.CURRENT_WELL_ID
        df_rt.drop_duplicates(self.bi_drill_utility.DATETIME, inplace=True)

        # Procesar conexiones
        df_rt = self.bi_drill_utility.conn_activity_label(
            df_rt, activity='CNX (trip)', LABEL=self.bi_drill_utility.LABELct
        )
        df_rt = self.bi_drill_utility.conn_activity_label(
            df_rt, activity='CNX (drill)', LABEL=self.bi_drill_utility.LABELcd
        )
        df_rt = self.bi_drill_utility.conn_activity_label_between(
            df_rt, activity='CNX (trip)', LABEL=self.bi_drill_utility.LABELbtwn
        )

        # Añadir month_day
        df_rt[self.bi_drill_utility.MMDD] = (
            df_rt[self.bi_drill_utility.DATETIME].dt.month.astype('str')
            + '.'
            + df_rt[self.bi_drill_utility.DATETIME].dt.day.astype('str').apply(lambda x: x.zfill(2))
        )

        # Calcular stand number
        bins = np.arange(0, df_rt[self.bi_drill_utility.HD].max() + self.bi_drill_utility.STAND_LENGTH, self.bi_drill_utility.STAND_LENGTH)
        labels = np.arange(1, len(bins))
        bins_labels_dict = dict(zip(labels, bins + self.bi_drill_utility.STAND_LENGTH / 2))
        df_rt[self.bi_drill_utility.STAND] = pd.cut(df_rt[self.bi_drill_utility.HD], bins=bins, labels=labels)

        # Añadir rop_mse class
        df_rt = self.bi_drill_utility.add_rop_mse(df, df_rt)

        # Añadir duración
        min_time = pd.Timedelta(seconds=dtime_rt)
        duration_map = df_rt.groupby(self.bi_drill_utility.LABEL).apply(
            lambda x: x[self.bi_drill_utility.TIME].nunique() * min_time
        )
        df_rt[self.bi_drill_utility.DUR] = df_rt[self.bi_drill_utility.LABEL].map(duration_map)

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

        # ---------------------------------------------------------------------
        # 4) Guardamos resultados en CSV con un lock para evitar colisiones
        # ---------------------------------------------------------------------
        with _output_write_lock:
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

        end_time = time.time()
        print(f'\n\n*** Time it took to run bi_drill_main.py is {end_time - start_time} s.***')

        return self.bi_drill_utility.SAVE_FOLDER
