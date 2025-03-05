import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from io import StringIO

import seaborn as sns
import matplotlib.pyplot as plt

from src.entities.bi_drill_utility import BI_Drill_Utility
from utils_backend import query_execute  # Para DB si lo usas

###############################################################################
# Lector concurrente de JSON en Redis
###############################################################################
json_thread_pool = ThreadPoolExecutor(max_workers=8)

_plan_df_lock = Lock()  # Caché si fuera necesario

def _read_json_from_redis(redis_client, redis_key: str, parse_dates=None) -> pd.DataFrame:
    """Lee un JSON orient='records' desde Redis y lo convierte a DF."""
    json_content = redis_client.get(redis_key)
    if not json_content:
        return pd.DataFrame()
    df = pd.read_json(StringIO(json_content), orient="records")
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def _read_json_from_redis_async(redis_client, redis_key, parse_dates=None):
    return _read_json_from_redis(redis_client, redis_key, parse_dates)


###############################################################################
# Función para guardar DF en Redis en formato JSON orient="records"
###############################################################################
def store_df_in_redis(df: pd.DataFrame, redis_client, base_key: str, subkey: str):
    """
    Convierte un DataFrame a JSON y lo almacena en:
       {base_key}:{subkey}
    """
    if df is None or df.empty:
        json_str = "[]"
    else:
        json_str = df.to_json(orient="records")
    redis_key = f"{base_key}:{subkey}"
    redis_client.set(redis_key, json_str)
    print(f"[DEBUG] Stored DataFrame (shape={df.shape}) in Redis key: {redis_key}")


class DrillDataProcessor:
    def __init__(self, redis_client, base_key: str):
        """
        :param redis_client: conexión a Redis.
        :param base_key: "input_data:client:project:stream:username:scenario" para leer.
        
        Nota: No usamos input_folder. Todo se basa en base_key de Redis.
        """
        self.redis_client = redis_client
        self.base_key = base_key  # base key de INPUT (para leer)

        # 1) Extraemos client, project, stream, etc. del base_key
        #    Asumiendo que base_key = "input_data:client:project:stream:username:scenario"
        splitted = base_key.split(":")
        if len(splitted) < 6 or splitted[0] != "input_data":
            raise ValueError(f"El base_key no cumple formato 'input_data:client:project:stream:username:scenario': {base_key}")

        self.client    = splitted[1]
        self.project   = splitted[2]
        self.stream    = splitted[3]
        self.username  = splitted[4]
        self.scenario  = splitted[5]

        # 2) Creamos un base_key de salida, ej. "output_data:client:project:stream:username:scenario:real_time_update"
        self.output_base_key = f"output_data:{self.client}:{self.project}:{self.stream}:{self.username}:{self.scenario}:real_time_update"
        print(f"[DEBUG] OUTPUT base key: {self.output_base_key}")

        # 3) Instanciamos la utilidad de perforación (sin input_folder)
        #    Basta con pasar un string vacío, o algo genérico
        self.bi_drill_utility = BI_Drill_Utility(base_key, redis_client)

        # 4) Tiempos/logs si deseas
        self.start_time = time.time()
        self.current_datetime = datetime.now()
        self.formatted_datetime = self.current_datetime.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

    def process(self):
        start_time = time.time()

        #####################################################################
        # 1) Definir "claves de INPUT" en Redis
        #####################################################################
        files_to_read = {
            "df_plan": {
                "redis_key": f'{self.base_key}:plan:time_depth_plan',
                "parse_dates": None
            },
            "df_direct_plan": {
                "redis_key": f'{self.base_key}:plan:directional_plan',
                "parse_dates": None
            },
            "df_form_plan": {
                "redis_key": f'{self.base_key}:plan:formation_tops_plan',
                "parse_dates": None
            },
            "df": {
                "redis_key": f'{self.base_key}:database:time_based_drill',
                "parse_dates": [self.bi_drill_utility.DATETIME]
            },
            "df_rt": {
                "redis_key": f'{self.base_key}:real_time:time_based_drill_current_well',
                "parse_dates": [self.bi_drill_utility.DATETIME]
            },
            "df_survey_rt": {
                "redis_key": f'{self.base_key}:real_time:official_survey_current_well',
                "parse_dates": None
            },
        }

        #####################################################################
        # 2) Lectura concurrente desde Redis
        #####################################################################
        futures = {}
        for key, info in files_to_read.items():
            future = json_thread_pool.submit(
                _read_json_from_redis_async,
                self.redis_client,
                info["redis_key"],
                info["parse_dates"]
            )
            futures[future] = key

        df_plan = df_direct_plan = df_form_plan = df = df_rt = df_survey_rt = None

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

        #####################################################################
        # 3) Lógica de tu script original (cálculos, merges, etc.)
        #####################################################################
        bit_sizes_max = [df_plan[self.bi_drill_utility.BS].unique().max()]
        formation_palette = sns.color_palette("husl", df_form_plan[self.bi_drill_utility.FORM].nunique())
        form_colors_dict = dict(zip(df_form_plan[self.bi_drill_utility.FORM].unique(), formation_palette))

        # Manejo de dummy data si NO_RT
        if self.bi_drill_utility.NO_RT:
            df_rt, df_survey_rt = self.bi_drill_utility.make_dummy_data(bit_sizes_max)
        else:
            if df_survey_rt.shape[0] <= 1:
                df_aux = pd.DataFrame(
                    columns=self.bi_drill_utility.survey_raw_cols,
                    data=np.zeros([2, len(self.bi_drill_utility.survey_raw_cols)]),
                    index=np.arange(1, 3)
                )
                df_survey_rt = df_survey_rt.append(df_aux)

        df_survey_rt.drop_duplicates(inplace=True)
        df_survey_rt[self.bi_drill_utility.WELLID] = self.bi_drill_utility.CURRENT_WELL_ID
        df_survey_rt = self.bi_drill_utility.calculate_tvd_ns_ew_dls(df_survey_rt)
        df_survey_rt[self.bi_drill_utility.UWDEP] = np.sqrt(
            df_survey_rt[self.bi_drill_utility.NS]**2 + df_survey_rt[self.bi_drill_utility.EW]**2
        )

        # Ajuste plan direccional
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

        # Añadir inc, azm, dls
        for feature in [self.bi_drill_utility.INC, self.bi_drill_utility.AZM, self.bi_drill_utility.DLS]:
            df_rt = self.bi_drill_utility.add_survey(df_rt, df_survey_rt, feature)

        # Procesar bit size y mud motor
        if not self.bi_drill_utility.READ_BS_MM_CASING:
            df_rt[self.bi_drill_utility.BS] = np.nan
            df_rt[self.bi_drill_utility.MM] = np.nan
            group_col = self.bi_drill_utility.TIME
            df_plan_group = df_plan.groupby(
                [self.bi_drill_utility.BS, self.bi_drill_utility.MM]
            )[group_col].agg(['min','max'])

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

        # Rig activity labels
        if self.bi_drill_utility.DTIME_RT is None:
            dtime_rt = (df_rt[self.bi_drill_utility.TIME].diff() * 60).round().dropna().astype(int).mode()[0]
        else:
            dtime_rt = self.bi_drill_utility.DTIME_RT

        if dtime_rt not in [1, 5, 10, 15]:
            warnings.warn(
                f"Check the data frequency of real-time data: atypical value of {dtime_rt} s automatically identified"
            )

        dtime_dict = {self.bi_drill_utility.CURRENT_WELL_ID: pd.Timedelta(seconds=dtime_rt)}

        # rig_design: leemos de Redis
        # "input_data:client:project:stream:username:scenario:database:rig_design"
        rig_design_key = f"{self.base_key}:database:rig_design"
        df_rig = _read_json_from_redis(self.redis_client, rig_design_key)
        if not df_rig.empty:
            df_rig_rt = df_rig.loc[df_rig[self.bi_drill_utility.RIG] == self.bi_drill_utility.RIG_NAME].squeeze()
        else:
            df_rig_rt = pd.DataFrame()

        if not self.bi_drill_utility.READ_CURRENT_RIG_LABELS and not df_rig_rt.empty:
            df_rt = self.bi_drill_utility.rig_labels(df_rt, df_rig_rt, self.bi_drill_utility.RIG_NAME, dtime_rt, real_time=False)

        # Reformatear df_rt
        miss_cols_rt = [col for col in self.bi_drill_utility.time_based_drill_cols if col not in df_rt.columns]
        for col in miss_cols_rt:
            df_rt[col] = np.nan
        df_rt = df_rt[self.bi_drill_utility.time_based_drill_cols]

        # Dummy data
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

        df_rt = self.bi_drill_utility.conn_activity_label(df_rt, 'CNX (trip)', self.bi_drill_utility.LABELct)
        df_rt = self.bi_drill_utility.conn_activity_label(df_rt, 'CNX (drill)', self.bi_drill_utility.LABELcd)
        df_rt = self.bi_drill_utility.conn_activity_label_between(df_rt, 'CNX (trip)', self.bi_drill_utility.LABELbtwn)

        # month_day
        df_rt[self.bi_drill_utility.MMDD] = (
            df_rt[self.bi_drill_utility.DATETIME].dt.month.astype(str)
            + '.' + df_rt[self.bi_drill_utility.DATETIME].dt.day.astype(str).str.zfill(2)
        )

        # stand_number
        bins = np.arange(0, df_rt[self.bi_drill_utility.HD].max() + self.bi_drill_utility.STAND_LENGTH, self.bi_drill_utility.STAND_LENGTH)
        labels = np.arange(1, len(bins))
        df_rt[self.bi_drill_utility.STAND] = pd.cut(df_rt[self.bi_drill_utility.HD], bins=bins, labels=labels)

        # rop_mse
        df_rt = self.bi_drill_utility.add_rop_mse(df, df_rt)

        # DUR
        min_time = pd.Timedelta(seconds=dtime_rt)
        duration_map = df_rt.groupby(self.bi_drill_utility.LABEL).apply(
            lambda x: x[self.bi_drill_utility.TIME].nunique() * min_time
        )
        df_rt[self.bi_drill_utility.DUR] = df_rt[self.bi_drill_utility.LABEL].map(duration_map)

        print('\n*** real-time data ***\n', df_rt.head())

        # Tablas de tripping y connection time
        df_wt_wt_rt = self.bi_drill_utility.compute_df_wt_wt(df_rt)
        df_trip_rt = self.bi_drill_utility.compute_df_trip(df_rt)
        df_conn_drill_rt = self.bi_drill_utility.compute_df_conn(
            df_rt, compute_col=self.bi_drill_utility.CONNDTIME, activity='CNX (drill)', dtime_dict=dtime_dict
        )
        df_conn_trip_rt = self.bi_drill_utility.compute_df_conn(
            df_rt, compute_col=self.bi_drill_utility.CONNTTIME, activity='CNX (trip)', dtime_dict=dtime_dict
        )

        # ---------------------------------------------------------------------
        # 4) Guardamos los resultados en Redis
        # ---------------------------------------------------------------------
        store_df_in_redis(df_rt,       self.redis_client, self.output_base_key, "time_based_drill_current_well_out")
        store_df_in_redis(df_survey_rt,self.redis_client, self.output_base_key, "official_survey_current_well_out")

        # bit_sizes
        current_bit_size = df_rt[self.bi_drill_utility.BS].iloc[-1]
        include_bits = df_rt[self.bi_drill_utility.BS].unique()
        bit_sizes = ['all'] + list(include_bits)
        bit_sizes = [bit for bit in bit_sizes if str(bit) != 'nan']

        # rig_activity_summary & kpi_boxplot, sin escribir disco
        dtime_dict_out = {self.bi_drill_utility.CURRENT_WELL_ID: pd.Timedelta(seconds=dtime_rt)}
        for super_activity in ['all', 'DRILL', 'TRIP']:
            for hole_diameter in bit_sizes:
                self.bi_drill_utility.rig_activity_summary(
                    df_rt,
                    self.bi_drill_utility.WELLID,
                    super_activity,
                    hole_diameter,
                    self.bi_drill_utility.WELLS_SELECT_ID,
                    self.bi_drill_utility.CURRENT_WELL_ID,
                    dtime_dict_out,
                    add_naming='_rt',
                    save_folder=None  # No guardas nada en disco
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
                save_folder=None  # No guardas en disco
            )

        end_time = time.time()
        print(f'\n\n*** Time it took to run bi_drill_main.py is {end_time - start_time} s.***')

        return self.output_base_key
