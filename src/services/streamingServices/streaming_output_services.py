import os
import json
import traceback
import pandas as pd
import time
from flask import Response
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importamos módulos propios
from src.entities.plan_loader import PlanLoader
from src.entities.config_backend import ConfigBackend
from src.entities.streaming_input import StreamingInput
from src.repositories.well_repository import WellRepository
from src.repositories.rig_repository import RigRepository
from src.repositories.streaming_repository import StreamingRepository
from src.entities.drill_data_processor import DrillDataProcessor
from src.repositories.connection_redis_repository import ConnectionRedisRepository
from src.entities.directory_structure_redis import DirectoryStructureRedis
from src.entities.read_redis_utils import read_json_from_redis
from src.entities.bi_drill_utility import BI_Drill_Utility

# Executor global para tareas concurrentes (si lo requieres)
executor = ThreadPoolExecutor(max_workers=10)

class StreamingData:
    def __init__(self, client, project, stream, username, scenario, number_of_rows):
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__number_of_rows = number_of_rows  # IMPORTANTE: Asegúrate de asignarlo, ej. 'aggregated' o un entero
        self.__api_name = 'get_input_data_streams'

        # Inicializamos los repositorios
        self.__well_repo = WellRepository(client, project, stream, username, scenario, self.__api_name)
        self.__rig_repo = RigRepository(client, project, stream, username, scenario, self.__api_name)
        self.__streaming_repo = StreamingRepository(client, project, stream, username, scenario, self.__api_name)

    def get_data(self):
        try:
            t0 = time.time()
            # 1. Conectar a Redis y obtener base_key de entrada/salida
            db = ConnectionRedisRepository()
            redis_connection = db.get_connection()
            ds = DirectoryStructureRedis(self.__client, self.__project, self.__stream, self.__username, self.__scenario, redis_connection)
            input_base_key = ds.create_directory_structure()  # Ejemplo: "input_data:client:project:..."
            print(f"[DEBUG] Input base key: {input_base_key}")

            # 2. Obtener y guardar datos (leídos de BD) en Redis
            data = self.__get_and_save_data(input_base_key, redis_connection)
            if data == {}:
                raise ValueError("No se encontraron datos en la base de datos.") #mnesaje de error en inglés

            current_measured_depth = data['current_measured_depth']

            # 3. Procesar datos de streaming: obtener el bit size y guardar inputs
            streaming_input = StreamingInput(self.__client, self.__project, self.__stream, self.__username, self.__scenario, self.__api_name, input_base_key, redis_connection)
            current_diameter, diameters_list = streaming_input.get_current_bit_size(current_measured_depth)
            streaming_input.create_inputs_json(current_diameter)
            print(f"[DEBUG] current_diameter: {current_diameter} diameters_list: {diameters_list}")

            # 4. Procesar datos de perforación (DrillDataProcessor)
            drill_processor = DrillDataProcessor(redis_connection, input_base_key)
            process_path = drill_processor.process()  # process_path es la base_key de salida
            print(f"[DEBUG] Output base key: {process_path}")

            # 5. Inicializar BI_Drill_Utility (para constantes y formateo)
            bi_drill_util = BI_Drill_Utility(input_base_key, redis_connection)

            # 6. Leer los DataFrames de salida desde Redis (ya invertidos)
            df_rt_tbd = read_json_from_redis(redis_connection, f"{process_path}:time_based_drill_current_well_out", 
                                               parse_dates=[bi_drill_util.DATETIME]).iloc[::-1]
            df_rt_os = read_json_from_redis(redis_connection, f"{process_path}:official_survey_current_well_out").iloc[::-1]

            # 7. Aplicar lógica de muestreo/limitación de filas
            df_rt_tbd = self.__apply_row_limit_or_aggregation(df_rt_tbd)
            df_rt_os = self.__apply_row_limit_or_aggregation(df_rt_os)

            # 8. Rellenar NaN y aplicar formateos
            df_rt_tbd = df_rt_tbd.fillna(-999.25)
            df_rt_os = df_rt_os.fillna(-999.25)
            if "datetime" in df_rt_tbd.columns:
                df_rt_tbd["datetime"] = pd.to_datetime(df_rt_tbd["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            if "datetime" in df_rt_os.columns:
                df_rt_os["datetime"] = pd.to_datetime(df_rt_os["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            if "duration" in df_rt_tbd.columns:
                df_rt_tbd["duration"] = df_rt_tbd["duration"].apply(lambda x: str(pd.Timedelta(minutes=x)))
            # Opcional: reintegrar índice si se requiere como columna
            df_rt_tbd = df_rt_tbd.reset_index().rename(columns={"index": "Unnamed: 0"})
            df_rt_os = df_rt_os.reset_index().rename(columns={"index": "Unnamed: 0"})

            # 9. Convertir DataFrames a diccionarios para la salida
            dict_rt_tbd = df_rt_tbd.to_dict(orient='records')
            dict_rt_os = df_rt_os.to_dict(orient='records')

            t1 = time.time()
            print(f"[DEBUG] Total processing time: {t1 - t0:.3f} s")
            return {
                "tbd": dict_rt_tbd,
                "os": dict_rt_os
            }

        except Exception as e:
            print("Traceback completo:")
            print(traceback.format_exc())
            raise ValueError(f"Error en get_data: {str(e)}")

    def __get_and_save_data(self, base_key, redis_connection):
        try:
            result_data = {}

            # Obtener datos de pozos y drill
            well_data = self.__well_repo.get_well_general()
            if well_data is not None:
                redis_connection.set(f"{base_key}:database:well_general", well_data.to_json(orient='records'))

            tbd_data = self.__well_repo.get_time_based_drill()
            if tbd_data is not None:
                redis_connection.set(f"{base_key}:database:time_based_drill", tbd_data.to_json(orient='records'))
                
            # Procesar plan
            plan_loader = PlanLoader(client=self.__client, project=self.__project, stream=self.__stream, username=self.__username, scenario=self.__scenario, base_key=base_key, redis_connection=redis_connection, api_name=self.__api_name)
            plan_loader.load_plan_files()
          
            # Rig design
            rig_data = self.__rig_repo.get_rig_design()
            if rig_data is not None:
                for col in ['filter_dict', 'filter_dict_1', 'filter_dict_5', 'filter_dict_10', 'filter_dict_15']:
                    if col in rig_data.columns:
                        rig_data[col] = rig_data[col].apply(lambda x: '{' + ', '.join(x) + '}' if isinstance(x, list) else str(x))
                redis_connection.set(f"{base_key}:database:rig_design", rig_data.to_json(orient='records'))

            # Streaming data
            real_time_tbd = self.__streaming_repo.get_real_time_tbd()
            if real_time_tbd is not None:
                redis_connection.set(f"{base_key}:real_time:time_based_drill_current_well", real_time_tbd.to_json(orient='records'))

            real_time_os = self.__streaming_repo.get_real_time_os()
            if real_time_os is not None:
                redis_connection.set(f"{base_key}:real_time:official_survey_current_well", real_time_os.to_json(orient='records'))

            current_measured_depth = self.__streaming_repo.get_current_measured_depth()
            if current_measured_depth is not None:
                result_data['current_measured_depth'] = current_measured_depth
                redis_connection.set(f"{base_key}:real_time:current_measured_depth", str(current_measured_depth))

            return result_data

        except Exception as e:
            raise ValueError(f"Error en get_and_save_data: {str(e)}")

    def __apply_row_limit_or_aggregation(self, df):
        # La función de muestreo se mantiene igual
        if self.__number_of_rows == 'aggregated':
            length = len(df)
            if length > 100000:
                df = df.iloc[::80, :]
            elif length > 50000:
                df = df.iloc[::40, :]
            elif length > 20000:
                df = df.iloc[::20, :]
            elif length > 10000:
                df = df.iloc[::10, :]
            elif length > 5000:
                df = df.iloc[::8, :]
            elif length > 2000:
                df = df.iloc[::4, :]
            elif length > 1000:
                df = df.iloc[::2, :]
            return df
        else:
            try:
                limit = int(self.__number_of_rows)
                return df.iloc[:limit]
            except ValueError:
                return df