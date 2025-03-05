import os
import json
import traceback
import pandas as pd
from flask import Response

from src.entities.session_manager import SessionManager
from src.entities.plan_loader import PlanLoader
from src.entities.config_backend import ConfigBackend
from src.entities.streaming_input import StreamingInput
from src.repositories.well_repository import WellRepository
from src.repositories.rig_repository import RigRepository
from src.repositories.streaming_repository import StreamingRepository
from src.entities.drill_data_processor import DrillDataProcessor
from utils_backend import logging_report, modify_strings

from src.repositories.connection_redis_repository import ConnectionRedisRepository
from src.entities.directory_structure_redis import DirectoryStructureRedis


class StreamingData:
    def __init__(self, client, project, stream, username, scenario, number_of_rows):
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__number_of_rows = number_of_rows
        self.__api_name = 'get_input_data_streams'

        self.__well_repo = WellRepository(client, project, stream, username, scenario, self.__api_name)
        self.__rig_repo = RigRepository(client, project, stream, username, scenario, self.__api_name)
        self.__streaming_repo = StreamingRepository(client, project, stream, username, scenario, self.__api_name)

    def get_data(self):
        try:
            db = ConnectionRedisRepository()
            redis_connection = db.get_connection()
            ds = DirectoryStructureRedis(self.__client, self.__project, self.__stream, self.__username, self.__scenario, redis_connection)
            base_key = ds.create_directory_structure()
            # data = self.__get_and_save_data(base_key, redis_connection)
            
            # if data == {}:
            #     raise ValueError("No se encontraron datos en la base de datos.")
            # else:
            current_measured_depth = 9656.55 #data['current_measured_depth']
            streaming_input = StreamingInput(self.__client, self.__project, self.__stream, self.__username, self.__scenario, self.__api_name, base_key, redis_connection)
            current_diameter, diameters_list = streaming_input.get_current_bit_size(current_measured_depth)
            streaming_input.create_inputs_json(current_diameter)
            print(f"current_diameter: {current_diameter} diameters_list: {diameters_list}")
                
            # # 4.5) Ajustes si hay measured_depth
            # if 'current_measured_depth' in data:
            #     current_measured_depth = data['current_measured_depth']
            #     streaming_input = StreamingInput(
            #         self.client,
            #         self.project,
            #         self.stream,
            #         self.username,
            #         self.scenario,
            #         self.api_name
            #     )
            #     current_hole_diameter, _ = streaming_input.get_current_bit_size(
            #         input_folder, current_measured_depth
            #     )
            #     streaming_input.create_inputs_json(input_folder, current_hole_diameter)

            # # 4.6) Procesa la data con DrillDataProcessor
            process_drilling_data = DrillDataProcessor(redis_connection,base_key)
            process_path = process_drilling_data.process()
            print(f"process_path: {process_path}")

            # # 4.7) Carga CSVs resultantes
            # rt_tbd_path = os.path.join(process_path, 'real_time_update', 'time_based_drill_current_well_out.csv')
            # rt_os_path = os.path.join(process_path, 'real_time_update', 'official_survey_current_well_out.csv')

            # df_rt_tbd = pd.read_csv(rt_tbd_path).iloc[::-1]  # invertimos orden
            # df_rt_os = pd.read_csv(rt_os_path).iloc[::-1]

            # # 4.8) Aplica agregación / limitación de filas
            # df_rt_tbd = self.__apply_row_limit_or_aggregation(df_rt_tbd)
            # df_rt_os = self.__apply_row_limit_or_aggregation(df_rt_os)

            # # 4.9) Rellena NaN
            # df_rt_tbd = df_rt_tbd.fillna(-999.25)
            # df_rt_os = df_rt_os.fillna(-999.25)

            # dict_rt_tbd = df_rt_tbd.to_dict(orient='records')
            # dict_rt_os = df_rt_os.to_dict(orient='records')

            # # 4.10) Retorna un dict con la data
            # return {
            #     "tbd": dict_rt_tbd,
            #     "os": dict_rt_os
            # }

        except Exception as e:
            # Loggea y relanza para que el controller maneje la excepción
            print(f"Error en calculate_get_streaming_output: {str(e)}")
            print("Traceback completo:")
            print(traceback.format_exc())
            raise

    def __get_and_save_data(self, base_key, redis_connection):
        try:
            result_data = {}

            well_data = self.__well_repo.get_well_general()
            if well_data is not None:
                redis_key = f"{base_key}:database:well_general"
                redis_connection.set(redis_key, well_data.to_json(orient='records'))

            tbd_data = self.__well_repo.get_time_based_drill()
            if tbd_data is not None:
                redis_key = f"{base_key}:database:time_based_drill"
                redis_connection.set(redis_key, tbd_data.to_json(orient='records'))
                
            plan_loader = PlanLoader(client=self.__client, project=self.__project, stream=self.__stream, username=self.__username, scenario=self.__scenario, base_key=base_key, redis_connection=redis_connection, api_name=self.__api_name)
            plan_loader.load_plan_files()
          
            rig_data = self.__rig_repo.get_rig_design()
            if rig_data is not None:
                for col in ['filter_dict', 'filter_dict_1', 'filter_dict_5',
                            'filter_dict_10', 'filter_dict_15']:
                    if col in rig_data.columns:
                        rig_data[col] = rig_data[col].apply(lambda x: '{' + ', '.join(x) + '}' if isinstance(x, list) else str(x))
                redis_key = f"{base_key}:database:rig_design"
                redis_connection.set(redis_key, rig_data.to_json(orient='records'))

            real_time_tbd = self.__streaming_repo.get_real_time_tbd()
            if real_time_tbd is not None:
                redis_key = f"{base_key}:real_time:time_based_drill_current_well"
                redis_connection.set(redis_key, real_time_tbd.to_json(orient='records'))

            real_time_os = self.__streaming_repo.get_real_time_os()
            if real_time_os is not None:
                redis_key = f"{base_key}:real_time:official_survey_current_well"
                redis_connection.set(redis_key, real_time_os.to_json(orient='records'))

            current_measured_depth = self.__streaming_repo.get_current_measured_depth()
            if current_measured_depth is not None:
                result_data['current_measured_depth'] = current_measured_depth
                depth_key = f"{base_key}:real_time:current_measured_depth"
                redis_connection.set(depth_key, str(current_measured_depth))

            return result_data

        except Exception as e:
            raise ValueError(f"Error en get_and_save_data: {str(e)}")


    # -------------------------------------------------------------------------
    # 3) Agregación o limitación de filas en DataFrames
    # -------------------------------------------------------------------------
    def __apply_row_limit_or_aggregation(self, df):
        """
        Aplica la lógica de agregación (por 'aggregated') o limitación
        de filas (por self.number_of_rows) a un DataFrame.
        """
        # Caso 'aggregated': reducimos filas gradualmente según su longitud.
        if self.number_of_rows == 'aggregated':
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
            # Caso limitación a X filas
            try:
                limit = int(self.number_of_rows)
                return df.iloc[:limit]
            except ValueError:
                # Si no es entero y tampoco es 'aggregated', devolvemos df completo
                return df