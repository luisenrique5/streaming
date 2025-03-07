import os
import json
import traceback
import logging
import time
import concurrent.futures
from typing import Union, Dict, Any

import pandas as pd
from flask import Response

# Módulos propios
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

from src.repositories.postgressql import ConnectionSqlAlchemyRepository
from src.repositories.database_connection import DatabaseConnection
# ↑ Asegúrate de que este sea el nombre y ruta real

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StreamingData:
    def __init__(
        self,
        client: str,
        project: str,
        stream: str,
        username: str,
        scenario: str,
        number_of_rows: Union[str, int]
    ):
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__number_of_rows = number_of_rows
        self.__api_name = 'get_input_data_streams'

        # 1) Creamos la conexión con SQLAlchemy (similar al snippet que mostraste)
        #    Nota: si la BD se llama, p.ej., "streaming_drillbi_{client}_{project}_db",
        #    ajusta aquí la lógica para armar el nombre.
      
        db_name = "rig_design_db"
        db = ConnectionSqlAlchemyRepository(db_name)
        engine = db.get_connection()
        db_connection_value = DatabaseConnection(engine)
        self.__rig_repo = RigRepository(db_connection_value, client, project, stream, username, scenario, self.__api_name)
        self.__streaming_repo = StreamingRepository(client, project, stream, username, scenario, self.__api_name)
        # db_name = f"rig_design_db"
        # db = ConnectionSqlAlchemyRepository(db_name)
        # engine = db.get_connection()
        # self.__streaming_repo = StreamingRepository(engine, client, project, stream, username, scenario, self.__api_name)

    def get_data(self) -> Dict[str, Any]:
        try:
            t0 = time.time()

            # 1. Conexión a Redis y obtención de claves
            print("Conexión a Redis y obtención de claves")
            db = ConnectionRedisRepository()
            redis_connection = db.get_connection()
            ds = DirectoryStructureRedis(
                self.__client, self.__project, self.__stream,
                self.__username, self.__scenario, redis_connection
            )
            input_base_key = ds.create_directory_structure()
            logger.debug(f"Input base key: {input_base_key}")

            # 2. Obtener y guardar datos (leídos de BD) en Redis
            data = self.__get_and_save_data(input_base_key, redis_connection)
            if not data:
                raise ValueError("No se encontraron datos en la base de datos.")

            current_measured_depth = data["current_measured_depth"]

            # 3. Procesar datos de streaming (bit size, etc.)
            streaming_input = StreamingInput(
                self.__client, self.__project, self.__stream,
                self.__username, self.__scenario, self.__api_name,
                input_base_key, redis_connection
            )
            current_diameter, diameters_list = streaming_input.get_current_bit_size(current_measured_depth)
            streaming_input.create_inputs_json(current_diameter)
            logger.debug(f"current_diameter: {current_diameter}, diameters_list: {diameters_list}")

            # 4. Procesar datos de perforación
            drill_processor = DrillDataProcessor(redis_connection, input_base_key)
            process_path = drill_processor.process()  # Devuelve la base_key de salida
            logger.debug(f"Output base key: {process_path}")

            # 5. Inicializar utilidades para formateo
            bi_drill_util = BI_Drill_Utility(input_base_key, redis_connection)

            # 6. Leer los DataFrames de salida desde Redis (invertidos)
            df_rt_tbd = read_json_from_redis(
                redis_connection,
                f"{process_path}:time_based_drill_current_well_out",
                parse_dates=[bi_drill_util.DATETIME]
            ).iloc[::-1]

            df_rt_os = read_json_from_redis(
                redis_connection,
                f"{process_path}:official_survey_current_well_out"
            ).iloc[::-1]

            # 7. Aplicar muestreo/limitación de filas
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
                df_rt_tbd["duration"] = df_rt_tbd["duration"].apply(
                    lambda x: str(pd.Timedelta(minutes=x))
                )

            # Reintegrar índice si se requiere
            df_rt_tbd = df_rt_tbd.reset_index().rename(columns={"index": "Unnamed: 0"})
            df_rt_os = df_rt_os.reset_index().rename(columns={"index": "Unnamed: 0"})

            dict_rt_tbd = df_rt_tbd.to_dict(orient="records")
            dict_rt_os = df_rt_os.to_dict(orient="records")

            t1 = time.time()
            logger.info(f"Tiempo total de procesado: {t1 - t0:.3f}s")

            return {"tbd": dict_rt_tbd, "os": dict_rt_os}

        except Exception as e:
            logger.error("Error en get_data", exc_info=True)
            raise ValueError(f"Error en get_data: {str(e)}") from e

    def __get_and_save_data(self, base_key: str, redis_connection) -> Dict[str, Any]:
        try:
            result_data = {}
            plan_loader = PlanLoader(
                client=self.__client,
                project=self.__project,
                stream=self.__stream,
                username=self.__username,
                scenario=self.__scenario,
                base_key=base_key,
                redis_connection=redis_connection,
                api_name=self.__api_name
            )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                db_name = "well_general_db"
                db = ConnectionSqlAlchemyRepository(db_name)
                engine = db.get_connection()
                db_connection_value = DatabaseConnection(engine)
                well_repo = WellRepository(db_connection_value, self.__client, self.__project, self.__stream, self.__username, self.__scenario, self.__api_name)
                queryParameters_future = executor.submit(well_repo.get_well_general)
                queryParameters = queryParameters_future.result()
                well_data = db_connection_value.execute_query(queryParameters)
                db_name = "time_based_drill_db"
                db = ConnectionSqlAlchemyRepository(db_name)
                engine = db.get_connection()
                db_connection_value = DatabaseConnection(engine)
                well_repo = WellRepository(db_connection_value, self.__client, self.__project, self.__stream, self.__username, self.__scenario, self.__api_name)
                queryParameters_future = executor.submit(well_repo.get_time_based_drill)
                queryParameters = queryParameters_future.result()
                tbd_data = db_connection_value.execute_query(queryParameters)
                future_plan = executor.submit(plan_loader.load_plan_files)
                queryParameters = executor.submit(self.__rig_repo.get_rig_design)
                queryParameters = queryParameters_future.result()
                rig_data = db_connection_value.execute_query(queryParameters)
                rig_data = pd.DataFrame(rig_data)
                future_rt_tbd = executor.submit(self.__streaming_repo.get_real_time_tbd)
                future_rt_os = executor.submit(self.__streaming_repo.get_real_time_os)
                future_cmd = executor.submit(self.__streaming_repo.get_current_measured_depth)

                # plan_loader no retorna objeto, se ejecuta y listo
                future_plan.result()
                real_time_tbd = future_rt_tbd.result()
                real_time_os = future_rt_os.result()
                current_measured_depth = future_cmd.result()

            pipe = redis_connection.pipeline()
            if well_data is not None:
                df_well = pd.DataFrame(well_data)
                pipe.set(f"{base_key}:database:well_general", df_well.to_json(orient='records'))
                
            if tbd_data is not None:
                df_well = pd.DataFrame(tbd_data)
                pipe.set(f"{base_key}:database:time_based_drill", df_well.to_json(orient='records'))    

            if rig_data is not None:
                for col in ['filter_dict', 'filter_dict_1', 'filter_dict_5', 'filter_dict_10', 'filter_dict_15']:
                    if col in rig_data.columns:
                        rig_data[col] = rig_data[col].apply(lambda x: '{' + ', '.join(x) + '}' if isinstance(x, list) else str(x))
                redis_connection.set(f"{base_key}:database:rig_design", rig_data.to_json(orient='records'))


            if real_time_tbd is not None:
                pipe.set(
                    f"{base_key}:real_time:time_based_drill_current_well",
                    real_time_tbd.to_json(orient='records')
                )
            if real_time_os is not None:
                pipe.set(
                    f"{base_key}:real_time:official_survey_current_well",
                    real_time_os.to_json(orient='records')
                )

            if current_measured_depth is not None:
                result_data["current_measured_depth"] = current_measured_depth
                pipe.set(f"{base_key}:real_time:current_measured_depth", str(current_measured_depth))

            pipe.execute()
            return result_data

        except Exception as e:
            logger.error("Error en __get_and_save_data", exc_info=True)
            raise ValueError(f"Error en get_and_save_data: {str(e)}") from e

    def __apply_row_limit_or_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica muestreo escalonado o limitación de filas según `self.__number_of_rows`.
        Retorna el DataFrame resultante.
        """
        if self.__number_of_rows == 'aggregated':
            length = len(df)
            if length > 100000:
                return df.iloc[::80, :]
            elif length > 50000:
                return df.iloc[::40, :]
            elif length > 20000:
                return df.iloc[::20, :]
            elif length > 10000:
                return df.iloc[::10, :]
            elif length > 5000:
                return df.iloc[::8, :]
            elif length > 2000:
                return df.iloc[::4, :]
            elif length > 1000:
                return df.iloc[::2, :]
            else:
                return df
        else:
            # Si es un número, devolvemos las primeras N filas
            try:
                limit = int(self.__number_of_rows)
                return df.iloc[:limit]
            except ValueError:
                # En caso de error de conversión, devolvemos todo
                return df
