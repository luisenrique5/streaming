import os
import json
import traceback
import pandas as pd
from flask import Response

# Entities y repositorios (ajusta si cambian las rutas)
from src.entities.streaming_environment import StreamingEnvironment
from src.entities.session_manager import SessionManager
from src.entities.plan_loader import PlanLoader
from src.entities.config_backend import ConfigBackend
from src.entities.directory_structure import DirectoryStructure
from src.entities.streaming_input import StreamingInput
from src.repositories.well_repository import WellRepository
from src.repositories.rig_repository import RigRepository
from src.repositories.streaming_repository import StreamingRepository
from src.entities.real_time_part import DrillDataProcessor
from utils_backend import logging_report, modify_strings


class GetStreamingOutput:
    def __init__(self, client, project, stream, username, scenario, number_of_rows):
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.number_of_rows = number_of_rows
        self.api_name = 'get_input_data_streams'

        # Inicialización de repositorios
        self.well_repo = WellRepository(
            client, project, stream, username, scenario, self.api_name
        )
        self.rig_repo = RigRepository(
            client, project, stream, username, scenario, self.api_name
        )
        self.streaming_repo = StreamingRepository(
            client, project, stream, username, scenario, self.api_name
        )

    # -------------------------------------------------------------------------
    # 1) Configuración del environment y SessionManager
    # -------------------------------------------------------------------------
    def __set_environment_variables(self):
        """
        Configura variables de entorno en SessionManager para que otras partes
        del sistema puedan utilizarlas.
        """
        session_manager = SessionManager()
        variables = {
            'CLIENT': self.client,
            'PROJECT': self.project,
            'STREAM': self.stream,
            'USERNAME': self.username,
            'SCENARIO': self.scenario
        }
        for key, value in variables.items():
            session_manager.set_variable(key, value)
            logging_report(f'Set {key}: {value}', 'DEBUG', self.api_name)

        return session_manager

    def __setup_directories(self, username, scenario):
        """
        Crea la estructura de directorios necesarios para el proceso
        (base, database, real_time, etc.).
        """
        dir_structure = DirectoryStructure(
            self.client, self.project, self.stream, username, scenario
        )
        return dir_structure.create_directory_structure()

    # -------------------------------------------------------------------------
    # 2) Lógica para obtener y guardar datos
    # -------------------------------------------------------------------------
    def __get_and_save_data(self, input_folder):
        """
        Obtiene datos de varios repositorios y los guarda en CSV.  
        Devuelve un diccionario con algunos datos relevantes (por ejemplo,
        current_measured_depth).
        """
        try:
            result_data = {}

            # 2.1) well_general
            well_data = self.well_repo.get_well_general()
            if well_data is not None:
                csv_path = os.path.join(input_folder, 'database', 'well_general.csv')
                well_data.to_csv(csv_path, index=False)
                logging_report(
                    f'EXECUTED | 200011 | well_general table saved', 'INFO', self.api_name
                )

            # 2.2) time_based_drill
            tbd_path = os.path.join(input_folder, 'database', 'time_based_drill.csv')
            if not os.path.exists(tbd_path):
                tbd_data = self.well_repo.get_time_based_drill()
                if tbd_data is not None:
                    tbd_data.to_csv(tbd_path, index=False)
                    logging_report(
                        f'EXECUTED | 200012 | time_based_drill table saved',
                        'INFO', self.api_name
                    )

            # 2.3) plan_loader
            plan_loader = PlanLoader(
                input_folder, self.client, self.project,
                self.stream, self.username, self.scenario,
                self.api_name
            )
            plan_loader.load_plan_files()

            # 2.4) rig_design
            rig_data = self.rig_repo.get_rig_design()
            if rig_data is not None:
                # Convertir listas a strings en columnas filter_dict
                for col in ['filter_dict', 'filter_dict_1', 'filter_dict_5',
                            'filter_dict_10', 'filter_dict_15']:
                    # Evita error si la columna no existe
                    if col in rig_data.columns:
                        rig_data[col] = rig_data[col].apply(
                            lambda x: '{' + ', '.join(x) + '}' if isinstance(x, list) else str(x)
                        )

                csv_path = os.path.join(input_folder, 'database', 'rig_design.csv')
                rig_data.to_csv(csv_path, index=False)
                logging_report(
                    f'EXECUTED | 200014 | rig_design table saved', 'INFO', self.api_name
                )

            # 2.5) real_time/time_based_drill_current_well
            real_time_tbd = self.streaming_repo.get_real_time_tbd()
            if real_time_tbd is not None:
                csv_path = os.path.join(input_folder, 'real_time', 'time_based_drill_current_well.csv')
                real_time_tbd.to_csv(csv_path, index=False)
                logging_report(
                    f'EXECUTED | 200015 | time_based_drill_current_well saved',
                    'INFO', self.api_name
                )

            # 2.6) real_time/official_survey_current_well
            real_time_os = self.streaming_repo.get_real_time_os()
            if real_time_os is not None:
                csv_path = os.path.join(input_folder, 'real_time', 'official_survey_current_well.csv')
                real_time_os.to_csv(csv_path, index=False)
                logging_report(
                    f'EXECUTED | 200016 | official_survey_current_well saved',
                    'INFO', self.api_name
                )

            # 2.7) current_measured_depth
            current_measured_depth = self.streaming_repo.get_current_measured_depth()
            if current_measured_depth is not None:
                result_data['current_measured_depth'] = current_measured_depth

            return result_data

        except Exception as e:
            logging_report(
                f'FAILURE | 400000 | Error in get_and_save_data: {str(e)}',
                'ERROR', self.api_name
            )
            raise

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

    # -------------------------------------------------------------------------
    # 4) Método principal: calculate_get_streaming_output
    # -------------------------------------------------------------------------
    def calculate_get_streaming_output(self):
        """
        - Configura ambiente y directorios
        - Obtiene datos (historic y real-time) y los guarda en CSV
        - Procesa la data con DrillDataProcessor
        - Retorna un dict con los dataframes finales (tbd, os)
        """
        try:
            logging_report(
                f'START | 000000 | {self.client} | {self.project} | {self.stream} | '
                f'{self.username} | {self.scenario} | START {self.api_name}.py',
                'INFO', self.api_name
            )

            # 4.1) Normaliza username y scenario
            strings = modify_strings(self.username, self.scenario)
            norm_username, norm_scenario = strings[0], strings[1]

            # 4.2) Crea ambiente de streaming
            streaming_env = StreamingEnvironment(
                client=self.client,
                project=self.project,
                stream=self.stream,
                username=norm_username,
                scenario=norm_scenario,
                environment="development"  # o 'production' si corresponde
            )

            # 4.3) Configura environment y directorios
            streaming_env.setup_environment()
            self.__set_environment_variables()
            input_folder = self.__setup_directories(norm_username, norm_scenario)

            # 4.4) Obtiene datos y los escribe a CSV
            data = self.__get_and_save_data(input_folder)

            # 4.5) Ajustes si hay measured_depth
            if 'current_measured_depth' in data:
                current_measured_depth = data['current_measured_depth']
                streaming_input = StreamingInput(
                    self.client,
                    self.project,
                    self.stream,
                    self.username,
                    self.scenario,
                    self.api_name
                )
                current_hole_diameter, _ = streaming_input.get_current_bit_size(
                    input_folder, current_measured_depth
                )
                streaming_input.create_inputs_json(input_folder, current_hole_diameter)

            # 4.6) Procesa la data con DrillDataProcessor
            process_drilling_data = DrillDataProcessor(input_folder)
            process_path = process_drilling_data.process()

            # 4.7) Carga CSVs resultantes
            rt_tbd_path = os.path.join(process_path, 'real_time_update', 'time_based_drill_current_well_out.csv')
            rt_os_path = os.path.join(process_path, 'real_time_update', 'official_survey_current_well_out.csv')

            df_rt_tbd = pd.read_csv(rt_tbd_path).iloc[::-1]  # invertimos orden
            df_rt_os = pd.read_csv(rt_os_path).iloc[::-1]

            # 4.8) Aplica agregación / limitación de filas
            df_rt_tbd = self.__apply_row_limit_or_aggregation(df_rt_tbd)
            df_rt_os = self.__apply_row_limit_or_aggregation(df_rt_os)

            # 4.9) Rellena NaN
            df_rt_tbd = df_rt_tbd.fillna(-999.25)
            df_rt_os = df_rt_os.fillna(-999.25)

            dict_rt_tbd = df_rt_tbd.to_dict(orient='records')
            dict_rt_os = df_rt_os.to_dict(orient='records')

            # 4.10) Retorna un dict con la data
            return {
                "tbd": dict_rt_tbd,
                "os": dict_rt_os
            }

        except Exception as e:
            # Loggea y relanza para que el controller maneje la excepción
            print(f"Error en calculate_get_streaming_output: {str(e)}")
            print("Traceback completo:")
            print(traceback.format_exc())
            raise
