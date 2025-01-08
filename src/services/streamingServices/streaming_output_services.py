import os
import json
import traceback

# Asegúrate de que pandas está importado
import pandas as pd

# Si tu servicio retorna un objeto Flask Response
from flask import Response

# Ajusta estos imports según tu estructura real
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
from utils_backend import *


class GetStreamingOutput:
    def __init__(self, client, project, stream, username, scenario, number_of_rows):
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__number_of_rows = number_of_rows
        self.__api_name = 'get_input_data_streams'
        
        # Inicializar repositorios
        self.__well_repo = WellRepository(client, project, stream, username, scenario, self.__api_name)
        self.__rig_repo = RigRepository(client, project, stream, username, scenario, self.__api_name)
        self.__streaming_repo = StreamingRepository(client, project, stream, username, scenario, self.__api_name)
        
    def __set_environment_variables(self):
        """Configura las variables de entorno"""
        session_manager = SessionManager()
        
        variables = {
            'CLIENT': self.__client,
            'PROJECT': self.__project,
            'STREAM': self.__stream,
            'USERNAME': self.__username,
            'SCENARIO': self.__scenario
        }
        
        for key, value in variables.items():
            session_manager.set_variable(key, value)
            logging_report(f'Set {key}: {value}', 'DEBUG', self.__api_name)
        
        return session_manager
    
    def __setup_directories(self, username, scenario):
        """Configura la estructura de directorios"""
        dir_structure = DirectoryStructure(
            self.__client, self.__project, self.__stream, 
            username, scenario
        )
        return dir_structure.create_directory_structure()
    
    def __get_and_save_data(self, input_folder):
        """Obtiene y guarda todos los datos necesarios"""
        try:
            result_data = {}
            
            # Historic tables
            well_data = self.__well_repo.get_well_general()
            if well_data is not None:
                well_data.to_csv(f'{input_folder}database/well_general.csv', index=False)
                logging_report(f'EXECUTED | 200011 | well_general table saved', 'INFO', self.__api_name)
            
            # Time based drill
            tbd_path = f'{input_folder}database/time_based_drill.csv'
            if not os.path.exists(tbd_path):
                tbd_data = self.__well_repo.get_time_based_drill()
                if tbd_data is not None:
                    tbd_data.to_csv(tbd_path, index=False)
                    logging_report(f'EXECUTED | 200012 | time_based_drill table saved', 'INFO', self.__api_name)
                    
            plan_loader = PlanLoader(
                input_folder, self.__client, self.__project,
                self.__stream, self.__username, self.__scenario, self.__api_name
            )    
            plan_loader.load_plan_files()
            
            # Rig design
            rig_data = self.__rig_repo.get_rig_design()
            if rig_data is not None:
                # Procesar filter_dict fields
                for col in ['filter_dict', 'filter_dict_1', 'filter_dict_5', 'filter_dict_10', 'filter_dict_15']:
                    rig_data[col] = rig_data[col].apply(lambda x: '{' + ', '.join(x) + '}')
                rig_data.to_csv(f'{input_folder}database/rig_design.csv', index=False)
                logging_report(f'EXECUTED | 200014 | rig_design table saved', 'INFO', self.__api_name)

            # Real time data - TBD
            real_time_tbd = self.__streaming_repo.get_real_time_tbd()
            if real_time_tbd is not None:
                real_time_tbd.to_csv(f'{input_folder}real_time/time_based_drill_current_well.csv', index=False)
                logging_report(f'EXECUTED | 200015 | time_based_drill_current_well saved', 'INFO', self.__api_name)

            # Real time data - OS
            real_time_os = self.__streaming_repo.get_real_time_os()
            if real_time_os is not None:
                real_time_os.to_csv(f'{input_folder}real_time/official_survey_current_well.csv', index=False)
                logging_report(f'EXECUTED | 200016 | official_survey_current_well saved', 'INFO', self.__api_name)

            # Get current measured depth
            current_measured_depth = self.__streaming_repo.get_current_measured_depth()
            if current_measured_depth is not None:
                result_data['current_measured_depth'] = current_measured_depth

            return result_data
                
        except Exception as e:
            logging_report(f'FAILURE | 400000 | Error in get_and_save_data: {str(e)}', 'ERROR', self.__api_name)
            raise
         
    def calculate_get_streaming_output(self):
        """
        Método principal que:
        - Configura ambiente y estructura de directorios.
        - Obtiene datos (historic y real-time).
        - Procesa la data con DrillDataProcessor.
        - Devuelve un JSON (por medio de Flask `Response`) con data en orient='records'.
        """
        try:
            current_dir = os.path.dirname(__file__)
            # Ruta base para tus utilidades, ajústala según tu estructura real
            base_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'utils'))
            
            logging_report(
                f'START | 000000 | {self.__client} | {self.__project} | {self.__stream} | '
                f'{self.__username} | {self.__scenario} | START get_input_data_streams.py',
                'INFO', self.__api_name
            )
            
            # Modificar strings
            strings = modify_strings(self.__username, self.__scenario)
            username = strings[0]
            scenario = strings[1]
            
            # Crear y configurar el ambiente
            streaming_env = StreamingEnvironment(
                client=self.__client,
                project=self.__project,
                stream=self.__stream,
                username=username,
                scenario=scenario,
                environment="development"
            )
            
            # Configurar ambiente y directorios
            log_file = streaming_env.setup_environment()
            session_manager = self.__set_environment_variables()
            input_folder = self.__setup_directories(username, scenario)
            
            # Obtener y guardar datos
            data = self.__get_and_save_data(input_folder)
            
            # Si obtuvimos measured_depth, procedemos
            if 'current_measured_depth' in data:
                current_measured_depth = data['current_measured_depth']
                streaming_input = StreamingInput(
                    self.__client,
                    self.__project,
                    self.__stream,
                    self.__username,
                    self.__scenario,
                    self.__api_name
                )
                current_hole_diameter, _ = streaming_input.get_current_bit_size(
                    input_folder, current_measured_depth
                )
                streaming_input.create_inputs_json(input_folder, current_hole_diameter)

            # Procesar la data con DrillDataProcessor
            process_drilling_data = DrillDataProcessor(input_folder)
            # Este .process() suponemos que retorna la ruta donde se guardan los CSVs procesados
            process = process_drilling_data.process()
            
            # Cargar los CSVs resultantes
            df_rt_tbd = pd.read_csv(f'{process}/real_time_update/time_based_drill_current_well_out.csv')
            df_rt_os = pd.read_csv(f'{process}/real_time_update/official_survey_current_well_out.csv')

            # Orden inverso
            df_rt_tbd = df_rt_tbd.iloc[::-1]
            df_rt_os = df_rt_os.iloc[::-1]

            # Manejo de la granularidad (self.__number_of_rows)
            if self.__number_of_rows == 'aggregated':
                # Lógica de muestreo cada cierto número de filas
                if len(df_rt_tbd) > 100000:
                    df_rt_tbd = df_rt_tbd.iloc[::80, :]
                elif len(df_rt_tbd) > 50000:
                    df_rt_tbd = df_rt_tbd.iloc[::40, :]
                elif len(df_rt_tbd) > 20000:
                    df_rt_tbd = df_rt_tbd.iloc[::20, :]
                elif len(df_rt_tbd) > 10000:
                    df_rt_tbd = df_rt_tbd.iloc[::10, :]
                elif len(df_rt_tbd) > 5000:
                    df_rt_tbd = df_rt_tbd.iloc[::8, :]
                elif len(df_rt_tbd) > 2000:
                    df_rt_tbd = df_rt_tbd.iloc[::4, :]
                elif len(df_rt_tbd) > 1000:
                    df_rt_tbd = df_rt_tbd.iloc[::2, :]
            else:
                df_rt_tbd = df_rt_tbd.iloc[:int(self.__number_of_rows)]

            # Rellenar NaN con un valor distinto
            df_rt_tbd = df_rt_tbd.fillna(-999.25)
            dict_rt_tbd = df_rt_tbd.to_dict(orient='records')

            # Repetir la misma lógica para official survey
            if self.__number_of_rows == 'aggregated':
                if len(df_rt_os) > 100000:
                    df_rt_os = df_rt_os.iloc[::80, :]
                elif len(df_rt_os) > 50000:
                    df_rt_os = df_rt_os.iloc[::40, :]
                elif len(df_rt_os) > 20000:
                    df_rt_os = df_rt_os.iloc[::20, :]
                elif len(df_rt_os) > 10000:
                    df_rt_os = df_rt_os.iloc[::10, :]
                elif len(df_rt_os) > 5000:
                    df_rt_os = df_rt_os.iloc[::8, :]
                elif len(df_rt_os) > 2000:
                    df_rt_os = df_rt_os.iloc[::4, :]
                elif len(df_rt_os) > 1000:
                    df_rt_os = df_rt_os.iloc[::2, :]
            else:
                df_rt_os = df_rt_os.iloc[:int(self.__number_of_rows)]

            df_rt_os = df_rt_os.fillna(-999.25)
            dict_rt_os = df_rt_os.to_dict(orient='records')

            data = {"os": dict_rt_os,
                    "tbd": dict_rt_tbd}
            return data

        except Exception as e:
            print(f"Error en calculate_get_streaming_output: {str(e)}")
            print("Traceback completo:")
            print(traceback.format_exc())
            # Relanzas la excepción para que tu manejador superior la pueda registrar
            raise
