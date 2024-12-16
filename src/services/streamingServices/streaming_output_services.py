from src.entities.streaming_environment import StreamingEnvironment
from src.entities.session_manager import SessionManager
from src.entities.directory_structure import DirectoryStructure
from src.repositories.well_repository import WellRepository
from src.repositories.rig_repository import RigRepository
from src.repositories.streaming_repository import StreamingRepository
from utils_backend import modify_strings, logging_report
import traceback
import json

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
        # Obtener datos históricos
        well_data = self.__well_repo.get_well_general()
        if well_data is not None:
            well_data.to_csv(f'{input_folder}database/well_general.csv', index=False)
            logging_report(f'EXECUTED | 200011 | {self.__client} | {self.__project} | {self.__stream} | {self.__username} | {self.__scenario} | well_general table successfully downloaded', 'INFO', self.__api_name)
        
        # Datos time based drill
        tbd_data = self.__well_repo.get_time_based_drill()
        if tbd_data is not None:
            tbd_data.to_csv(f'{input_folder}database/time_based_drill.csv', index=False)
            logging_report(f'EXECUTED | 200012 | {self.__client} | {self.__project} | {self.__stream} | {self.__username} | {self.__scenario} | time_based_drill table successfully downloaded', 'INFO', self.__api_name)
        
        # Datos de rig design
        rig_data = self.__rig_repo.get_rig_design()
        if rig_data is not None:
            rig_data.to_csv(f'{input_folder}database/rig_design.csv', index=False)
            logging_report(f'EXECUTED | 200014 | {self.__client} | {self.__project} | {self.__stream} | {self.__username} | {self.__scenario} | rig_design table successfully downloaded', 'INFO', self.__api_name)
        
        # Datos en tiempo real
        real_time_tbd = self.__streaming_repo.get_real_time_tbd()
        if real_time_tbd is not None:
            real_time_tbd.to_csv(f'{input_folder}real_time/time_based_drill_current_well.csv', index=False)
            logging_report(f'EXECUTED | 200015 | {self.__client} | {self.__project} | {self.__stream} | {self.__username} | {self.__scenario} | time_based_drill_current_well table successfully downloaded', 'INFO', self.__api_name)
        
        real_time_os = self.__streaming_repo.get_real_time_os()
        if real_time_os is not None:
            real_time_os.to_csv(f'{input_folder}real_time/official_survey_current_well.csv', index=False)
            logging_report(f'EXECUTED | 200016 | {self.__client} | {self.__project} | {self.__stream} | {self.__username} | {self.__scenario} | official_survey_current_well table successfully downloaded', 'INFO', self.__api_name)
        
        # Obtener datos adicionales para el JSON
        current_measured_depth = self.__streaming_repo.get_current_measured_depth()
        
        return {
            'well_data': well_data,
            'tbd_data': tbd_data,
            'rig_data': rig_data,
            'real_time_tbd': real_time_tbd,
            'real_time_os': real_time_os,
            'current_measured_depth': current_measured_depth
        }
    
    def __create_input_json(self, input_folder, current_measured_depth):
        """Crea el archivo JSON con los inputs necesarios"""
        input_data = {
            'current_bit_size': current_measured_depth,
            'CURRENT_WELL_NAME': 'RT',
            'WELLS_SELECT_NAME': []  # Aquí podrías agregar la lógica para obtener los nombres de los pozos
        }
        
        with open(f'{input_folder}inputs_for_rt.json', 'w', encoding='utf-8') as f:
            json.dump(input_data, f, indent=6)
            
        logging_report(f'EXECUTED | 200021 | {self.__client} | {self.__project} | {self.__stream} | {self.__username} | {self.__scenario} | inputs_for_rt.json file successfully constructed', 'INFO', self.__api_name)
        
    def calculate_get_streaming_output(self):
        try:
            logging_report(
                f'START | 000000 | {self.__client} | {self.__project} | {self.__stream} | {self.__username} | {self.__scenario} | START get_input_data_streams.py',
                'INFO', self.__api_name)
            
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
            
            # Crear JSON de entrada
            if data['current_measured_depth'] is not None:
                self.__create_input_json(input_folder, data['current_measured_depth'])
            
            logging_report(
                f'END | 999999 | {self.__client} | {self.__project} | {self.__stream} | {self.__username} | {self.__scenario} | END get_input_data_streams.py',
                'INFO', self.__api_name)
            
        except Exception as e:
            print(f"Error en calculate_get_streaming_output: {str(e)}")
            print("Traceback completo:")
            print(traceback.format_exc())
            raise