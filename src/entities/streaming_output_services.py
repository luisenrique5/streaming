from src.entities.directory_structure import DirectoryStructure
from src.entities.database_manager import DatabaseManager
from utils_backend import modify_strings

class GetStreamingOutput:
    def __init__(self, client, project, stream, username, scenario, number_of_rows):
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__number_of_rows = number_of_rows
        self.__api_name = 'get_input_data_streams'
        
    def calculate_get_streaming_output(self):
        try:
            # Modificar strings
            strings = modify_strings(self.__username, self.__scenario)
            username = strings[0]
            scenario = strings[1]
            
            # Crear estructura de directorios
            dir_structure = DirectoryStructure(
                self.__client, self.__project, self.__stream, 
                username, scenario
            )
            input_folder = dir_structure.create_directory_structure()
            
            # Obtener datos de base de datos
            db_manager = DatabaseManager(
                self.__client, self.__project, self.__stream,
                username, scenario, self.__api_name
            )
            
            # Aquí continuaríamos con el resto de la lógica...
            
        except Exception as e:
            print(f"Error en calculate_get_streaming_output: {str(e)}")
            raise