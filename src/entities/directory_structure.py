import os
from pathlib import Path

class DirectoryStructure:
    def __init__(self, client, project, stream, username, scenario):
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.input_folder = self._create_input_folder()
        
    def _create_input_folder(self):
        base_path = f"./input_data/{self.client}/{self.project}/{self.stream}/{self.username}/{self.scenario}/"
        return base_path
        
    def create_directory_structure(self):
        """Crea toda la estructura de directorios necesaria"""
        # Directorios principales
        directories = [
            self.input_folder,
            f'{self.input_folder}plan',
            f'{self.input_folder}database',
            f'{self.input_folder}real_time',
            self.input_folder + 'csv/',
            self.input_folder + 'real_time_update/',
            self.input_folder + 'real_time_update/csv/'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        return self.input_folder