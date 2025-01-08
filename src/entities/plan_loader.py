import os
from utils_backend import download_file, xlsx_to_csvs, logging_report

class PlanLoader:
    def __init__(self, input_folder, client, project, stream, username, scenario, api_name):
        self.input_folder = input_folder
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.api_name = api_name

    def load_plan_files(self):
        """Descarga y convierte archivos de plan a CSV"""
        try:
            # Crear carpetas necesarias
            scenario_folder = f'{self.client}/{self.project}/{self.stream}/{self.username}/{self.scenario}'
            plan_path = os.path.join(self.input_folder, "plan")
            os.makedirs(plan_path, exist_ok=True)

            # Descargar archivo de GCP
            bucket = 'drillbi'
            object_name = f'{scenario_folder}/plan.xlsx'
            file_name = os.path.join(plan_path, "plan.xlsx")

            resp = download_file(bucket, object_name, file_name, self.api_name)
            if not resp:
                output_message = f'Blob file "{object_name}" does not exist.'
                logging_report(f'FAILURE | 400013 | {self.client} | {self.project} | {self.stream} | {self.username} | {self.scenario} | {output_message}', 'ERROR', self.api_name)
                return False

            # Convertir archivo XLSX a CSV
            xlsx_to_csvs(plan_path, 'plan.xlsx', self.api_name)
            logging_report(f'EXECUTED | 200013 | Plan files loaded successfully', 'INFO', self.api_name)
            return True

        except Exception as e:
            logging_report(f'FAILURE | 400013 | Error loading plan files: {str(e)}', 'ERROR', self.api_name)
            raise
