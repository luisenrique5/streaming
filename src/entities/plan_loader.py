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

    def load_plan_files(self, skip_if_local_exists=False):
        """
        Descarga y convierte archivos de plan a CSV.
        
        Parámetros:
        -----------
        skip_if_local_exists : bool
            - False (por defecto): siempre descarga desde GCP.
            - True: si ya existe el plan.xlsx local, omite la descarga para ahorrar tiempo.
            
        Retorna:
        --------
        bool:
            True si se descargó (o se omitió) y convirtió exitosamente el plan.
            False si el blob no existía en GCP.
        
        Lanza excepción en caso de errores inesperados.
        """
        try:
            # 1) Crear carpeta "plan" si no existe
            scenario_folder = f'{self.client}/{self.project}/{self.stream}/{self.username}/{self.scenario}'
            plan_path = os.path.join(self.input_folder, "plan")
            os.makedirs(plan_path, exist_ok=True)

            # 2) Definir bucket, objeto y ruta local
            bucket = 'drillbi'
            object_name = f'{scenario_folder}/plan.xlsx'
            file_name = os.path.join(plan_path, "plan.xlsx")

            # 3) Verificar si se omite la descarga (opcional)
            if skip_if_local_exists and os.path.isfile(file_name):
                logging_report(
                    f'SKIP_DOWNLOAD | {self.api_name} | El archivo local "{file_name}" ya existe. Se omite la descarga.',
                    'INFO',
                    self.api_name
                )
            else:
                # 4) Descargar desde GCP
                resp = download_file(bucket, object_name, file_name, self.api_name)
                if not resp:
                    output_message = f'Blob file "{object_name}" does not exist.'
                    logging_report(
                        f'FAILURE | 400013 | {self.client} | {self.project} | {self.stream} | '
                        f'{self.username} | {self.scenario} | {output_message}',
                        'ERROR',
                        self.api_name
                    )
                    return False

            # 5) Convertir plan.xlsx a CSV
            xlsx_to_csvs(plan_path, 'plan.xlsx', self.api_name)

            # 6) Log final de éxito
            logging_report(
                f'EXECUTED | 200013 | Plan files loaded successfully', 
                'INFO', 
                self.api_name
            )
            return True

        except Exception as e:
            logging_report(
                f'FAILURE | 400013 | Error loading plan files: {str(e)}', 
                'ERROR', 
                self.api_name
            )
            raise
