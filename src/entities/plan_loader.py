import os
import io
import pandas as pd
import concurrent.futures
from utils_backend import download_file, logging_report
from pathlib import Path

class PlanLoader:
    def __init__(
        self,
        client: str,
        project: str,
        stream: str,
        username: str,
        scenario: str,
        base_key: str,
        redis_connection,
        api_name: str
    ):
        """
        Constructor de PlanLoader.

        :param client: Nombre del cliente
        :param project: Nombre del proyecto
        :param stream: Nombre del stream
        :param username: Nombre de usuario
        :param scenario: Escenario
        :param base_key: Clave base en Redis, ejemplo: "input_data:client:project:stream:username:scenario"
        :param redis_connection: Conexión a Redis
        :param api_name: Nombre de la API (para logging)
        """
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__base_key = base_key
        self.__redis = redis_connection
        self.__api_name = api_name  # ¡Ojo! Necesitamos esto para download_file(..., self.__api_name)

    def load_plan_files(self):
        """
        1. Descarga plan.xlsx a un archivo local temporal (porque NO podemos cambiar download_file).
        2. Carga todas las hojas en memoria con pandas.
        3. Guarda cada hoja en Redis bajo la "carpeta plan".
        4. Elimina el archivo local.
        5. Usa ThreadPoolExecutor para procesar las hojas en paralelo (opcional).
        """
        temp_file = None
        try:
            # Construir la ruta del blob en GCP
            bucket = "drillbi"
            scenario_folder = (
                f"{self.__client}/"
                f"{self.__project}/"
                f"{self.__stream}/"
                f"{self.__username}/"
                f"{self.__scenario}"
            )
            object_name = f"{scenario_folder}/plan.xlsx"

            # Ruta local temporal (p.ej. /tmp)
            temp_dir = "/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, "plan_tmp.xlsx")

            # Descargar con la función que NO se puede modificar
            downloaded = download_file(bucket, object_name, temp_file, self.__api_name)
            if not downloaded:
                raise ValueError(
                    f"Blob file '{object_name}' does not exist in bucket '{bucket}'."
                )

            # Leer el archivo en memoria
            with open(temp_file, "rb") as f:
                file_bytes = f.read()

            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet_names = excel_file.sheet_names  # lista de hojas

            # Clave "carpeta" plan en Redis
            plan_key = f"{self.__base_key}:plan"

            # Función interna para procesar cada hoja
            def process_sheet(sheet_name):
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                redis_key = f"{plan_key}:{sheet_name}"
                self.__redis.set(redis_key, df.to_json(orient="records"))

            # Procesar en paralelo (opcional)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(process_sheet, sheet_names)

            return True

        except Exception as e:
            raise ValueError(f"Error loading plan files: {str(e)}")

        finally:
            # Borrar el archivo temporal
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as rm_err:
                    raise ValueError(f"Error removing temp file: {str(rm_err)}")
