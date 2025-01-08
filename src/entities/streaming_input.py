import os
import json
import pandas as pd
import concurrent.futures
from threading import Lock
from utils_backend import query_execute, logging_report

###############################################################################
# POOL DE HILOS Y LOCKS GLOBALES
###############################################################################
# Pool de hilos para lecturas concurrentes de CSV. Ajusta max_workers según tu servidor.
csv_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# Caché global de DataFrames para no leer el mismo archivo repetidamente.
_plan_csv_cache = {}
_plan_csv_lock = Lock()

# Lock para escritura de JSON y evitar colisiones al escribirlo.
_json_lock = Lock()


def _read_csv_in_thread(path: str) -> pd.DataFrame:
    """
    Función interna para leer un CSV usando pandas dentro de un hilo del pool.
    Se llama de forma concurrente en get_current_bit_size.
    """
    return pd.read_csv(path)


class StreamingInput:
    def __init__(self, client, project, stream, username, scenario, api_name):
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.api_name = api_name

    def get_current_bit_size(self, input_folder, current_measured_depth):
        """
        Obtiene el bit size actual y lista de diámetros previos.
        Lectura de 'time_depth_plan.csv' con concurrencia + caché.
        La lógica final se mantiene idéntica a la original:
          - Se filtra por measured_depth <= current_measured_depth
          - Se toma la última fila en 'hole_diameter'
          - Se genera la lista de diámetros con '.' reemplazado por 'p'
        """
        try:
            # Construimos la ruta absoluta
            plan_path = os.path.join(input_folder, 'plan', 'time_depth_plan.csv')
            abs_path = os.path.abspath(plan_path)

            # Bloqueamos para consultar/escribir en el caché
            with _plan_csv_lock:
                # Si el CSV ya fue leído antes, lo tomamos del caché
                if abs_path in _plan_csv_cache:
                    df = _plan_csv_cache[abs_path]
                else:
                    # Enviamos la lectura al pool de hilos y esperamos resultado
                    future = csv_thread_pool.submit(_read_csv_in_thread, abs_path)
                    df = future.result()
                    _plan_csv_cache[abs_path] = df

            # Aseguramos que la columna 'measured_depth' sea numérica
            df['measured_depth'] = pd.to_numeric(df['measured_depth'], errors='coerce')

            # Tomamos la fila final con measured_depth <= current_measured_depth
            filtered_df = df[df['measured_depth'] <= current_measured_depth]
            current_hole_diameter = filtered_df['hole_diameter'].iloc[-1]

            # Construimos la lista de diámetros con '.' reemplazado por 'p'
            distinct_hole_diameters = df['hole_diameter'].unique()
            currt_n_prev_hole_diameters = [
                str(d).replace(".", "p") for d in distinct_hole_diameters
            ]

            logging_report(
                f'EXECUTED | 200017 | current_hole_diameter retrieved', 
                'INFO', 
                self.api_name
            )

            return current_hole_diameter, currt_n_prev_hole_diameters

        except Exception as e:
            logging_report(
                f'FAILURE | 400017 | Error getting bit size: {str(e)}', 
                'ERROR', 
                self.api_name
            )
            raise

    def get_wells_select_name(self):
        """
        Obtiene la lista de wells_select_name desde la base de datos,
        ignorando los que tengan 'RT'.
        La lógica sigue igual: si hay error en la query, se loguea pero 
        no se lanza excepción (retornará lista vacía).
        """
        try:
            wells_select_name = []
            database_name = 'wcr_summary_well_bs_all_db'
            query = (
                f"select well_id_name "
                f"from t_{self.client}_{self.project}_{self.stream}_{self.username}_{self.scenario};"
            )

            data, error = query_execute(query, database_name, True, self.api_name)
            if error:
                # Se conserva la misma lógica que el original: se loguea y se continúa.
                # Si deseas romper la ejecución, tendrías que "raise RuntimeError(...)"
                logging_report(
                    f'FAILURE | 400020 | Error en query: {error}', 
                    'ERROR', 
                    self.api_name
                )
            else:
                for row in data:
                    # Estructura original: separar por '-', ignorar 'RT'
                    well_in_row = str(row[0]).split('-', 1)[1].lstrip()
                    if well_in_row != 'RT':
                        wells_select_name.append(well_in_row)

                logging_report(
                    f'EXECUTED | 200020 | wells_select_name retrieved', 
                    'INFO', 
                    self.api_name
                )

            return wells_select_name

        except Exception as e:
            logging_report(
                f'FAILURE | 400020 | Error getting wells: {str(e)}', 
                'ERROR', 
                self.api_name
            )
            raise

    def create_inputs_json(self, input_folder, current_hole_diameter):
        """
        Crea el JSON con los inputs necesarios:
          - current_bit_size
          - CURRENT_WELL_NAME = 'RT'
          - WELLS_SELECT_NAME (sin 'RT')
        Uso de lock para evitar que múltiples hilos sobrescriban el mismo archivo a la vez.
        """
        try:
            wells_select_name = self.get_wells_select_name()

            inputs_for_rt = {
                'current_bit_size': current_hole_diameter,
                'CURRENT_WELL_NAME': 'RT',
                'WELLS_SELECT_NAME': wells_select_name
            }

            json_path = os.path.join(input_folder, 'inputs_for_rt.json')
            # Lock para evitar colisiones de escritura simultánea
            with _json_lock:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(inputs_for_rt, f, indent=6)

            logging_report(
                f'EXECUTED | 200021 | JSON file created', 
                'INFO', 
                self.api_name
            )

        except Exception as e:
            logging_report(
                f'FAILURE | 400021 | Error creating JSON: {str(e)}', 
                'ERROR', 
                self.api_name
            )
            raise
