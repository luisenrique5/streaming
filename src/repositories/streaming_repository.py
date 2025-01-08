# src/repositories/streaming_repository.py
import pandas as pd
from utils_backend import query_execute, get_data_at_interval, logging_report

class StreamingRepository:
    """
    Maneja la obtención de datos de streaming en tiempo real (TBD, OS y profundidad actual).
    """

    def __init__(self, client, project, stream, username, scenario, api_name):
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.api_name = api_name

        # Construimos el nombre de la base de datos y tablas con la convención actual
        self.streaming_db = f'streaming_drillbi_{client}_{project}_db'
        self.streaming_tbd_table = f'streaming_tbd_{stream}'
        self.streaming_os_table = f'streaming_os_{stream}'

    def get_real_time_tbd(self):
        """
        Obtiene los datos de Time-Based Drilling (TBD) en 'interval_seconds' por defecto 10.
        Retorna un DataFrame con columnas:
          ['datetime','measured_depth','bit_depth','block_height','rop','wob','hook_load',
           'flow_rate','spp','torque','surface_rpm','motor_rpm','bit_rpm']
        o None si ocurre un error.
        """
        try:
            data = get_data_at_interval(
                self.client, self.project, self.stream,
                self.username, self.scenario,
                self.streaming_db,
                self.streaming_tbd_table,
                self.api_name,
                interval_seconds=10
            )

            df = pd.DataFrame(data, columns=[
                'id', 'datetime', 'measured_depth', 'bit_depth', 'block_height',
                'rop', 'wob', 'hook_load', 'flow_rate', 'spp', 'torque',
                'surface_rpm', 'motor_rpm', 'bit_rpm'
            ])

            # Eliminamos la columna id
            df.drop(columns='id', inplace=True)
            return df

        except Exception as e:
            logging_report(
                f'FAILURE | 400015 | {self.client} | {self.project} | {self.stream} | '
                f'{self.username} | {self.scenario} | Error getting TBD data: {str(e)}',
                'ERROR',
                self.api_name
            )
            return None

    def get_real_time_os(self):
        """
        Obtiene los datos de Official Survey (OS) ordenados por measured_depth asc.
        Retorna un DataFrame con columnas ['measured_depth','incl','azm'], o None si falla.
        """
        query = f'SELECT * FROM {self.streaming_os_table} ORDER BY measured_depth ASC;'
        data, error = query_execute(query, self.streaming_db, True, self.api_name)

        if error:
            logging_report(
                f'FAILURE | 400016 | {self.client} | {self.project} | {self.stream} | '
                f'{self.username} | {self.scenario} | {data}',
                'ERROR',
                self.api_name
            )
            return None

        df = pd.DataFrame(data, columns=['id', 'measured_depth', 'incl', 'azm'])
        df.drop(columns='id', inplace=True)
        return df

    def get_current_measured_depth(self):
        """
        Obtiene la última profundidad medida (measured_depth) de TBD, es decir, la fila con el
        mayor id. Retorna un float o None si falla.
        """
        query = (
            f"SELECT measured_depth "
            f"FROM {self.streaming_tbd_table} "
            f"ORDER BY id DESC LIMIT 1"
        )
        data, error = query_execute(query, self.streaming_db, False, self.api_name)
        if error:
            logging_report(
                f'FAILURE | 400017 | {self.client} | {self.project} | {self.stream} | '
                f'{self.username} | {self.scenario} | {data}',
                'ERROR',
                self.api_name
            )
            return None

        # Se asume que data[0] contiene el valor numérico de measured_depth
        return float(data[0])
