import pandas as pd
from typing import Any, List, Tuple
from utils_backend import query_execute, get_data_at_interval, logging_report

class StreamingRepository:
    """
    Repositorio para obtener datos de streaming, incluyendo:
      - Datos en tiempo real de "time_based_drill" (TBD)
      - Datos en tiempo real de "official_survey" (OS)
      - La última profundidad medida
    """

    def __init__(self, client: str, project: str, stream: str, username: str, scenario: str, api_name: str) -> None:
        """
        Inicializa el repositorio con parámetros de conexión y nombres de tabla.

        Args:
            client: Nombre del cliente.
            project: Nombre del proyecto.
            stream: Nombre del stream.
            username: Nombre de usuario.
            scenario: Escenario.
            api_name: Nombre de la API para logs.
        """
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__api_name = api_name

        # Nombre de la base de datos y de las tablas de streaming
        self.__streaming_db = f'streaming_drillbi_{client}_{project}_db'
        self.__streaming_tbd_table = f'streaming_tbd_{stream}'
        self.__streaming_os_table = f'streaming_os_{stream}'

    def get_real_time_tbd(self) -> pd.DataFrame:
        """
        Obtiene los datos en tiempo real de la tabla TBD utilizando get_data_at_interval.

        Returns:
            DataFrame con las columnas:
                ['datetime', 'measured_depth', 'bit_depth', 'block_height', 'rop',
                 'wob', 'hook_load', 'flow_rate', 'spp', 'torque', 'surface_rpm',
                 'motor_rpm', 'bit_rpm']

        Raises:
            ValueError: Si ocurre un error durante la consulta.
        """
        try:
            data = get_data_at_interval(
                self.__client,
                self.__project,
                self.__stream,
                self.__username,
                self.__scenario,
                self.__streaming_db,
                self.__streaming_tbd_table,
                self.__api_name,
                interval_seconds=10
            )

            # Definición de columnas esperadas (incluyendo 'id')
            columns: List[str] = [
                "id", "datetime", "measured_depth", "bit_depth", "block_height",
                "rop", "wob", "hook_load", "flow_rate", "spp", "torque",
                "surface_rpm", "motor_rpm", "bit_rpm"
            ]
            df = pd.DataFrame(data, columns=columns)
            df = df.drop(columns=["id"])
            return df

        except Exception as e:
            raise ValueError(f"Error en get_real_time_tbd: {str(e)}") from e

    def get_real_time_os(self) -> pd.DataFrame:
        """
        Obtiene los datos en tiempo real de la tabla OS, ordenados por 'measured_depth' ascendente.

        Returns:
            DataFrame con las columnas:
                ['measured_depth', 'incl', 'azm']

        Raises:
            ValueError: Si ocurre un error durante la consulta.
        """
        try:
            query = f"SELECT * FROM {self.__streaming_os_table} ORDER BY measured_depth ASC;"
            data, error = query_execute(query, self.__streaming_db, True, self.__api_name)
            if error:
                raise ValueError(f"Error en get_real_time_os: {data}")

            df = pd.DataFrame(data, columns=['id', 'measured_depth', 'incl', 'azm'])
            df = df.drop(columns=["id"])
            return df

        except Exception as e:
            raise ValueError(f"Error en get_real_time_os: {str(e)}") from e

    def get_current_measured_depth(self) -> float:
        """
        Obtiene la última profundidad medida (medida_depth) de la tabla TBD, ordenada de forma descendente.

        Returns:
            La profundidad medida actual como float.

        Raises:
            ValueError: Si ocurre un error durante la consulta.
        """
        try:
            query = (
                f"SELECT measured_depth "
                f"FROM {self.__streaming_tbd_table} "
                f"ORDER BY id DESC LIMIT 1"
            )
            data, error = query_execute(query, self.__streaming_db, False, self.__api_name)
            if error:
                raise ValueError(f"Error en get_current_measured_depth: {data}")

            return float(data[0])
        except Exception as e:
            raise ValueError(f"Error en get_current_measured_depth: {str(e)}") from e
