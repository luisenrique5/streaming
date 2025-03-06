import pandas as pd
from typing import Any
from utils_backend import query_execute, get_data_at_interval

class WellRepository:
    def __init__(self, client: str, project: str, stream: str, username: str, scenario: str, api_name: str) -> None:
        """
        Repositorio para acceder a los datos generales de pozos y la tabla de time_based_drill.
        """
        self.__client = client
        self.__project = project
        self.__stream = stream
        self._username = username
        self._scenario = scenario
        self.__api_name = api_name

    def get_well_general(self) -> pd.DataFrame:
        """
        Consulta la tabla well_general_{client}_{project} en la base de datos 'well_general_db'
        y retorna un DataFrame con la información de los pozos.
        """
        try:
            query = f"SELECT * FROM well_general_{self.__client}_{self.__project};"
            data, error = query_execute(query, "well_general_db", True, self.__api_name)
            if error:
                raise ValueError(f"Error en get_well_general: {data}")

            df = pd.DataFrame(data)
            # Se asignan los nombres de las columnas de forma explícita
            df.columns = [
                "id", "well_id", "well_name", "well_type", "client", "rig", "field",
                "spud_date", "bi_datetime_from", "bi_datetime_to", "total_time",
                "total_depth", "location", "latitude", "longitude"
            ]
            df = df.drop(columns=["id"])
            return df

        except Exception as e:
            raise ValueError(f"Error en get_well_general: {str(e)}") from e

    def get_time_based_drill(self) -> pd.DataFrame:
        """
        Consulta la tabla time_based_drill_table_{client}_{project} en la base de datos 'time_based_drill_db'
        y retorna un DataFrame con la información del drilling basado en el tiempo.
        """
        try:
            query = f"SELECT * FROM time_based_drill_table_{self.__client}_{self.__project};"
            data, error = query_execute(query, "time_based_drill_db", True, self.__api_name)
            if error:
                raise ValueError(f"Error en get_time_based_drill: {data}")

            df = pd.DataFrame(data)
            df.columns = [
                "id", "well_id", "datetime", "cumulative_time", "day_number",
                "measured_depth", "tvd", "incl", "azm", "dls", "well_section",
                "bit_depth", "hole_diameter", "formation", "block_height", "rop",
                "wob", "hook_load", "flow_rate", "pit_volume", "diff_pressure", "spp",
                "annular_pressure", "torque", "surface_rpm", "motor_rpm", "bit_rpm",
                "mse", "mud_motor", "casing", "rig_super_state", "rig_sub_activity",
                "consecutive_labels"
            ]
            df = df.drop(columns=["id"])
            return df
        
        except Exception as e:
            raise ValueError(f"Error en get_time_based_drill: {str(e)}") from e
