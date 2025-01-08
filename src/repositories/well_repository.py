import pandas as pd
from utils_backend import query_execute, get_data_at_interval, logging_report

class WellRepository:
    """
    Maneja la obtención de datos generales de los pozos (well_general)
    y la tabla de time_based_drill (time_based_drill_table_<client>_<project>).
    """

    def __init__(self, client, project, stream, username, scenario, api_name):
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.api_name = api_name
        self.streaming_db = f"streaming_drillbi_{client}_{project}_db"

    def get_well_general(self):
        """
        Consulta la tabla well_general_<client>_<project> en la BD 'well_general_db'.
        Retorna un DataFrame con las columnas:
          ['well_id','well_name','well_type','client','rig','field','spud_date',
           'bi_datetime_from','bi_datetime_to','total_time','total_depth','location',
           'latitude','longitude']
        o None si ocurre un error o la consulta falla.
        """
        # 1) Construimos la consulta
        query = f"SELECT * FROM well_general_{self.client}_{self.project};"

        # 2) Ejecutamos la consulta
        data, error = query_execute(query, "well_general_db", True, self.api_name)
        if error:
            logging_report(
                f"FAILURE | 400011 | {self.client} | {self.project} | {self.stream} | "
                f"{self.username} | {self.scenario} | {data}",
                "ERROR",
                self.api_name
            )
            return None

        # 3) Creamos el DataFrame
        df = pd.DataFrame(data)
        df.columns = [
            "id", "well_id", "well_name", "well_type", "client", "rig", "field",
            "spud_date", "bi_datetime_from", "bi_datetime_to", "total_time",
            "total_depth", "location", "latitude", "longitude"
        ]
        # Eliminamos la columna id
        del df["id"]
        return df

    def get_time_based_drill(self):
        """
        Consulta la tabla time_based_drill_table_<client>_<project> en la BD 'time_based_drill_db'.
        Retorna un DataFrame con las columnas:
          ['well_id','datetime','cumulative_time','day_number','measured_depth','tvd','incl','azm',
           'dls','well_section','bit_depth','hole_diameter','formation','block_height','rop',
           'wob','hook_load','flow_rate','pit_volume','diff_pressure','spp','annular_pressure',
           'torque','surface_rpm','motor_rpm','bit_rpm','mse','mud_motor','casing','rig_super_state',
           'rig_sub_activity','consecutive_labels']
        o None si ocurre un error o la consulta falla.
        """
        # 1) Construimos la consulta
        query = f"SELECT * FROM time_based_drill_table_{self.client}_{self.project};"

        # 2) Ejecutamos la consulta
        data, error = query_execute(query, "time_based_drill_db", True, self.api_name)
        if error:
            logging_report(
                f"FAILURE | 400012 | {self.client} | {self.project} | {self.stream} | "
                f"{self.username} | {self.scenario} | {data}",
                "ERROR",
                self.api_name
            )
            return None

        # 3) Creamos el DataFrame
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
        # Eliminamos la columna id
        del df["id"]
        return df
