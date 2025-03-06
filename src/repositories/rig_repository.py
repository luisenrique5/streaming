import pandas as pd
from typing import Any
from utils_backend import query_execute, logging_report

class RigRepository:
    def __init__(self, client: str, project: str, stream: str, username: str, scenario: str, api_name: str) -> None:
        """
        Repositorio para acceder a los datos de diseño de la plataforma (rig design).
        
        Args:
            client: Nombre del cliente.
            project: Nombre del proyecto.
            stream: Nombre del stream.
            username: Nombre de usuario.
            scenario: Escenario.
            api_name: Nombre de la API para los logs.
        """
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__api_name = api_name

    def get_rig_design(self) -> pd.DataFrame:
        """
        Ejecuta la consulta para obtener el diseño de la plataforma (rig design) de la base de datos 'rig_design_db'
        y retorna un DataFrame con los datos formateados.

        Returns:
            DataFrame con la información de rig design.

        Raises:
            ValueError: Si ocurre un error en la consulta o al procesar los datos.
        """
        try:
            query = f"SELECT * FROM rig_design_{self.__client}_{self.__project};"
            data, error = query_execute(query, "rig_design_db", True, self.__api_name)
            if error:
                raise ValueError(f"Error en get_rig_design: {data}")

            df = pd.DataFrame(data)
            df.columns = [
                "id", "rig_id", "rig", "client", "stand_length", "depth_onb_thr",
                "depth_conn_thr", "depth_conn_start", "bd_conn", "depth_super_thr",
                "depth_ooh_thr", "depth_trip_thr", "depth_start_change_sld",
                "hl_conn_drill_thr", "hl_conn_drill1_thr", "hl_conn_trip_thr",
                "hl_conn_trip1_thr", "hl_null_thr", "gpm_thr", "rpm_thr", "spp_thr",
                "spp_stat_thr", "wob_thr", "rpm_rot_thr", "rpm_rot_thr_change_sld",
                "rpm_stat_thr", "n_tseq_static", "n_tseq_trip", "n_tseq_circ",
                "filter_dict", "filter_dict_1", "filter_dict_5", "filter_dict_10",
                "filter_dict_15"
            ]
            df = df.drop(columns=["id"])
            return df

        except Exception as e:
            raise ValueError(f"Error en get_rig_design: {str(e)}") from e
