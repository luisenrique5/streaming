# rig_repository.py

from sqlalchemy import select

class RigRepository:
    """
    Repositorio para acceder a los datos de diseño de la plataforma (rig design),
    usando un enfoque similar a 'QueryFindParameterValuesByNamesAndDateRange'.

    En este estilo, recibimos un 'db_connection' que expone un método:
        db_connection.get_table(<nombre_de_tabla>)
    Y retornamos un statement final (CTE o SELECT) para que sea ejecutado
    externamente. NO importamos la clase de conexión directamente.
    """

    def __init__(
        self,
        db_connection,
        client: str,
        project: str,
        stream: str,
        username: str,
        scenario: str,
        api_name: str
    ) -> None:
        """
        Args:
            db_connection: Objeto que debe tener un método get_table(nombre_tabla)
            client: Nombre del cliente.
            project: Nombre del proyecto.
            stream: Nombre del stream.
            username: Nombre de usuario.
            scenario: Escenario.
            api_name: Nombre de la API (para logs o referencia).
        """
        self.__db_connection = db_connection
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__api_name = api_name

    def get_rig_design(self):
        """
        Construye la consulta SQLAlchemy para leer la tabla 'rig_design_{client}_{project}'
        y retorna un statement final (CTE) que luego se puede ejecutar externamente.

        Returns:
            SQLAlchemy Select: una consulta que se puede ejecutar con .execute()
            para extraer las filas de rig design.
        
        Raises:
            ValueError: Si ocurre un error creando la consulta.
        """
        try:
            # 1) Obtener tabla a través de db_connection
            table_name = f"rig_design_{self.__client}_{self.__project}"
            rig_table = self.__db_connection.get_table(table_name)

            # 2) Definir CTE con todas las columnas de la tabla rig_design
            #    (similar a tu ejemplo con QueryFindParameterValuesByNamesAndDateRange).
            rig_table_data = (
                select(
                    rig_table.c.id,
                    rig_table.c.rig_id,
                    rig_table.c.rig,
                    rig_table.c.client,
                    rig_table.c.stand_length,
                    rig_table.c.depth_onb_thr,
                    rig_table.c.depth_conn_thr,
                    rig_table.c.depth_conn_start,
                    rig_table.c.bd_conn,
                    rig_table.c.depth_super_thr,
                    rig_table.c.depth_ooh_thr,
                    rig_table.c.depth_trip_thr,
                    rig_table.c.depth_start_change_sld,
                    rig_table.c.hl_conn_drill_thr,
                    rig_table.c.hl_conn_drill1_thr,
                    rig_table.c.hl_conn_trip_thr,
                    rig_table.c.hl_conn_trip1_thr,
                    rig_table.c.hl_null_thr,
                    rig_table.c.gpm_thr,
                    rig_table.c.rpm_thr,
                    rig_table.c.spp_thr,
                    rig_table.c.spp_stat_thr,
                    rig_table.c.wob_thr,
                    rig_table.c.rpm_rot_thr,
                    rig_table.c.rpm_rot_thr_change_sld,
                    rig_table.c.rpm_stat_thr,
                    rig_table.c.n_tseq_static,
                    rig_table.c.n_tseq_trip,
                    rig_table.c.n_tseq_circ,
                    rig_table.c.filter_dict,
                    rig_table.c.filter_dict_1,
                    rig_table.c.filter_dict_5,
                    rig_table.c.filter_dict_10,
                    rig_table.c.filter_dict_15
                )
            ).cte("rig_table_data")

            # 3) Retornar la query final (SELECT * FROM rig_table_data)
            final_query = select(rig_table_data)
            return final_query

        except Exception as e:
            raise ValueError(f"Error en get_rig_design: {str(e)}") from e
