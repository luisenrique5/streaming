# well_repository.py
from sqlalchemy import select, text, func, literal, union_all
import traceback

class WellRepository:
    """
    Repositorio que, de forma similar a QueryFindParameterValuesByNamesAndDateRange,
    utiliza SQLAlchemy para consultar dos tablas:
      1) well_general_{client}_{project}
      2) time_based_drill_table_{client}_{project}

    En vez de ejecutar directamente, retornamos la query final (o cte) para
    que el 'service' decida cómo y cuándo ejecutar el statement.
    """

    def __init__(self, db_connection, client: str, project: str, stream: str,
                 username: str, scenario: str, api_name: str) -> None:
        """
        Args:
            db_connection: Conexión o wrapper de SQLAlchemy que tenga un método
                get_table(tabla: str) -> Table
            client (str): Cliente
            project (str): Proyecto
            stream (str): Nombre de la 'corriente' o 'stream'
            username (str): Usuario
            scenario (str): Escenario
            api_name (str): Nombre del API que consume este repositorio (para referencia)
        """
        self.__db_connection = db_connection
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__api_name = api_name

    def get_well_general(self):
        """
        Construye la consulta (en estilo SQLAlchemy) a la tabla
        well_general_{client}_{project}. Retorna el statement final, que luego
        puede ejecutarse con db_connection.execute(...) desde el 'service'.
        """
        try:
            # 1) Obtenemos la tabla de la conexión
            table_name = f"well_general_{self.__client}_{self.__project}"
            well_table = self.__db_connection.get_table(table_name)

            # 2) Definimos un CTE con todos los datos de la tabla
            #    (similar a tu ejemplo, aunque sea un SELECT directo)
            well_table_data = (
                select(well_table.c.id,
                       well_table.c.well_id,
                       well_table.c.well_name,
                       well_table.c.well_type,
                       well_table.c.client,
                       well_table.c.rig,
                       well_table.c.field,
                       well_table.c.spud_date,
                       well_table.c.bi_datetime_from,
                       well_table.c.bi_datetime_to,
                       well_table.c.total_time,
                       well_table.c.total_depth,
                       well_table.c.location,
                       well_table.c.latitude,
                       well_table.c.longitude)
            ).cte("well_table_data")

            # 3) Retornamos una consulta final; puede ser simplemente select de la CTE
            final_query = select(well_table_data)

            return final_query

        except Exception as e:
            error_traceback = traceback.format_exc()
            raise ValueError(f"Error en get_well_general: {str(e)}\n\nTraceback completo:\n{error_traceback}") from e

    def get_time_based_drill(self):
        """
        Construye la consulta (en estilo SQLAlchemy) a la tabla
        time_based_drill_table_{client}_{project}.
        Retorna un statement final (CTE), similar al método anterior.
        """
        try:
            # 1) Obtenemos la tabla de la conexión
            table_name = f"time_based_drill_table_{self.__client}_{self.__project}"
            drill_table = self.__db_connection.get_table(table_name)

            # 2) Definimos un CTE con todas las columnas
            drill_table_data = (
                select(drill_table.c.id,
                       drill_table.c.well_id,
                       drill_table.c.datetime,
                       drill_table.c.cumulative_time,
                       drill_table.c.day_number,
                       drill_table.c.measured_depth,
                       drill_table.c.tvd,
                       drill_table.c.incl,
                       drill_table.c.azm,
                       drill_table.c.dls,
                       drill_table.c.well_section,
                       drill_table.c.bit_depth,
                       drill_table.c.hole_diameter,
                       drill_table.c.formation,
                       drill_table.c.block_height,
                       drill_table.c.rop,
                       drill_table.c.wob,
                       drill_table.c.hook_load,
                       drill_table.c.flow_rate,
                       drill_table.c.pit_volume,
                       drill_table.c.diff_pressure,
                       drill_table.c.spp,
                       drill_table.c.annular_pressure,
                       drill_table.c.torque,
                       drill_table.c.surface_rpm,
                       drill_table.c.motor_rpm,
                       drill_table.c.bit_rpm,
                       drill_table.c.mse,
                       drill_table.c.mud_motor,
                       drill_table.c.casing,
                       drill_table.c.rig_super_state,
                       drill_table.c.rig_sub_activity,
                       drill_table.c.consecutive_labels
                )
            ).cte("drill_table_data")

            # 3) Consulta final
            final_query = select(drill_table_data)

            return final_query

        except Exception as e:
            raise ValueError(f"Error en get_time_based_drill: {str(e)}") from e
