# connection_sqlalchemy_repository.py
import os
from sqlalchemy import create_engine

class ConnectionSqlAlchemyRepository:
    """
    Clase encargada de crear y retornar la conexión (engine) a PostgreSQL mediante SQLAlchemy.
    """
    def __init__(self, database_name: str):
        self.database_name = database_name

        # Validación: si la base inicia con 'streaming', usamos credenciales de 'streaming'
        if self.database_name.startswith("streaming"):
            db_password = os.getenv("DBRTSRVPASSWORD")     # Ajusta si tienes otro nombre de variable
            db_host = os.getenv("DBRTSRVENDPOINT")        # idem
        else:
            db_password = os.getenv("DBSRVSRVDRILLBIPASSW")
            db_host = os.getenv("DBSRVDRILLBIEP")

        if not db_password or not db_host:
            raise ValueError(
                f"No se encontraron las variables de entorno para conectarse a la base '{self.database_name}'. "
                f"Revisa que estén configuradas correctamente."
            )

        # Puerto por defecto de PostgreSQL
        db_port = 5432

        # Construimos la URL de conexión
        connection_url = f"postgresql://postgres:{db_password}@{db_host}:{db_port}/{self.database_name}"

        # Creamos el engine
        self.__engine = create_engine(connection_url)

    def get_connection(self):
        """
        Retorna el engine de SQLAlchemy para la base de datos configurada.
        """
        return self.__engine
