import redis
from typing import List

class DirectoryStructureRedis:
    """
    Emula la creación de una estructura de "directorios" en Redis, donde cada "directorio"
    se representa como una clave con el formato:
        input_data:client:project:stream:username:scenario
    """

    def __init__(self, client: str, project: str, stream: str, username: str, scenario: str, redis_client: redis.Redis) -> None:
        """
        Args:
            client: Nombre del cliente.
            project: Nombre del proyecto.
            stream: Nombre del stream.
            username: Nombre de usuario.
            scenario: Escenario.
            redis_client: Conexión a Redis.
        """
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.redis_client = redis_client

    def create_directory_structure(self) -> str:
        """
        Emula la creación de la siguiente estructura en Redis:

            input_data/{client}/{project}/{stream}/{username}/{scenario}
                ├── csv
                ├── database
                ├── plan
                ├── real_time
                └── real_time_update
                    └── csv

        Cada "directorio" se guarda como una clave con el separador ':'.

        Returns:
            La clave base con la estructura 'input_data:client:project:stream:username:scenario'
        """
        base_key = f"input_data:{self.client}:{self.project}:{self.stream}:{self.username}:{self.scenario}"
        directories: List[str] = [
            base_key,
            f"{base_key}:csv",
            f"{base_key}:database",
            f"{base_key}:plan",
            f"{base_key}:real_time",
            f"{base_key}:real_time_update",
            f"{base_key}:real_time_update:csv"
        ]

        try:
            pipe = self.redis_client.pipeline()
            # Con setnx se establece el valor solo si la clave no existe
            for directory in directories:
                pipe.setnx(directory, "{}")
            pipe.execute()
            return base_key
        except redis.exceptions.RedisError as error:
            raise ValueError(f"Error creando estructura en Redis: {error}") from error
