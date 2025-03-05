import redis

class DirectoryStructureRedis:
    def __init__(self, client, project, stream, username, scenario, redis_connection):
        """
        Emula tu clase DirectoryStructure, pero cada directorio es una clave separada
        en Redis. Usamos ':' como separador para que RedisInsight muestre subniveles.
        """
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.redis = redis_connection

    def create_directory_structure(self):
        """
        Emula la creación de:
         input_data/{client}/{project}/{stream}/{username}/{scenario}/
           ├── csv
           ├── database
           ├── plan
           ├── real_time
           └── real_time_update
               └── csv

        Pero cada carpeta es una clave con formato 'input_data:client:project:...'
        """
        base_key = f"input_data:{self.client}:{self.project}:{self.stream}:{self.username}:{self.scenario}"

        # Lista de subcarpetas
        directories = [
            base_key,
            f"{base_key}:csv",
            f"{base_key}:database",
            f"{base_key}:plan",
            f"{base_key}:real_time",
            f"{base_key}:real_time_update",
            f"{base_key}:real_time_update:csv"
        ]

        try:
            # Para que RedisInsight muestre algo, guardamos un valor
            # En este caso, un simple "{}" o "" o lo que gustes
            for directory in directories:
                if not self.redis.exists(directory):
                    self.redis.set(directory, "{}")

            return base_key  # Por si quieres retornarlo
        except redis.exceptions.RedisError as e:
            raise ValueError(f"Error creando estructura en Redis: {str(e)}")
