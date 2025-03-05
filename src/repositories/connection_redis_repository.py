import os
import redis
from dotenv import load_dotenv

# Cargamos las variables de entorno desde el .env
load_dotenv()

class ConnectionRedisRepository:
    def __init__(self):
        """
        Emula la lógica de conexión que tenías con PostgreSQL, pero usando Redis.
        Los parámetros operator, lease, region son ilustrativos; 
        úsalos si necesitas personalizar la conexión o la base de datos en Redis.
        """
        self.__host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.__port = int(os.getenv("REDIS_PORT", 6379))
        self.__password = os.getenv("REDIS_PASSWORD", None)

        # Creamos la conexión a Redis
        self.__connection = redis.Redis(
            host=self.__host,
            port=self.__port,
            password=self.__password,
            decode_responses=True  # para que devuelva strings en lugar de bytes
        )

    def get_connection(self):
        """
        Retorna el cliente de Redis para que puedas usarlo en otras clases
        (similar a get_connection() de SQLAlchemy).
        """
        return self.__connection
