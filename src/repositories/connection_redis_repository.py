import os
import redis
from dotenv import load_dotenv

load_dotenv()

class ConnectionRedisRepository:
    __pool = None

    def __init__(self):
        if ConnectionRedisRepository.__pool is None:
            host = os.getenv("REDIS_HOST")
            port = int(os.getenv("REDIS_PORT"))
            password = os.getenv("REDIS_PASSWORD")
          
            ConnectionRedisRepository.__pool = redis.ConnectionPool(
                host=host,
                port=port,
                password=password,
                decode_responses=True 
            )
        self.__connection = redis.Redis(connection_pool=ConnectionRedisRepository.__pool)

    def get_connection(self):
        return self.__connection
