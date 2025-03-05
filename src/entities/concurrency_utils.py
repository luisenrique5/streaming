# src/entities/concurrency_utils.py

from concurrent.futures import ThreadPoolExecutor, as_completed
from .read_redis_utils import read_json_from_redis

def read_json_from_redis_async(redis_client, redis_key, parse_dates=None):
    """Función wrapper para usar en submit(...) del ThreadPoolExecutor."""
    return read_json_from_redis(redis_client, redis_key, parse_dates)

# Creamos un pool global (si quieres un pool local, lo defines en la clase)
json_thread_pool = ThreadPoolExecutor(max_workers=8)
