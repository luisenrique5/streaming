# src/entities/read_redis_utils.py

import pandas as pd
from io import StringIO
import logging

def read_json_from_redis(redis_client, redis_key: str, parse_dates=None) -> pd.DataFrame:
    """
    Lee un JSON (orient='records') desde Redis y lo convierte a DataFrame.
    - redis_key: la clave en Redis
    - parse_dates: lista de columnas a parsear como fecha
    """
    json_content = redis_client.get(redis_key)
    if not json_content:
        logging.debug(f"No content in Redis key: {redis_key}")
        return pd.DataFrame()

    df = pd.read_json(StringIO(json_content), orient="records")
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    logging.debug(f"Read DataFrame from {redis_key}, shape={df.shape}")
    return df
