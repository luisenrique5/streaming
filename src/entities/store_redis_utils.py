# src/entities/store_redis_utils.py

import pandas as pd
from io import StringIO
import logging

def store_df_in_redis(df: pd.DataFrame, redis_client, base_key: str, subkey: str):
    """
    Convierte un DataFrame a JSON orient='records' y lo guarda en:
      {base_key}:{subkey}
    """
    if df is None or df.empty:
        json_str = "[]"
    else:
        json_str = df.to_json(orient="records")

    redis_key = f"{base_key}:{subkey}"
    redis_client.set(redis_key, json_str)
    logging.debug(f"Stored DF (shape={df.shape}) in Redis key: {redis_key}")
