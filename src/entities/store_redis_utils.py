import pandas as pd
import logging
from typing import Any

def store_df_in_redis(df: pd.DataFrame, redis_client: Any, base_key: str, subkey: str) -> None:
    """
    Convierte un DataFrame a JSON (orient='records') y lo almacena en Redis
    bajo la clave formada como "{base_key}:{subkey}".

    Args:
        df (pd.DataFrame): DataFrame a convertir y almacenar. Si es None o está vacío,
                           se guarda como una lista vacía "[]".
        redis_client (Any): Cliente de Redis.
        base_key (str): Clave base (por ejemplo, "output_data:client:project:...").
        subkey (str): Subclave a concatenar (por ejemplo, "time_based_drill_current_well_out").
    """
    # Verificar si el DataFrame es None o está vacío
    if df is None or df.empty:
        json_str = "[]"
        df_shape = (0, 0)
    else:
        json_str = df.to_json(orient="records")
        df_shape = df.shape

    redis_key = f"{base_key}:{subkey}"
    redis_client.set(redis_key, json_str)
    logging.debug(f"Stored DF (shape={df_shape}) in Redis key: {redis_key}")
