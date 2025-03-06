import pandas as pd
from io import StringIO
import logging
from typing import Any, Optional, List

def read_json_from_redis(redis_client: Any, redis_key: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Lee un JSON (orient='records') desde Redis y lo convierte a un DataFrame.

    Args:
        redis_client: Cliente de Redis.
        redis_key (str): Clave en Redis.
        parse_dates (Optional[List[str]]): Lista de columnas a convertir a datetime.

    Returns:
        pd.DataFrame: DataFrame con los datos leídos. Si no hay contenido, retorna un DataFrame vacío.

    Raises:
        Exception: Si ocurre un error durante la lectura o conversión del JSON.
    """
    try:
        json_content = redis_client.get(redis_key)
        if not json_content:
            logging.debug(f"No content in Redis key: {redis_key}")
            return pd.DataFrame()
        
        # En caso de que json_content sea de tipo bytes, decodificarlo a string.
        if isinstance(json_content, bytes):
            json_content = json_content.decode('utf-8')
            
        df = pd.read_json(StringIO(json_content), orient="records")
        if parse_dates:
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        logging.debug(f"Read DataFrame from {redis_key}, shape={df.shape}")
        return df

    except Exception as e:
        logging.error(f"Error reading JSON from Redis key '{redis_key}': {str(e)}")
        raise
