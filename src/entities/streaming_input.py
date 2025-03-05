import json
import pandas as pd
import concurrent.futures
from threading import Lock
from io import StringIO
from utils_backend import query_execute

###############################################################################
# POOL DE HILOS Y LOCKS GLOBALES
###############################################################################
json_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# Caché global de DataFrames para no parsear el mismo JSON repetidamente.
_plan_df_cache = {}
_plan_df_lock = Lock()

_json_lock = Lock()

def _read_json_from_redis_in_thread(redis_client, redis_key: str) -> pd.DataFrame:
    """
    Lee el contenido JSON en Redis y lo convierte a DataFrame.
    - Espera un array JSON (orient="records"), ej:
      [
        {"cumulative_time":0,"measured_depth":0,"hole_diameter":17.5, ...},
        ...
      ]
    """
    try:
        json_content = redis_client.get(redis_key)
        if not json_content:
            raise ValueError(f"No JSON content found in Redis key '{redis_key}'")

        # Leemos en memoria con orient="records"
        # StringIO para que pandas trate el string como un archivo
        df = pd.read_json(StringIO(json_content), orient="records")
        return df

    except Exception as e:
        raise ValueError(f"Error reading JSON from Redis key '{redis_key}': {str(e)}") from e


class StreamingInput:
    def __init__(self, client, project, stream, username, scenario, api_name, base_key, redis_connection):
        self._client = client
        self._project = project
        self._stream = stream
        self._username = username
        self._scenario = scenario
        self._api_name = api_name
        self._base_key = base_key
        self._redis = redis_connection

    def get_current_bit_size(self, current_measured_depth: float):
        print("base_key", self._base_key, "redis", self._redis)
        redis_key_for_json = f"{self._base_key}:plan:time_depth_plan"

        try:
            with _plan_df_lock:
                if redis_key_for_json in _plan_df_cache:
                    df = _plan_df_cache[redis_key_for_json]
                else:
                    future = json_thread_pool.submit(
                        _read_json_from_redis_in_thread,
                        self._redis,
                        redis_key_for_json
                    )
                    df = future.result()
                    _plan_df_cache[redis_key_for_json] = df

            if df.empty:
                raise ValueError(f"No data found in JSON from Redis key '{redis_key_for_json}'")

            df['measured_depth'] = pd.to_numeric(df['measured_depth'], errors='coerce')
            filtered_df = df[df['measured_depth'] <= current_measured_depth]

            if filtered_df.empty:
                return None, []

            current_hole_diameter = filtered_df['hole_diameter'].iloc[-1]
            distinct_diameters = df['hole_diameter'].unique()

            return current_hole_diameter, distinct_diameters
        except Exception as e:
            raise ValueError(f"Error calculating current bit size: {str(e)}") from e

    def get_wells_select_name(self):
        """
        Obtiene la lista de pozos (well_id_name) desde la base wcr_summary_well_bs_all_db,
        omitiendo los que tengan 'RT'.
        """
        wells_select_name = []
        query = (
            f"SELECT well_id_name "
            f"FROM t_{self._client}_{self._project}_{self._stream}_{self._username}_{self._scenario};"
        )
        try:
            data, error = query_execute(query, "wcr_summary_well_bs_all_db", True, self._api_name)
            if not error and data:
                for row in data:
                    parts = str(row[0]).split('-', 1)
                    if len(parts) > 1:
                        well_in_row = parts[1].strip()
                        if well_in_row != 'RT':
                            wells_select_name.append(well_in_row)
            return wells_select_name
        except Exception as e:
            raise ValueError(f"Error querying wells_select_name: {str(e)}") from e

    def create_inputs_json(self, current_hole_diameter):
        redis_key_for_json = f"{self._base_key}:inputs_for_rt"
        try:
            wells_select_name = self.get_wells_select_name()
            inputs_for_rt = {
                'current_bit_size': current_hole_diameter,
                'CURRENT_WELL_NAME': 'RT',
                'WELLS_SELECT_NAME': wells_select_name
            }
            with _json_lock:
                self._redis.set(redis_key_for_json, json.dumps(inputs_for_rt))
        except Exception as e:
            raise ValueError(f"Error creating JSON in Redis: {str(e)}") from e
