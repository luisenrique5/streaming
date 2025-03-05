# src/repositories/streaming_repository.py
import pandas as pd
from utils_backend import query_execute, get_data_at_interval, logging_report

class StreamingRepository:
    def __init__(self, client, project, stream, username, scenario, api_name):
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__api_name = api_name

        self.__streaming_db = f'streaming_drillbi_{client}_{project}_db'
        self.__streaming_tbd_table = f'streaming_tbd_{stream}'
        self.__streaming_os_table = f'streaming_os_{stream}'

    def get_real_time_tbd(self):
        try:
            data = get_data_at_interval(
                self.__client, self.__project, self.__stream,
                self.__username, self.__scenario,
                self.__streaming_db,
                self.__streaming_tbd_table,
                self.__api_name,
                interval_seconds=10
            )

            df = pd.DataFrame(data, columns=[
                'id', 'datetime', 'measured_depth', 'bit_depth', 'block_height',
                'rop', 'wob', 'hook_load', 'flow_rate', 'spp', 'torque',
                'surface_rpm', 'motor_rpm', 'bit_rpm'
            ])

            df.drop(columns='id', inplace=True)
            return df

        except Exception as e:
            raise ValueError(f"Error en get_real_time_tbd: {str(e)}")

    def get_real_time_os(self):
        try: 
            query = f'SELECT * FROM {self.__streaming_os_table} ORDER BY measured_depth ASC;'
            data, error = query_execute(query, self.__streaming_db, True, self.__api_name)

            if error:
                raise ValueError(f"Error en get_real_time_os: {data}")

            df = pd.DataFrame(data, columns=['id', 'measured_depth', 'incl', 'azm'])
            df.drop(columns='id', inplace=True)
            return df
        
        except Exception as e:
            raise ValueError(f"Error en get_real_time_os: {str(e)}")

    def get_current_measured_depth(self):
        try: 
            query = (
                f"SELECT measured_depth "
                f"FROM {self.__streaming_tbd_table} "
                f"ORDER BY id DESC LIMIT 1"
            )
            data, error = query_execute(query, self.__streaming_db, False, self.__api_name)
            if error:
                raise ValueError(f"Error en get_current_measured_depth: {data}")

            return float(data[0])
        
        except Exception as e:
            raise ValueError(f"Error en get_current_measured_depth: {str(e)}")
