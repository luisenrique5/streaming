# src/repositories/streaming_repository.py
import pandas as pd
from utils_backend import query_execute, get_data_at_interval, logging_report

class StreamingRepository:
    def __init__(self, client, project, stream, username, scenario, api_name):
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.api_name = api_name
        self.streaming_db = f'streaming_drillbi_{client}_{project}_db'
        self.streaming_tbd_table = f'streaming_tbd_{stream}'
        self.streaming_os_table = f'streaming_os_{stream}'

    def get_real_time_tbd(self):
        try:
            data = get_data_at_interval(
                self.client, self.project, self.stream, 
                self.username, self.scenario, 
                self.streaming_db, self.streaming_tbd_table, 
                self.api_name, interval_seconds=10
            )
            
            df = pd.DataFrame(data, columns=[
                'id', 'datetime','measured_depth','bit_depth','block_height',
                'rop','wob','hook_load','flow_rate','spp','torque','surface_rpm',
                'motor_rpm','bit_rpm'
            ])
            del df['id']
            return df
            
        except Exception as e:
            logging_report(f'FAILURE | 400015 | {self.client} | {self.project} | {self.stream} | {self.username} | {self.scenario} | Error getting TBD data: {str(e)}', 'ERROR', self.api_name)
            return None

    def get_real_time_os(self):
        query = f'select * from {self.streaming_os_table} order by measured_depth asc;'
        data, error = query_execute(query, self.streaming_db, True, self.api_name)
        if error:
            logging_report(f'FAILURE | 400016 | {self.client} | {self.project} | {self.stream} | {self.username} | {self.scenario} | {data}', 'ERROR', self.api_name)
            return None
            
        df = pd.DataFrame(data, columns=['id','measured_depth','incl','azm'])
        del df['id']
        return df

    def get_current_measured_depth(self):
        query = f"select measured_depth from {self.streaming_tbd_table} order by id desc limit 1"
        data, error = query_execute(query, self.streaming_db, False, self.api_name)
        if error:
            logging_report(f'FAILURE | 400017 | {self.client} | {self.project} | {self.stream} | {self.username} | {self.scenario} | {data}', 'ERROR', self.api_name)
            return None
        return float(data[0])