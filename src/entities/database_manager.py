import pandas as pd
from utils_backend import query_execute, get_data_at_interval

class DatabaseManager:
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

    def get_well_general_data(self):
        query = f'select * from well_general_{self.client}_{self.project};'
        data, error = query_execute(query, 'well_general_db', True, self.api_name)
        if not error:
            df = pd.DataFrame(data)
            df.columns = ['id','well_id','well_name','well_type','client','rig','field',
                         'spud_date','bi_datetime_from','bi_datetime_to','total_time',
                         'total_depth','location','latitude','longitude']
            del df['id']
            return df
        return None