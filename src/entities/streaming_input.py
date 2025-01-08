import json
import pandas as pd
from utils_backend import query_execute, logging_report

class StreamingInput:
    def __init__(self, client, project, stream, username, scenario, api_name):
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.api_name = api_name

    def get_current_bit_size(self, input_folder, current_measured_depth):
        """Obtiene el bit size actual y lista de diámetros previos"""
        try:
            plan_path = f'{input_folder}plan/time_depth_plan.csv'
            df = pd.read_csv(plan_path)
            df['measured_depth'] = pd.to_numeric(df['measured_depth'], errors='coerce')
            
            current_hole_diameter = df[df['measured_depth'] <= current_measured_depth]['hole_diameter'].iloc[-1]
            
            distinct_hole_diameters = df['hole_diameter'].unique()
            currt_n_prev_hole_diameters = [str(diameter).replace(".", "p") for diameter in distinct_hole_diameters]
            
            logging_report(f'EXECUTED | 200017 | current_hole_diameter retrieved', 'INFO', self.api_name)
            
            return current_hole_diameter, currt_n_prev_hole_diameters
            
        except Exception as e:
            logging_report(f'FAILURE | 400017 | Error getting bit size: {str(e)}', 'ERROR', self.api_name)
            raise

    def get_wells_select_name(self):
        """Obtiene la lista de wells_select_name"""
        try:
            wells_select_name = []
            database_name = 'wcr_summary_well_bs_all_db'
            query = f'select well_id_name from t_{self.client}_{self.project}_{self.stream}_{self.username}_{self.scenario};'
            
            data, error = query_execute(query, database_name, True, self.api_name)
            if error:
                logging_report(f'FAILURE | 400020 | Error en query', 'ERROR', self.api_name)
            else:
                for row in data:
                    well_in_row = str(row[0]).split('-',1)[1].lstrip()
                    if well_in_row != 'RT':
                        wells_select_name.append(well_in_row)
                        
                logging_report(f'EXECUTED | 200020 | wells_select_name retrieved', 'INFO', self.api_name)
                
            return wells_select_name
        except Exception as e:
            logging_report(f'FAILURE | 400020 | Error getting wells: {str(e)}', 'ERROR', self.api_name)
            raise

    def create_inputs_json(self, input_folder, current_hole_diameter):
        """Crea el JSON con los inputs necesarios"""
        try:
            wells_select_name = self.get_wells_select_name()
            
            inputs_for_rt = {
                'current_bit_size': current_hole_diameter,
                'CURRENT_WELL_NAME': 'RT',
                'WELLS_SELECT_NAME': wells_select_name
            }
            
            json_path = f'{input_folder}inputs_for_rt.json'
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(inputs_for_rt, f, indent=6)
                
            logging_report(f'EXECUTED | 200021 | JSON file created', 'INFO', self.api_name)
            
        except Exception as e:
            logging_report(f'FAILURE | 400021 | Error creating JSON: {str(e)}', 'ERROR', self.api_name)
            raise