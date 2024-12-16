from datetime import datetime
import os

class StreamingEnvironment:
    def __init__(self, client, project, stream, username, scenario, environment):
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.environment = environment
        self.current_date = datetime.now().strftime("%Y%m%d-%H")

    def setup_environment(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base_output = os.path.join(project_root, "output_data")
        
        base_path = os.path.join(base_output, self.client, self.project, 
                                self.stream, self.username, self.scenario)
        
        os.makedirs(base_path, exist_ok=True)
        
        csv_path = os.path.join(base_path, "csv")
        real_time_path = os.path.join(base_path, "real_time_update", "csv")
        os.makedirs(csv_path, exist_ok=True)
        os.makedirs(real_time_path, exist_ok=True)
        
        file_suffix = f"{self.stream}_{self.current_date}"
        log_file = os.path.join(base_path, f"getstreamingoutputrt_{file_suffix}.log")
     
        return log_file