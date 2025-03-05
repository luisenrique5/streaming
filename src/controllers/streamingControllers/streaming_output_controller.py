from src.services.api_token_services import Authorization
from src.services.logging_report_services import LoggingReport
from src.services.streamingServices.streaming_output_services import StreamingData

class StreamingOutput:
    def __init__(self, client, project, stream, username, scenario, number_of_rows):
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__number_of_rows = number_of_rows
        
    def get_streaming_output(self):
        api_name = 'get_streaming_output'
        
        try:
            Authorization.token_check(self.__client, self.__scenario, api_name)
            
            Streaming_data = StreamingData(self.__client, self.__project, self.__stream, self.__username, self.__scenario, self.__number_of_rows)
            results = Streaming_data.get_data()
            return LoggingReport.logging_data(results, api_name)
         
        except Exception as e:
            return LoggingReport.logging_error(api_name, e)   


