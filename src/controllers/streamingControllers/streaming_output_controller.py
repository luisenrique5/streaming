from flask import jsonify, request, Response
from src.services.streamingServices.streaming_output_services import GetStreamingOutput
from utils_backend import *

class StreamingOutput:
    def __init__(self, client,project, stream, username, scenario, number_of_rows):
        self.__client = client
        self.__project = project
        self.__stream = stream
        self.__username = username
        self.__scenario = scenario
        self.__number_of_rows = number_of_rows
        
    def get_streaming_output(self):
        api_name = 'streaming_output'
        
        token_check = api_token_check(request.headers, self.__client, self.__username, api_name)
        token_check_dict = json.loads(token_check)
        if str(token_check_dict["statusType"]) != 'SUCCESS':
            resp = Response(response=token_check, status=401, mimetype="application/json")
            resp.headers["Content-Type"] = "application/json"
            return resp

        try:
            streaming_output_services = GetStreamingOutput(
                self.__client,
                self.__project,
                self.__stream,
                self.__username,
                self.__scenario,
                self.__number_of_rows
            )
            results = streaming_output_services.calculate_get_streaming_output()
            
            logging_report(f"{api_name} 200044 SUCCESS.", 'INFO', api_name)

            return jsonify({
                'apiStatus': "success",
                'requestData': results,
                'statusType': "Success",
                'statusCode': 200,
                'statusMessage': "Results retrieved successfully"
            }), 200
        except Exception as e:
            logging_report(f'500000 | Error on {api_name}: {str(e)}', 'ERROR', api_name)
            return jsonify({
                'apiStatus': "error",
                'requestData': None,
                'statusType': "InternalError",
                'statusCode': 500,
                'statusMessage': f"An error occurred while retrieving results: {str(e)}"
            }), 500
               