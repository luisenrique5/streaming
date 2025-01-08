from flask import jsonify, request, Response
from src.services.streamingServices.streaming_output_services import GetStreamingOutput
from utils_backend import api_token_check, logging_report
import json

class StreamingOutput:
    def __init__(self, client, project, stream, username, scenario, number_of_rows):
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.number_of_rows = number_of_rows
        self.api_name = 'streaming_output'

    def get_streaming_output(self):
        token_check = api_token_check(
            request.headers,
            self.client,
            self.project,
            self.stream,
            self.username,
            self.scenario,
            api_name=self.api_name
        )

        token_check_dict = json.loads(token_check)

        if token_check_dict.get("statusType") != "VALID":
            return Response(
                response=token_check, 
                status=401,
                mimetype="application/json"
            )

        try:
            streaming_service = GetStreamingOutput(
                self.client,
                self.project,
                self.stream,
                self.username,
                self.scenario,
                self.number_of_rows
            )
            results = streaming_service.calculate_get_streaming_output()

            output_message = (
                f"{self.number_of_rows} streaming rows of data retrieved. "
                f"client: '{self.client}', project: '{self.project}', stream: '{self.stream}'"
            )
            logging_report(f"{self.api_name} 200044 SUCCESS.", 'INFO', self.api_name)

            return jsonify({
                'apiStatus': "success",
                'requestData': results,
                'statusType': "Success",
                'statusCode': 200042,
                'statusMessage': output_message
            }), 200

        except Exception as e:
            logging_report(f'500000 | Error on {self.api_name}: {str(e)}', 'ERROR', self.api_name)
            return jsonify({
                'apiStatus': "error",
                'requestData': None,
                'statusType': "InternalError",
                'statusCode': 500,
                'statusMessage': f"An error occurred while retrieving results: {str(e)}"
            }), 500
