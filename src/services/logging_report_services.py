from utils_backend import *
from flask import jsonify, Response

class LoggingReport:

    @staticmethod
    def logging(api_name):
        logging_report("JSON data successfully processed.", 'INFO', api_name)
        return jsonify({
            'apiStatus': True,
            'data': None,
            'statusType': 'SUCCESS',
            'statusCode': 200044,
            'statusMessage': 'JSON data successfully processed.'
        })
            
    @staticmethod
    def logging_error(api_name, e):     
        errMessage = f'Error: {str(e)}'
        apiStatus = True
        requestData = None
        statusType = "ERROR"
        statusCode = 500201
        statusMessage = errMessage
        json_return = json_return_constructor(apiStatus, requestData, statusType, 
                        statusCode, statusMessage)
        logging_report(f'{statusCode} | {statusMessage}', 'ERROR', api_name)
        resp = Response(response=json_return, status=500, mimetype="application/json")
        resp.headers["Content-Type"] = "application/json"
        return resp

    @staticmethod
    def logging_data(result_data, api_name):
        logging_report("result data successfully retrieved.", 'INFO', api_name)
        return jsonify({
                'apiStatus': True,
                'data': result_data,
                'statusType': 'SUCCESS',
                'statusCode': 200044,
                'statusMessage': f"Information delivered correctly"
        })
    @staticmethod
    def deleted_data(api_name):
        logging_report("Data deleted successfully", 'INFO', api_name)
        return jsonify({
                'apiStatus': True,
                'data': None,
                'statusType': 'SUCCESS',
                'statusCode': 200044,
                'statusMessage': f"Data deleted successfully"
        })