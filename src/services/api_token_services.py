from flask import Response, request

# import backend utilities
from utils_backend import *

class Authorization:
    
    @staticmethod
    def token_check(operator_name, lease_name, api_name):
            ####################### check if API token is valid #######################
        token_check = api_token_check(request.headers, operator_name, lease_name, api_name)
        token_check_dict=json.loads(token_check)
        if str(token_check_dict["statusType"]) != 'SUCCESS':
            # raise ValueError("Token inválido")
            
            raise ValueError(token_check_dict["statusMessage"])
        ###########################################################################
