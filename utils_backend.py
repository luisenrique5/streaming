## packages for local linux system access
import os

## packages for data treatment
import json

## packages for logging module
from datetime import datetime
import time
from functools import wraps
import logging

import psycopg2

from os.path import join, dirname
from dotenv import load_dotenv

## set environmet file "./.env"
dotenv_path = join(dirname(__file__), './env/.env')
load_dotenv(dotenv_path)

### get databases access values from environment variables
dbsrvendpoint = os.getenv('DBSRVPANALYTICSEP')
dbsrvpassword = os.getenv('DBSRVSRVPANALYTICSPASSW')
dbrtsrvendpoint = os.getenv('DBRTSRVENDPOINT')
dbrtsrvpassword = os.getenv('DBRTSRVPASSWORD')
APITOKEN2 = os.getenv('INTERNAL_API_TOKEN')

##### Opening config JSON file
with open('./config/config.json', 'r', encoding="utf-8") as file:
    config_json = file.read()

##### returns JSON object as a dictionary
config_dict = json.loads(config_json)

ENVIRONMENT = config_dict["ENVIRONMENT"]
LOGGINGLEVEL = config_dict["LOGGINGLEVEL"]
REMOTEDOCKERIMAGE = config_dict["REMOTEDOCKERIMAGE"]

#### set current date and time
current_datetime = datetime.now().strftime("%Y"+"-"+"%m"+"-"+"%d"+" "+"%H"+":"+"%M"+":"+"%S")
##############################

#*************************************************************************************************************#
#************************** Input functions for backend arquitecture deployment ******************************#
#*************************************************************************************************************#

############################################################################
####################### logging_report module ##############################
def logging_report(writing: str, level: str, api_name: str):
    """
    Function to generate and write events logs reports in the
    corresponding location and log file.
    """
    ##### set up log folder
    if ENVIRONMENT == 'development':
        logpath = '/home/ubuntu/log/prodanalytics_volume_tracker/requests'
    else:
        logpath = '/mnt/log/prodanalytics_volume_tracker/requests'

    ##### create log folder in case it doesn't exist
    save_path = f'{logpath}/{api_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.chown(save_path, 1000, 1000)

    trace_date = datetime.now().strftime("%Y"+"%m"+"%d")
    log_file_path = f'{save_path}{api_name}_{trace_date}.log'

    logger = logging.getLogger('mylogger')

    if logger.hasHandlers():
        # Logger is already configured, remove all handlers
        logger.handlers = []

    handler = logging.FileHandler(log_file_path)
    logger.addHandler(handler)
    formato = '%(asctime)s | %(levelname)s | %(message)s'
    handler.setFormatter(logging.Formatter(formato))
    logger.setLevel(LOGGINGLEVEL)

    if level == 'DEBUG':
        logger.debug("%s", writing)
    elif level == 'INFO':
        logger.info("%s", writing)
    elif level == 'WARNING':
        logger.warning("%s", writing)
    elif level == 'ERROR':
        logger.error("%s", writing)
    else:
        logger.critical("%s", writing)

    os.chmod(log_file_path, 0o666)
    os.chown(log_file_path, 1000, 1000)
################### end of logging_report module ###########################


############################################################################
########################## Json return module ##############################
def json_return_constructor(apiStatus, requestData, statusType, statusCode, statusMessage):
    dict_return = {\
        'apiStatus' : apiStatus,\
        'data' : requestData,\
        'statusType' : statusType,\
        'statusCode' : statusCode,\
        'statusMessage' : statusMessage,\
        }
    json_return = json.dumps(dict_return)
    return json_return
####################### end of json return module ##########################

############################################################################
########################## Token check module ##############################
def api_token_check(request_headers, wellname, username, api_name):
    current_datetime = datetime.now().strftime("%Y"+"-"+"%m"+"-"+"%d"+" "+"%H"+":"+"%M"+":"+"%S")

    auth = request_headers.get('Authorization', None)
    if auth is None:
        apiStatus = True
        requestData = None
        statusType = "ERROR"
        statusCode = 401001
        statusMessage = 'Authorization header is not defined.'
        json_return = json_return_constructor(apiStatus, requestData, statusType, statusCode, statusMessage)
        
        logging_report(f'{statusCode} | {statusMessage}', 'ERROR', api_name)
        return json_return

    try:
        parts = auth.split()
        token = parts[1]
    except:
        apiStatus = True
        requestData = None
        statusType = "ERROR"
        statusCode = 401002
        statusMessage = 'Token value not found.'
        json_return = json_return_constructor(apiStatus, requestData, statusType, statusCode, statusMessage)
        
        logging_report(f'{statusCode} | {statusMessage}', 'ERROR', api_name)
        return json_return
    
    if token != APITOKEN2:
        apiStatus = True
        requestData = None
        statusType = "ERROR"
        statusCode = 401003
        statusMessage = 'Invalid Token.'
        json_return = json_return_constructor(apiStatus, requestData, statusType, statusCode, statusMessage)
        
        logging_report(f'{statusCode} | {statusMessage}', 'ERROR', api_name)
        return json_return
    
    apiStatus = True
    requestData = None
    statusType = "SUCCESS"
    statusCode = 200
    statusMessage = 'The Token is valid.'
    json_return = json_return_constructor(apiStatus, requestData, statusType, statusCode, statusMessage)
    
    logging_report(f'{statusCode} | {statusMessage}', 'INFO', api_name)
    return json_return
####################### end of Token check module ##########################

############################################################################
######################### Query execute module #############################
def query_execute(query_command, database_name, allrows, api_name):
    current_datetime = datetime.now().strftime("%Y"+"-"+"%m"+"-"+"%d"+" "+"%H"+":"+"%M"+":"+"%S")

    if database_name.startswith('streaming'):
        connection = psycopg2.connect(user = "postgres",
                                            password = str(dbrtsrvpassword),
                                            host = str(dbrtsrvendpoint),
                                            port = "5432",
                                            database = str(database_name))
    else:
        connection = psycopg2.connect(user = "postgres",
                                        password = str(dbsrvpassword),
                                        host = str(dbsrvendpoint),
                                        port = "5432",
                                        database = str(database_name))

    cursor = connection.cursor()
    
    try:
        cursor.execute(query_command)

    except Exception as error:
        query_error = True
        error = str(error).replace("\n", " ")
        error = str(error).replace("\\", "")
        error = str(error).replace("^", "")
        error = str(error).strip()
     
        errMessage = f'Database query error: {error}'
        print(errMessage)
        apiStatus = True
        requestData = None
        statusType = "ERROR"
        statusCode = 500001
        statusMessage = errMessage
        json_return = json_return_constructor(apiStatus, requestData, statusType,
                        statusCode, statusMessage)
   
        logging_report(f'{database_name} | {query_command} | {statusMessage}', 'ERROR', api_name)

        return json_return, query_error

    query_error = False
    if allrows:
        return cursor.fetchall(), query_error
    return cursor.fetchone(), query_error
#################### end of query execute module ##########################

############################################################################
####################### strings modification module ########################
def modify_strings(*args):
    """
    Function that modifies strings,
    in the following way:
    - replaces dot (.) with underscore (_).
    - replaces space ( ) with underscore (_).
    - converts all to lowercase.
    Here the inputs are:
    - any number of string arguments
    Here the outputs are:
    - A list with the modified strings
    """
    modified_args = []
    for arg in args:
        modified_arg = arg.replace(' ', '_').replace('.', '_').lower()
        modified_args.append(modified_arg)
    return modified_args
################### end of strings modification module #####################
