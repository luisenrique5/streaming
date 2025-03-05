## packages for local linux system access
import os

## packages for data treatment
import json
import pandas as pd


## packages for logging module
from datetime import datetime
import logging

import psycopg2

from os.path import join, dirname
from dotenv import load_dotenv

import google.cloud.storage as storage
from google.oauth2.service_account import Credentials

## set environmet file "./env/.env"
dotenv_path = join(dirname(__file__), './env/.env')
load_dotenv(dotenv_path)

### get USERNAME value from environment variables
dbsrvendpoint = os.getenv('DBSRVDRILLBIEP')
dbsrvpassword = os.getenv('DBSRVSRVDRILLBIPASSW')
dbrtsrvendpoint = os.getenv('DBRTSRVENDPOINT')
dbrtsrvpassword = os.getenv('DBRTSRVPASSWORD')
APITOKEN = os.getenv('APITOKEN5')

## set environmet file "./.datavar"
datavar_path = join(dirname(__file__), '.datavar')
load_dotenv(datavar_path)

### get CLIENT, PROJECT, STREAM, USERNAME and SCENARIO values from environment variables
client = os.getenv('CLIENT')
project = os.getenv('PROJECT')
stream = os.getenv('STREAM')
username = os.getenv('USERNAME')
scenario = os.getenv('SCENARIO')


##### Opening config JSON file
with open('./config/config.json', 'r', encoding="utf-8") as file:
    config_json = file.read()

##### returns JSON object as a dictionary
config_dict = json.loads(config_json)

ENVIRONMENT = config_dict["ENVIRONMENT"]
LOGGINGLEVEL = config_dict["LOGGINGLEVEL"]
REMOTEDOCKERIMAGE = config_dict["REMOTEDOCKERIMAGE"]
KEY_PATH = config_dict["KEYPATH"]

credentials = Credentials.from_service_account_file(
    KEY_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

# create input folder and save folder
INPUT_FOLDER = f"./input_data/{client}/{project}/{stream}/{username}/{scenario}/"
if not os.path.exists(INPUT_FOLDER):
    os.makedirs(INPUT_FOLDER)
SAVE_FOLDER = f"./output_data/{client}/{project}/{stream}/{username}/{scenario}/"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

#########################################################################################
################ Set CURRENT_WELL_NAME and WELLS_SELECT_NAME parameters #################
filters_json_path = f'{INPUT_FOLDER}inputs_for_rt.json'
if os.path.exists(filters_json_path):
    ##### Set CURRENT_WELL_NAME and WELLS_SELECT_NAME parameters input by user #####
    # Opening JSON file
    filters_file = open(filters_json_path,)
    
    # returns JSON object as a dictionary
    filters_data = json.load(filters_file)
    
    # Iterating through the json list
    CURRENT_WELL_NAME = ''
    WELLS_SELECT_NAME = []

    CURRENT_WELL_NAME = str(filters_data['CURRENT_WELL_NAME'])

    for i in filters_data['WELLS_SELECT_NAME']:
        WELLS_SELECT_NAME.append(i)


#### set current date and time
current_datetime = datetime.now().strftime("%Y"+"-"+"%m"+"-"+"%d"+" "+"%H"+":"+"%M"+":"+"%S")
##############################

#*************************************************************************************************************#
#************************** Input functions for backend arquitecture deployment ******************************#
#*************************************************************************************************************#

#################### addition of level 'TRACE' to the logging module ####################
##### Define the new registry level and its value
TRACE = 5  # Chose an integer value that is less than logging.DEBUG (10)

##### Add the new level to the logging module
logging.addLevelName(TRACE, "TRACE")

##### Create a function to enable logging at the new level
def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kws)

##### Associate the function with the logger
logging.Logger.trace = trace
#########################################################################################

############################################################################
####################### logging_report module ##############################
def logging_report(writing: str, level: str, api_name: str):
    """
    Function to generate and write events logs reports in the
    corresponding location and log file.
    """
    ##### set up log folder
    if ENVIRONMENT == 'development':
        logpath = '/home/ubuntu/log/drillbi_streaming/events'
    else:
        logpath = '/mnt/log/drillbi_streaming/events'

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

    if level == 'TRACE':
        logger.trace("%s", writing)
    elif level == 'DEBUG':
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
def json_return_constructor(apiStatus: bool, requestData: dict, statusType: str,
                            statusCode: int, statusMessage: str):
    """
    Function that construct a JSON text with status info,
    to be used as a return for a specific event.
    """
    dict_return = {
        'apiStatus' : apiStatus,
        'data' : requestData,
        'statusType' : statusType,
        'statusCode' : statusCode,
        'statusMessage' : statusMessage,    
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
    
    if token != APITOKEN:
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
####################### Query execute module old ###########################
def query_execute_old(query_command: str, database_name: str,
                allrows: bool, api_name: str):
    """
    Function that executes a query to a specific database,
    and returns the query_response and the error status.
    Here the inputs are:
    - query_command: This is the actual query string to the database.
    - database_name: This is the name of the database.
    - allrows: True if fetchall, False if fetchone.
    - api_name: This is the name of the API that does the query.
    Here the outputs are:
    - json_return (array or str): The one or two dimensional array with the query response.
    - query_error (bool): True if the query is not successfull, False if successful.
    """
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
        statusType = "FAILURE"
        statusCode = 500001
        statusMessage = errMessage
        json_return = json_return_constructor(apiStatus, requestData, statusType,
                        statusCode, statusMessage)

        logging_report(f'{database_name} | {query_command} | {statusMessage}', 'ERROR', api_name)

        return json_return, query_error


    query_error = False
    logging_report(
        f'{database_name} | {query_command} | Query executed successfully', 'INFO', api_name)

    if allrows:
        return cursor.fetchall(), query_error
    return cursor.fetchone(), query_error
################### end of query execute module old ########################

############################################################################
######################### Query execute module #############################
def query_execute(query_command: str, database_name: str, allrows: bool, api_name: str):
    """
    Function that executes a query to a specific database,
    and returns the query_response and the error status.
    Here the inputs are:
    - query_command: This is the actual query string to the database.
    - database_name: This is the name of the database.
    - allrows: True if fetchall, False if fetchone.
    - api_name: This is the name of the API that does the query.
    Here the outputs are:
    - json_return (array or str): The one or two dimensional array with the query response.
    - query_error (bool): True if the query is not successful, False if successful.
    """
    query_error = False
    query_return = None

    try:
        if database_name.startswith('streaming'):
            connection = psycopg2.connect(user="postgres",
                                          password=str(dbrtsrvpassword),
                                          host=str(dbrtsrvendpoint),
                                          port="5432",
                                          database=str(database_name))
        else:
            connection = psycopg2.connect(user="postgres",
                                          password=str(dbsrvpassword),
                                          host=str(dbsrvendpoint),
                                          port="5432",
                                          database=str(database_name))

        cursor = connection.cursor()
        cursor.execute(query_command)

        if allrows:
            query_return = cursor.fetchall()
        else:
            query_return = cursor.fetchone()

        logging_report(f'EXECUTED | 200001 | {database_name} | {query_command} | Query executed successfully', 'INFO', api_name)

    except Exception as error:
        query_error = True
        error_message = str(error).strip()

        apiStatus = True
        requestData = None
        statusType = "FAILURE"
        statusCode = 500001
        statusMessage = f'Database query error: {error_message}'

        query_return = json_return_constructor(apiStatus, requestData, statusType, statusCode, statusMessage)

        logging_report(f'{statusType} | {statusCode} | {database_name} | {query_command} | {statusMessage}', 'ERROR', api_name)

    finally:
        # Close the database connection
        if connection:
            cursor.close()
            connection.close()

    return query_return, query_error
##################### end of query execute module ##########################

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

#################################### FUNCTION: Download File ######################################
def download_file(bucket_name, object_name, file_name, api_name):
    """
    Downloads a blob from the GCP bucket to a specific local location.
    - bucket_name = The GCP bucket name
    - object_name = The path and name of the blob object "remote/path/to/file"
    - file_name = The path to which the file should be downloaded "local/path/to/file"
    - api_name = The name of the api that is calling this function
    """

    ######################################## Get Storage client #######################################
    try:
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(bucket_name)
        logging_report(f'Connected successfully to GCP bucket {bucket_name}', 'INFO', api_name)
    except Exception as err:
        err_message = f"Coudn't connect to bucket: {err}"
        status_code = 400101
        status_message = err_message
        logging_report(f'{status_code} | {status_message}', 'ERROR', api_name)
        raise
    ###################################################################################################

    blob = bucket.blob(object_name)
    blob_exists = blob.exists()
    if not blob_exists:
        logging_report(f'404001 | Blob file "{object_name}" does not exist.', 'ERROR', api_name)
        return False

    try:
        blob.download_to_filename(file_name)
        logging_report(f'Blob file "{object_name}" downloaded successfully to server', 'INFO', api_name)
    except Exception as err:
        logging_report(f'400105 | Could not download the blob file "{object_name}" to server: {err}', 'ERROR', api_name)
        raise
    return True
###################################################################################################

################################### FUNCTION: xlsx_to_csvs ###################################
def xlsx_to_csvs(path, filename, api_name):
    """
    Function that convert the sheets of a XLSX file into CSV files
    """
    # Read the Excel file
    xls = pd.ExcelFile(f"{path}/{filename}")

    # Iterate over each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame
        df = xls.parse(sheet_name)
        
        # Generate the CSV filename
        csv_filename = f"{sheet_name}.csv"
        
        # Save the DataFrame as a CSV file
        df.to_csv(f"{path}/{csv_filename}", index=False)
        logging_report(f"Saved {path}/{csv_filename} successfully!", 'INFO', api_name)
################################ end of FUNCTION: xlsx_to_csvs ###############################

############################### FUNCTION: get_unprocessed_data ###############################
def get_unprocessed_data(clientp, projectp, streamp, number_of_rows, api_name):
    # Construct the database name
    db_name = f"streaming_drillbi_{clientp}_{projectp}_db"

    # Construct the table name
    table_name = f"streaming_tbd_{streamp}"

    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            database=db_name,
            user="postgres",
            password=dbrtsrvpassword,
            host=dbrtsrvendpoint,
            port="5432"
        )

        # Create a cursor object
        cur = conn.cursor()

        # Execute the SQL query to retrieve the data
        query = f"""
            SELECT measured_depth, bit_depth, rop
            FROM {table_name}
            ORDER BY id DESC
            LIMIT {number_of_rows};
        """
        cur.execute(query)

        # Fetch the data and store it in a list of dictionaries
        data = [
            {
                "measured_depth": row[0],
                "bit_depth": row[1],
                "rop": row[2]
            }
            for row in cur.fetchall()
        ]

        logging_report(f'EXECUTED | 200002 | {clientp} | {projectp} | {streamp} | {number_of_rows} unprocessed data retrieved successfully', 'INFO', api_name)
        logging_report(f'EXECUTED | 200012 | {clientp} | {projectp} | {streamp} | {number_of_rows} unprocessed data:\n{data}', 'TRACE', api_name)

        # Close the cursor and connection
        cur.close()
        conn.close()

        return data

    except (Exception, psycopg2.Error) as error:
        print("Error connecting to PostgreSQL database:", error)
        logging_report(f"FAILURE | 500012 | {clientp} | {projectp} | {streamp} | Error connecting to PostgreSQL database: {error}", 'ERROR', api_name)
        raise
############################ end of FUNCTION: get_unprocessed_data ###########################

############################### FUNCTION: get_builduprate_data ###############################
def get_builduprate_data(clientp, projectp, streamp, api_name):
    # Construct the database name
    db_name = f"streaming_drillbi_{clientp}_{projectp}_db"

    # Construct the table name
    table_name = f"streaming_os_{streamp}"

    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            database=db_name,
            user="postgres",
            password=dbrtsrvpassword,
            host=dbrtsrvendpoint,
            port="5432"
        )

        # Create a cursor object
        cur = conn.cursor()

        # Execute the SQL query to retrieve the data
        query = f"""
            SELECT measured_depth, incl, azm
            FROM {table_name}
            ORDER BY id DESC;
        """
        cur.execute(query)

        # Fetch the data and store it in a list of dictionaries
        data = [
            {
                "measured_depth": row[0],
                "incl": row[1],
                "azm": row[2]
            }
            for row in cur.fetchall()
        ]

        logging_report(f'EXECUTED | 200006 | {clientp} | {projectp} | {streamp} | Build up rate data retrieved successfully', 'INFO', api_name)
        logging_report(f'EXECUTED | 200016 | {clientp} | {projectp} | {streamp} | Build up rate data:\n{data}', 'TRACE', api_name)

        # Close the cursor and connection
        cur.close()
        conn.close()

        return data

    except (Exception, psycopg2.Error) as error:
        print("Error connecting to PostgreSQL database:", error)
        logging_report(f"FAILURE | 500016 | {clientp} | {projectp} | {streamp} | Error connecting to PostgreSQL database: {error}", 'ERROR', api_name)
        raise
############################ end of FUNCTION: get_builduprate_data ###########################

################################ FUNCTION: get_torqdrag_data #################################
def get_torqdrag_data(clientp, projectp, streamp, number_of_rows, api_name):
    # Construct the database name
    db_name = f"streaming_drillbi_{clientp}_{projectp}_db"

    # Construct the table name
    table_name = f"streaming_tbd_{streamp}"

    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            database=db_name,
            user="postgres",
            password=dbrtsrvpassword,
            host=dbrtsrvendpoint,
            port="5432"
        )

        # Create a cursor object
        cur = conn.cursor()

        # Execute the SQL query to retrieve the data
        query = f"""
            SELECT measured_depth, bit_depth, torque, hook_load, rop
            FROM {table_name}
            ORDER BY id DESC
            LIMIT {number_of_rows};
        """
        cur.execute(query)

        # Fetch the data and store it in a list of dictionaries
        data = [
            {
                "measured_depth": row[0],
                "bit_depth": row[1],
                "torque": row[2],
                "hook_load": row[3],
                "rop": row[4]
            }
            for row in cur.fetchall()
        ]

        logging_report(f'EXECUTED | 200003 | {clientp} | {projectp} | {streamp} | {number_of_rows} torqdrag data retrieved successfully', 'INFO', api_name)
        logging_report(f'EXECUTED | 200013 | {clientp} | {projectp} | {streamp} | {number_of_rows} torqdrag data:\n{data}', 'TRACE', api_name)

        # Close the cursor and connection
        cur.close()
        conn.close()

        return data

    except (Exception, psycopg2.Error) as error:
        print("Error connecting to PostgreSQL database:", error)
        logging_report(f"FAILURE | 500013 | {clientp} | {projectp} | {streamp} | Error connecting to PostgreSQL database: {error}", 'ERROR', api_name)
        raise
############################# end of FUNCTION: get_torqdrag_data #############################

############################# FUNCTION: get_torqdrag_aggregated ##############################
def get_torqdrag_aggregated(clientp, projectp, streamp, filter_value, api_name):
    # Construct the database name
    db_name = f"streaming_drillbi_{clientp}_{projectp}_db"

    # Construct the table name
    table_name = f"streaming_tbd_{streamp}"

    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            database=db_name,
            user="postgres",
            password=dbrtsrvpassword,
            host=dbrtsrvendpoint,
            port="5432"
        )

        # Create a cursor object
        cur = conn.cursor()

        # Execute the SQL query to retrieve the data
        query = f"""
            SELECT measured_depth, bit_depth, torque, hook_load, rop
            FROM (
                SELECT *, ROW_NUMBER() OVER (ORDER BY id DESC) as row_num
                FROM {table_name}
            ) subquery
            WHERE row_num % {filter_value} = 0
            ORDER BY row_num;
        """

        ##### Uncomment the following if a dynamic step with a total number of rows of 5000 is wanted #####
        # # First, get the total number of rows
        # cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        # total_rows = cur.fetchone()[0]

        # # Determine the number of rows to fetch and the step size
        # if total_rows <= 5000:
        #     rows_to_fetch = total_rows
        #     step = 1
        # else:
        #     rows_to_fetch = 5000
        #     step = total_rows // 5000

        # # Execute the SQL query to retrieve the data
        # query = f"""
        #     SELECT measured_depth, bit_depth, torque, hook_load, rop
        #     FROM (
        #         SELECT *, ROW_NUMBER() OVER (ORDER BY id DESC) as row_num
        #         FROM {table_name}
        #     ) subquery
        #     WHERE row_num % {step} = 0
        #     ORDER BY row_num
        #     LIMIT {rows_to_fetch};
        # """
        ###################################################################################################

        cur.execute(query)

        # Fetch the data and store it in a list of dictionaries
        data = [
            {
                "measured_depth": row[0],
                "bit_depth": row[1],
                "torque": row[2],
                "hook_load": row[3],
                "rop": row[4]
            }
            for row in cur.fetchall()
        ]

        logging_report(f'EXECUTED | 200004 | {clientp} | {projectp} | {streamp} | torqdrag aggregated data retrieved successfully', 'INFO', api_name)
        logging_report(f'EXECUTED | 200014 | {clientp} | {projectp} | {streamp} | torqdrag aggregated data:\n{data}', 'TRACE', api_name)

        # Close the cursor and connection
        cur.close()
        conn.close()

        return data

    except (Exception, psycopg2.Error) as error:
        print("Error connecting to PostgreSQL database:", error)
        logging_report(f"FAILURE | 500014 | {clientp} | {projectp} | {streamp} | Error connecting to PostgreSQL database: {error}", 'ERROR', api_name)
        raise
########################## end of FUNCTION: get_torqdrag_aggregated ##########################

############################## FUNCTION: get_data_at_interval ################################
def get_data_at_interval(clientp, projectp, streamp, usernamep, scenariop, database_name, streaming_db_table, api_name, interval_seconds=10):
    # Database connection parameters
    db_params = {
        "dbname": str(database_name),
        "user": "postgres",
        "password": str(dbrtsrvpassword),
        "host": str(dbrtsrvendpoint),
        "port": "5432"
    }

    try:
        # Connect to the database
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()

        # query = f"""
        # WITH base_data AS (
        #     SELECT id, datetime::timestamp AS datetime, measured_depth, bit_depth, block_height, rop, 
        #            wob, hook_load, flow_rate, spp, torque, surface_rpm, motor_rpm, bit_rpm,
        #            ROW_NUMBER() OVER (ORDER BY datetime::timestamp) AS row_num
        #     FROM {streaming_db_table}
        # ),
        # min_datetime AS (
        #     SELECT MIN(datetime) AS min_dt FROM base_data
        # )
        # SELECT id, datetime, measured_depth, bit_depth, block_height, rop, 
        #        wob, hook_load, flow_rate, spp, torque, surface_rpm, motor_rpm, bit_rpm
        # FROM base_data, min_datetime
        # WHERE EXTRACT(EPOCH FROM (datetime - min_dt)) % {interval_seconds} < 1
        # ORDER BY datetime;
        # """

        query = f"""
        WITH base_data AS (
            SELECT id, datetime::timestamp AS datetime, measured_depth, bit_depth, block_height, rop, 
                   wob, hook_load, flow_rate, spp, torque, surface_rpm, motor_rpm, bit_rpm,
                   ROW_NUMBER() OVER (ORDER BY datetime::timestamp) AS row_num,
                   LAG(datetime::timestamp) OVER (ORDER BY datetime::timestamp) AS prev_datetime
            FROM {streaming_db_table}
        ),
        time_diff AS (
            SELECT *, 
                   EXTRACT(EPOCH FROM (datetime - prev_datetime)) AS seconds_diff
            FROM base_data
        ),
        data_frequency AS (
            SELECT MODE() WITHIN GROUP (ORDER BY seconds_diff) AS mode_diff
            FROM time_diff
            WHERE seconds_diff IS NOT NULL
        )
        SELECT id, datetime, measured_depth, bit_depth, block_height, rop, 
               wob, hook_load, flow_rate, spp, torque, surface_rpm, motor_rpm, bit_rpm
        FROM time_diff, data_frequency
        WHERE 
            CASE 
                WHEN mode_diff <= {interval_seconds} THEN 
                    (row_num - 1)::bigint % CEIL({interval_seconds}::float / NULLIF(mode_diff, 0))::bigint = 0
                ELSE 
                    seconds_diff >= {interval_seconds} OR row_num = 1
            END
        ORDER BY datetime;
        """


        # Execute the query
        cur.execute(query)
        # Fetch all results
        results = cur.fetchall()

        # Close cursor and connection
        cur.close()
        conn.close()

        logging_report(f'EXECUTED | 200005 | {clientp} | {projectp} | {streamp} | {usernamep} | {scenariop} | {streaming_db_table} data retrieved successfully', 'INFO', api_name)

        return results

    except (Exception, psycopg2.Error) as error:
        logging_report(f"FAILURE | 500015 | {clientp} | {projectp} | {streamp} | {usernamep} | {scenariop} | Error connecting to PostgreSQL database: {error}", 'ERROR', api_name)
        raise
########################### end of FUNCTION: get_data_at_interval ############################
