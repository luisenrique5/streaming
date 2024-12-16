import os
from datetime import datetime
from flask import Flask, request
from src.routes.streaming_routes import streaming_Blueprint
from utils_backend import ENVIRONMENT

def create_app():
    app = Flask(__name__)
    app.register_blueprint(streaming_Blueprint)
    
    ############################################################################
    ##################### requests logger modules ##############################
    @app.before_request
    def before_request_func():
        if ENVIRONMENT == 'development':
            logpath = '/home/ubuntu/log/streaming/requests'
        else:
            logpath = '/mnt/log/streaming/requests'

        urlstring = request.url
        try:
            api_name = urlstring.split("enovate.app/")[1].split("/")[0]
        except:
            api_name = "URL_ERROR"
        save_path = f'{logpath}/{api_name}/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.chown(save_path, 1000, 1000)

        log_date = datetime.now().strftime("%Y%m%d")
        source_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        with open(f'{save_path}{api_name}_{log_date}.log', 'a', encoding="utf-8") as log:
            log.write('%s %s REQUEST %s %s \n' % (source_ip, datetime.now(),
                                                  request.method, request.url))
        os.chown(f'{save_path}{api_name}_{log_date}.log', 1000, 1000)

    @app.after_request
    def after_request_func(response):
        if ENVIRONMENT == 'development':
            logpath = '/home/ubuntu/log/streaming/requests'
        else:
            logpath = '/mnt/log/streaming/requests'

        urlstring = request.url
        try:
            api_name = urlstring.split("enovate.app/")[1].split("/")[0]
        except:
            api_name = "URL_ERROR"
        save_path = f'{logpath}/{api_name}/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.chown(save_path, 1000, 1000)

        log_date = datetime.now().strftime("%Y%m%d")
        source_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        with open(f'{save_path}{api_name}_{log_date}.log', 'a', encoding="utf-8") as log:
            log.write('%s %s RESPONSE %s %s %s \n' % (source_ip, datetime.now(),
                                                      request.method, request.url, response.status))
        os.chown(f'{save_path}{api_name}_{log_date}.log', 1000, 1000)
        return response

    return app

app = create_app()

if __name__ == '__main__':
    app.run()