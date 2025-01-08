import os
from datetime import datetime
from flask import Flask, request
from src.routes.streaming_routes import streaming_Blueprint
from utils_backend import ENVIRONMENT

def create_app():
    app = Flask(__name__)
    app.register_blueprint(streaming_Blueprint)

    # -------------------------------------------------------------------------
    # 1. Funciones helper
    # -------------------------------------------------------------------------
    def get_log_base_path():
        """
        Devuelve la ruta base para los logs, dependiendo del ambiente.
        """
        if ENVIRONMENT == 'development':
            return '/home/ubuntu/log/streaming/requests'
        else:
            return '/mnt/log/streaming/requests'

    def extract_api_name(url):
        """
        Intenta extraer el nombre del 'api_name' de la URL.
        En caso de error, retorna 'URL_ERROR'.
        """
        try:
            # Ajusta 'enovate.app/' si necesitas otro dominio base
            return url.split("enovate.app/")[1].split("/")[0]
        except (IndexError, AttributeError):
            return "URL_ERROR"

    def get_or_create_logfile(api_name):
        """
        Retorna la ruta completa del archivo de log. 
        Crea la carpeta si no existe. 
        Opcionalmente, cambia permisos para usuario y grupo 1000, si fuera requerido.
        """
        base_path = get_log_base_path()
        save_path = os.path.join(base_path, api_name)

        # Crea el directorio si no existe (con exist_ok=True evitas error si ya existe)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            # Si es estrictamente necesario, ajustar dueño/carpetas acá
            os.chown(save_path, 1000, 1000)

        # Nombre de archivo con fecha YYYYMMDD
        log_date = datetime.now().strftime("%Y%m%d")
        log_filename = f"{api_name}_{log_date}.log"
        logfile_path = os.path.join(save_path, log_filename)
        return logfile_path

    def log_to_file(logfile, message):
        """
        Abre el archivo en modo append y escribe el 'message'.
        Cambia permisos (chown) solo si lo necesitas cada vez (puede ser costoso).
        """
        with open(logfile, 'a', encoding="utf-8") as log:
            log.write(message + "\n")

        # Si lo requieres, haz chown siempre; si no, podrías hacerlo solo si el archivo se crea
        os.chown(logfile, 1000, 1000)

    # -------------------------------------------------------------------------
    # 2. before_request: Registra el REQUEST
    # -------------------------------------------------------------------------
    @app.before_request
    def before_request_func():
        api_name = extract_api_name(request.url)
        logfile_path = get_or_create_logfile(api_name)

        source_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        method = request.method
        url_ = request.url

        # Mensaje REQUEST
        message = f"{source_ip} {now_str} REQUEST {method} {url_}"
        log_to_file(logfile_path, message)

    # -------------------------------------------------------------------------
    # 3. after_request: Registra la RESPONSE
    # -------------------------------------------------------------------------
    @app.after_request
    def after_request_func(response):
        api_name = extract_api_name(request.url)
        logfile_path = get_or_create_logfile(api_name)

        source_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        method = request.method
        url_ = request.url
        status = response.status

        # Mensaje RESPONSE
        message = f"{source_ip} {now_str} RESPONSE {method} {url_} {status}"
        log_to_file(logfile_path, message)

        return response

    return app

app = create_app()

if __name__ == '__main__':
    # Puedes habilitar debug=True si lo requieres en dev
    app.run(debug=True)
