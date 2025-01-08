from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Tuple

class StreamingEnvironment:
    """Manages the streaming environment setup and configuration."""

    BASE_PATH_DEV = "/home/ubuntu/log"
    BASE_PATH_PROD = "/mnt/log"

    # (1) Variable de clase para cachear directorios ya creados, si lo deseas.
    #     La clave del diccionario podría ser (client, project, stream, username, scenario, environment).
    _directories_created_cache = {}

    def __init__(self, client: str, project: str, stream: str,
                 username: str, scenario: str, environment: str):
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.environment = environment
        # Cache key
        self._cache_key = (client, project, stream, username, scenario, environment)

        # Ejemplo: la fecha-hora con precisión de hora
        self.current_date = datetime.now().strftime("%Y%m%d-%H")

        # Validaciones básicas
        if not all([self.client, self.project, self.stream, self.username, self.scenario]):
            raise ValueError("Missing required fields in StreamingEnvironment configuration")

    @property
    def base_path(self) -> str:
        """Determines the base path based on environment."""
        return self.BASE_PATH_DEV if self.environment == "development" else self.BASE_PATH_PROD

    def _get_directory_structure(self) -> list[Path]:
        """
        Returns the list of directories that need to be created.
        Ejem: /home/ubuntu/log/drillbi_streaming/traces/<client>/<project>/<stream>/<username>/<scenario>
        """
        base = Path(self.base_path) / "drillbi_streaming" / "traces"
        return [
            base / self.client,
            base / self.client / self.project,
            base / self.client / self.project / self.stream,
            base / self.client / self.project / self.stream / self.username,
            base / self.client / self.project / self.stream / self.username / self.scenario
        ]

    def _create_directories_if_needed(self) -> Tuple[bool, Optional[str]]:
        """
        Creates the directory structure for logs, only if not created before (via cache).
        Retorna (True, None) si se crearon o ya existían, (False, mensaje) si ocurrió error.
        """
        if self._directories_created_cache.get(self._cache_key):
            # Ya se crearon antes, no hacemos nada
            return True, None

        try:
            for directory in self._get_directory_structure():
                directory.mkdir(parents=True, exist_ok=True)
            # Guardamos en cache que estos directorios ya se crearon
            self._directories_created_cache[self._cache_key] = True
            return True, None
        except Exception as e:
            error_msg = f"Failed to create directory structure: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

    def _get_log_file_path(self) -> Path:
        """
        Genera la ruta completa del archivo de log, e.g.:
        <...>/<scenario>/getstreamingoutputrt_<stream>_YYYYMMDD-H.log
        """
        file_dir = self._get_directory_structure()[-1]
        file_suffix = f"{self.stream}_{self.current_date}"
        return file_dir / f"getstreamingoutputrt_{file_suffix}.log"

    def setup_environment(self) -> str:
        """
        Configura el entorno: crea directorios (solo si no existen)
        y devuelve la ruta completa del log file.

        IMPORTANTE: Si tu aplicación ya configura logging a nivel global,
        podrías remover la llamada a logging.basicConfig de aquí.
        """
        success, error = self._create_directories_if_needed()
        if not success:
            raise RuntimeError(f"Failed to setup environment: {error}")

        log_file_path = self._get_log_file_path()

        # (2) Recomendado: Configurar logging en un solo lugar de la aplicación
        #    pero si necesitas hacerlo aquí (ej. en modo local):
        logging.basicConfig(
            filename=str(log_file_path),
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        return str(log_file_path)
