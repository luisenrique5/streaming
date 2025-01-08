import json
import fcntl
import time
import logging
from pathlib import Path
from typing import Dict, Any

class SessionManager:
    """
    Maneja la persistencia de variables en un archivo JSON con bloqueos de fcntl para
    evitar corrupción al escribir. Introduce caching en memoria para reducir el overhead
    de leer el archivo cada vez.
    """

    def __init__(self, session_file: str = ".session.json"):
        self.session_file = Path(session_file)
        self.session_data: Dict[str, Any] = {}
        self._last_mtime = 0.0  # Momento de la última modificación que cargamos

        # Carga inicial (si existe) para no empezar vacío
        # (opcional: si prefieres que arranque vacío y cargue bajo demanda, puedes omitirlo)
        self._load_session(force=True)

    def _file_changed(self) -> bool:
        """
        Verifica si el archivo en disco cambió (mtime mayor a lo que tenemos registrado).
        """
        if not self.session_file.exists():
            return False
        mtime = self.session_file.stat().st_mtime
        return mtime > self._last_mtime

    def _load_session(self, force=False, max_retries=1, delay=0.1):
        """
        Carga la sesión desde disco si `force=True` o si el archivo cambió respecto a
        lo que tenemos en memoria.
        - max_retries y delay se pueden ajustar para casos de colisión de locks.
        """
        if not force and not self._file_changed():
            return  # No hacemos nada si no cambió el archivo

        for attempt in range(max_retries):
            if self.session_file.exists():
                try:
                    with self.session_file.open("r") as f:
                        # Bloqueo compartido (lectura)
                        fcntl.flock(f, fcntl.LOCK_SH)
                        try:
                            data_on_disk = json.load(f)
                            self.session_data = data_on_disk
                            # Actualizamos el timestamp local
                            self._last_mtime = self.session_file.stat().st_mtime
                            return
                        except json.JSONDecodeError as e:
                            logging.error(
                                f"Error decoding JSON (attempt {attempt+1}): {e}"
                            )
                        finally:
                            fcntl.flock(f, fcntl.LOCK_UN)

                except OSError as e:
                    logging.error(
                        f"Error reading session file (attempt {attempt+1}): {e}"
                    )

            time.sleep(delay)

        logging.warning(
            "Failed to load session after multiple attempts. "
            "Initializing empty session in memory."
        )
        self.session_data = {}

    def _save_session(self):
        """
        Guarda la sesión en disco bajo lock exclusivo (LOCK_EX).
        Es un método interno que asume que self.session_data ya está en memoria.
        """
        # Para guardar atómicamente, a veces se escribe en un archivo temporal y luego
        # se hace un rename. Aquí, por simplicidad, lo hacemos directo.
        with self.session_file.open("w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(self.session_data, f, indent=2)
                # Luego de escribir, actualizamos _last_mtime:
                self._last_mtime = self.session_file.stat().st_mtime
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def set_variable(self, key: str, value: Any):
        """
        Ajusta la variable `key` a `value` en la sesión y guarda inmediatamente.
        """
        # Primero cargamos si el archivo cambió
        self._load_session()
        # Actualizamos en memoria
        self.session_data[key] = value
        # Guardamos en disco
        self._save_session()
        logging.debug(f"Set {key}: {value}")

    def get_variable(self, key: str, default: Any = None) -> Any:
        """
        Retorna el valor de `key` si existe, o `default` en caso contrario.
        Hace un load si el archivo cambió.
        """
        self._load_session()
        return self.session_data.get(key, default)
