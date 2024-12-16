import json
import fcntl
from pathlib import Path
import time
import logging
from typing import Dict, Any

class SessionManager:
    def __init__(self, session_file: str = '.session.json'):
        self.session_file = Path(session_file)
        self.session_data: Dict[str, Any] = {}

    def load_session(self, max_retries=3, delay=0.1):
        for attempt in range(max_retries):
            if self.session_file.exists():
                with self.session_file.open('r') as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    try:
                        self.session_data = json.load(f)
                        return
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON (attempt {attempt+1}): {e}")
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
            time.sleep(delay)
        
        logging.warning("Failed to load session after multiple attempts. Initializing empty session.")
        self.session_data = {}

    def save_session(self):
        with self.session_file.open('w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(self.session_data, f, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def set_variable(self, key: str, value: str):
        self.load_session()
        self.session_data[key] = value
        self.save_session()
        logging.debug(f'Set {key}: {value}')

    def get_variable(self, key: str, default: Any = None) -> Any:
        self.load_session()
        return self.session_data.get(key, default)