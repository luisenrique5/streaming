from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

class StreamingEnvironment:
    """Manages the streaming environment setup and configuration."""
    
    BASE_PATH_DEV = "/home/ubuntu/log"
    BASE_PATH_PROD = "/mnt/log"

    def __init__(self, client: str, project: str, stream: str, 
                 username: str, scenario: str, environment: str):
        self.client = client
        self.project = project
        self.stream = stream
        self.username = username
        self.scenario = scenario
        self.environment = environment
        self.current_date = datetime.now().strftime("%Y%m%d-%H")

    @property
    def base_path(self) -> str:
        """Determines the base path based on environment."""
        return self.BASE_PATH_DEV if self.environment == "development" else self.BASE_PATH_PROD

    def _get_directory_structure(self) -> list[Path]:
        """Returns the list of directories that need to be created."""
        base = Path(self.base_path) / "drillbi_streaming" / "traces"
        return [
            base / self.client,
            base / self.client / self.project,
            base / self.client / self.project / self.stream,
            base / self.client / self.project / self.stream / self.username,
            base / self.client / self.project / self.stream / self.username / self.scenario
        ]

    def _create_directories(self) -> Tuple[bool, Optional[str]]:
        """Creates the directory structure for logs."""
        try:
            for directory in self._get_directory_structure():
                directory.mkdir(parents=True, exist_ok=True)
            return True, None
        except Exception as e:
            error_msg = f"Failed to create directory structure: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

    def _get_log_file_path(self) -> Path:
        """Generates the complete log file path."""
        file_dir = self._get_directory_structure()[-1]
        file_suffix = f"{self.stream}_{self.current_date}"
        return file_dir / f"getstreamingoutputrt_{file_suffix}.log"

    def setup_environment(self) -> str:
        """
        Configures the environment and returns the log file path.
        This is the main method called from services.
        """
        # Validate required fields
        if not all([self.client, self.project, self.stream, 
                   self.username, self.scenario]):
            raise ValueError("Missing required fields in StreamingEnvironment configuration")

        # Create directories
        success, error = self._create_directories()
        if not success:
            raise RuntimeError(f"Failed to setup environment: {error}")

        # Get and return log file path
        log_file_path = self._get_log_file_path()
        
        # Configure basic logging
        logging.basicConfig(
            filename=str(log_file_path),
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        return str(log_file_path)