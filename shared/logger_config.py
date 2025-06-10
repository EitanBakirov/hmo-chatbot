import logging
import json
from datetime import datetime
from typing import Dict, Any

class EnhancedLogger:
    def __init__(self) -> None:
        """Initialize the logger with file and console handlers"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, Any] = {}

    def info(self, message: str, **metrics) -> None:
        """Regular info logging with optional metrics"""
        if metrics:
            self.log_with_metrics(message, "info", **metrics)
        else:
            self.logger.info(message)

    def error(self, message: str, **metrics) -> None:
        """Error logging with optional metrics"""
        if metrics:
            self.log_with_metrics(message, "error", **metrics)
        else:
            self.logger.error(message)

    def log_with_metrics(self, message: str, level: str, **metrics) -> None:
        """Log message with associated metrics"""
        log_data = {
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }
        self.metrics.update(metrics)
        if level == "error":
            self.logger.error(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))

logger = EnhancedLogger()