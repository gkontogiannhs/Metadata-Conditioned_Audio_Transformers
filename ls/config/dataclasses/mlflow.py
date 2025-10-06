from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Any

@dataclass
class MLflowConfig:
    tracking_uri: str = ""
    tracking_username: str = ""
    tracking_password: str = ""
    experiment_name: str = ""