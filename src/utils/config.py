"""Configuration loader"""

import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str = None) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file (default: config/settings.yaml)
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Try to find config from current directory or parent
        current = Path.cwd()
        config_path = current / "config" / "settings.yaml"
        
        if not config_path.exists():
            # Try parent directory (for scripts/)
            config_path = current.parent / "config" / "settings.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError("config/settings.yaml not found")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
