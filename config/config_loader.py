"""
Configuration loader for the Adaptive RF Receiver system.
Handles loading and merging YAML configuration files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Load and manage configuration files."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing config files. If None, uses the config directory
                       relative to this file.
        """
        if config_dir is None:
            self.config_dir = Path(__file__).parent
        else:
            self.config_dir = Path(config_dir)
            
        self._system_config = None
        self._model_config = None
        self._merged_config = None
    
    def load_system_config(self) -> Dict[str, Any]:
        """Load system configuration."""
        if self._system_config is None:
            config_path = self.config_dir / "system_config.yaml"
            self._system_config = self._load_yaml(config_path)
        return self._system_config
    
    def load_model_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        if self._model_config is None:
            config_path = self.config_dir / "model_config.yaml"
            self._model_config = self._load_yaml(config_path)
        return self._model_config
    
    def get_detector_config(self) -> Dict[str, Any]:
        """
        Get configuration specifically formatted for AnomalyDetector.
        
        Returns:
            Dictionary with keys: model, training, threshold, preprocessing, model_dir
        """
        model_config = self.load_model_config()
        system_config = self.load_system_config()
        
        detector_config = {
            'model': model_config['model'],
            'training': model_config['training'],
            'threshold': model_config['threshold'],
            'preprocessing': model_config['preprocessing'],
            'model_dir': system_config['paths']['model_dir']
        }
        
        return detector_config
    
    def get_network_config(self) -> Dict[str, Any]:
        """Get network configuration."""
        system_config = self.load_system_config()
        return system_config['network']
    
    def get_signal_config(self) -> Dict[str, Any]:
        """Get signal processing configuration."""
        system_config = self.load_system_config()
        return system_config['signal']
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get device configuration."""
        system_config = self.load_system_config()
        return system_config['device']
    
    def get_merged_config(self) -> Dict[str, Any]:
        """Get merged system and model configuration."""
        if self._merged_config is None:
            system_config = self.load_system_config()
            model_config = self.load_model_config()
            
            # Merge configurations
            self._merged_config = {**system_config, **model_config}
            
        return self._merged_config
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML file."""
        try:
            if not file_path.exists():
                logger.warning(f"Config file not found: {file_path}")
                return {}
                
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                logger.warning(f"Empty config file: {file_path}")
                return {}
                
            logger.info(f"Loaded config from: {file_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            return {}
    
    def save_config(self, config: Dict[str, Any], file_path: str):
        """Save configuration to a YAML file."""
        try:
            config_path = self.config_dir / file_path
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
                
            logger.info(f"Saved config to: {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {e}")
            raise

# Global config loader instance
_config_loader = None

def get_config_loader() -> ConfigLoader:
    """Get the global config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

def load_detector_config() -> Dict[str, Any]:
    """Convenience function to load detector configuration."""
    return get_config_loader().get_detector_config()

def load_network_config() -> Dict[str, Any]:
    """Convenience function to load network configuration."""
    return get_config_loader().get_network_config()

def load_signal_config() -> Dict[str, Any]:
    """Convenience function to load signal configuration."""
    return get_config_loader().get_signal_config()
