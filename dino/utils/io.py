import yaml
import torch

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(state, filepath):
    """Save a checkpoint."""
    torch.save(state, filepath)
