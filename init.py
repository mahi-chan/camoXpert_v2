import os
from pathlib import Path

# Define the file structure
file_structure = {
    "data": {
        "COD10K": {"train": {}, "val": {}, "test": {}},
        "synthetic": {}
    },
    "models": {
        "camoxpert.py": "",
        "backbone.py": "",
        "experts.py": "",
        "fusion.py": "",
        "segmentation_head.py": "",
        "utils.py": ""
    },
    "losses": {
        "camoxpert_loss.py": "",
        "dice_loss.py": "",
        "structure_loss.py": ""
    },
    "metrics": {
        "cod_metrics.py": ""
    },
    "notebooks": {
        "camoxpert_experiments.ipynb": ""
    },
    "scripts": {
        "train.py": "",
        "validate.py": "",
        "test.py": "",
        "inference.py": "",
        "benchmark.py": ""
    },
    "configs": {
        "config.yaml": "",
        "hyperparameters.yaml": ""
    },
    "results": {
        "checkpoints": {},
        "figures": {},
        "logs": {}
    },
    "requirements.txt": "",
    "README.md": "",
    "main.py": ""
}

# Function to create the file structure
def create_structure(base_path, structure):
    for name, content in structure.items():
        path = Path(base_path) / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)
        else:
            path.touch()

# Create the file structure
base_directory = "project-root"
create_structure(base_directory, file_structure)
print(f" File structure created at: {Path(base_directory).resolve()}")