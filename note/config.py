import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    litellm_config: dict
    notes_storage: str

    @staticmethod
    def from_json(path):
        with open(path) as f:
            d = json.load(f)
        d["notes_storage"] = Path(d["notes_storage"]).expanduser().resolve()
        os.makedirs(d["notes_storage"], exist_ok=True)
        return Config(**d)
