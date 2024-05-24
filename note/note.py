import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .config import Config

CONFIG = Config.from_json(Path(__file__).resolve().parent / "config.json")


@dataclass
class Note:
    creation_date: datetime
    initial_message: str
    processed_message: str
    tags: list[str]

    def to_json(self):
        d = asdict(self)
        d["creation_date"] = d["creation_date"].strftime("%y-%m-%d-%H:%M:%S")
        # simply save to some file based on date
        with open(f"{CONFIG.notes_storage}/{d['creation_date']}.json", "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def from_json(path):
        with open(path) as f:
            d = json.load(f)
        d["creation_date"] = datetime.strptime(d["creation_date"], "%y-%m-%d-%H:%M:%S")
        return Note(**d)

    def show(self):
        print("Creation date:", self.creation_date)
        print("Original message:", self.initial_message)
        print("LLM processed message:", self.processed_message)
        print("Automatic tags:", self.tags)
