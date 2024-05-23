import datetime
import json
from glob import glob
from pathlib import Path

import click
from litellm import completion

from .config import Config
from .note import Note

CONFIG = Config.from_json(Path(__file__).resolve().parent / "config.json")

PROMPT = """
You are a note-taking assistant. User will prompt you with notes. Some notes could be imperative, like "Remind me about dentist appointment on weekends".
Others could be simple sentences. Normalize note to be simple sentences. If there's a time reference in a message, try converting it to a particular date, given today's date. So instead of tomorrow there will be date in normalized message
For each note try to detect it's broader meaning and assign tags, if possible.
For example, "appointment", "medical". Output should be json in a form
```
{
    "processed_message": normalized_message,
    "tags": ["tag1", "tag2",...,"tagn"]
}
```
Output should be ONLY the json
"""


@click.group()
def cli():
    pass


@cli.command()
@click.argument("msg")
def add(msg):
    current_date = datetime.datetime.now()
    date_str = current_date
    weekday = current_date.weekday
    response = completion(
        model="ollama/llama3",
        messages=[
            {"content": f"Today is {date_str} {weekday}." + PROMPT, "role": "system"},
            {"content": msg, "role": "user"},
        ],
        api_base="http://localhost:11434",
    )
    # TODO better error hadling in case of something unexpected
    max_reties = 3
    succeded = False
    for i in range(max_reties):
        try:
            d = json.loads(response["choices"][0]["message"].content)
            d["creation_date"] = current_date
            d["initial_message"] = msg
            note = Note(**d)
            note.to_json()
            succeded = True
            break
        except json.JSONDecodeError:
            # simply continue
            pass
    if not succeded:
        raise ValueError("Something unexpected happened")


@cli.command()
@click.option("--n", type=int, help="number of notes to show", default=0)
@click.option(
    "--ascending", help="show messages sorted by date in ascending order", is_flag=True
)
def show(
    n,
    ascending,
):
    print(ascending)
    notes = sorted(glob(f"{CONFIG.notes_storage}/*.json"), reverse=not ascending)
    for n in notes:
        print("-" * 80)
        Note.from_json(n).show()


if __name__ == "__main__":
    cli()
