import datetime
import json
import os
import re
import shutil
from glob import glob
from pathlib import Path

import click
import faiss
from litellm import completion
from sentence_transformers import SentenceTransformer

from .config import Config
from .note import Note

CONFIG = Config.from_json(Path(__file__).resolve().parent / "config.json")

PROMPT = """
You are a note-taking assistant. User will prompt you with notes. It can be appointments, todos,
interesting facts, important dates, foreign words translation, studying materials, person contacts or anythong else.
Some notes could be imperative, like "Remind me about dentist appointment on weekends".
Others could be simple sentences. Normalize note to be simple sentences.
If there's a time reference in a message, try converting it to a particular date, given today's date.
So instead of tomorrow there will be date in normalized message.
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

SEARCH_PROMPT = """
You are a note-taking assistant. User will prompt you with search request and notes to search through.
Notes are json files in a format.
```
{
    "processed_message": normalized_message,
    "tags": ["tag1", "tag2",...,"tagn"],
    "message_id": id 
}
```
Notes are at the start of this message
Your task is to find relevant notes, and give answer as json list.
[message_id1, message_id2, message_id3, ..., message_idn ]
Answer only with list
You query is:
"""

CHAT_PROMPT = "Given this collection of notes in json format, answer this question:"


@click.group()
def cli():
    pass


def create_index(path, d=None):
    if Path(path).exists():
        index = faiss.read_index(path)
    elif d is not None:
        # simple brute-force for small scales
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError("Index doesn't exist, and dimension d wasn't given")
    return index


def create_tags(path):
    if Path(path).exists():
        with open(path) as f:
            tags = list(sorted(([l.strip() for l in f])))
    else:
        tags = list()
    return tags


@cli.command()
@click.argument("msg")
def add(msg):
    current_date = datetime.datetime.now()
    date_str = current_date
    weekday = current_date.strftime("%A")
    today_str = f"Today is {date_str} {weekday}."
    response = completion(
        messages=[
            {"content": f"{today_str}" + PROMPT, "role": "system"},
            {"content": msg, "role": "user"},
        ],
        **CONFIG.litellm_config,
    )
    # TODO better error hadling in case of something unexpected
    json_pattern = r"({[^{}]*}|[\[{].*[\]}])"

    max_retries = 3
    succeded = False
    for i in range(max_retries):
        try:
            # llama3 sometimes adds additional text, so it's easier to simply find json and try to load it
            possible_json = re.findall(
                json_pattern, response["choices"][0]["message"].content, re.DOTALL
            )
            d = json.loads(possible_json[0])
            d["creation_date"] = current_date
            d["initial_message"] = msg
            note = Note(**d)
            note.to_json()
            # let's compile all the bits together and apply sentence transformer
            sentence_msg = (
                f"{today_str} {msg} {note.processed_message}. Tags: {note.tags}"
            )
            # according to here, this is a good model to use
            # https://huggingface.co/blog/mteb
            sentence_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2"
            )
            emb = sentence_model.encode(sentence_msg)
            succeded = True
            break
        except json.JSONDecodeError:
            # simply continue
            pass
    if not succeded:
        raise ValueError(
            f"LLM returned not a valid json {max_retries} in a row, you might try again or change message. LLM message was {response['choices'][0]['message'].content}"
        )

    # for checking results
    note.show()
    # handling index
    index = create_index(f"{CONFIG.notes_storage}/index", d=emb.shape[0])
    index.add(emb[None, :])
    faiss.write_index(index, f"{CONFIG.notes_storage}/index")
    # handling tags
    tags = create_tags(f"{CONFIG.notes_storage}/tags")
    # if tags not empty, update
    if note.tags:
        for t in note.tags:
            if t not in tags:
                tags.append(t)
        with open(f"{CONFIG.notes_storage}/tags", "w") as f:
            f.write(os.linesep.join(list(tags)))


@cli.command()
@click.argument("query")
@click.option(
    "--type",
    type=click.Choice(["sent", "llm"]),
    default="sent",
    help="Search type. Options: [sent, llm], sent is for sentence embedding search, and llm is for brute-force query approach",
)
@click.option("--n", type=int, default=10, help="How much notes to show")
def search(query, type, n):
    notes = list(sorted(glob(f"{CONFIG.notes_storage}/*.json")))
    if type == "sent":
        # let's search for all the notes first
        # because the filename is date,
        # their sorted list can be directly related to index
        index = create_index(f"{CONFIG.notes_storage}/index")

        current_date = datetime.datetime.now()
        date_str = current_date
        weekday = current_date.weekday
        today_str = f"Today is {date_str} {weekday}."

        sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        emb = sentence_model.encode(today_str + query)

        scores, top_k = index.search(emb[None,], n)
        for i in top_k[0]:
            print("-" * 80)
            Note.from_json(notes[i]).show()
    if type == "llm":
        # TODO reduce copy-paste
        current_date = datetime.datetime.now()
        date_str = current_date
        weekday = current_date.strftime("%A")
        today_str = f"Today is {date_str} {weekday}."
        note_list = []
        note_dict_list = []
        max_notes = n
        for i, n in enumerate(notes):
            note_dict_list.append(Note.from_json(n))
            with open(n) as f:
                note = json.load(f)
                note["message_id"] = i
                note_list.append(json.dumps(note, indent=2))
        notes_str = os.linesep.join(note_list)
        response = completion(
            messages=[
                {
                    "content": today_str
                    + notes_str
                    + SEARCH_PROMPT
                    + f"Your query is '{query}'",
                    "role": "user",
                },
            ],
            **CONFIG.litellm_config,
        )
        # TODO better error hadling in case of something unexpected
        json_pattern = r"({[^{}]*}|[\[{].*[\]}])"

        max_retries = 3
        succeded = False
        for i in range(max_retries):
            try:
                # llama3 sometimes adds additional text, so it's easier to simply find json and try to load it
                possible_json = re.findall(
                    json_pattern, response["choices"][0]["message"].content, re.DOTALL
                )
                ids = json.loads(possible_json[0])
                relevant_notes = [note_dict_list[j] for j in ids]
                succeded = True
                break
            except json.JSONDecodeError:
                # simply continue
                pass
        if not succeded:
            raise ValueError(
                f"LLM returned not a valid json {max_retries} in a row, you might try again or change message. LLM message was {response['choices'][0]['message'].content}"
            )
        for n in relevant_notes[:max_notes]:
            print("-" * 80)
            n.show()


@cli.command()
@click.argument("query", type=str)
def chat(query):
    notes = list(sorted(glob(f"{CONFIG.notes_storage}/*.json")))
    # TODO really reduce copy-paste
    current_date = datetime.datetime.now()
    date_str = current_date
    weekday = current_date.strftime("%A")
    today_str = f"Today is {date_str} {weekday}."
    note_list = []
    note_dict_list = []
    for i, n in enumerate(notes):
        note_dict_list.append(Note.from_json(n))
        with open(n) as f:
            note = json.load(f)
            note["message_id"] = i
            note_list.append(json.dumps(note, indent=2))
    notes_str = os.linesep.join(note_list)
    history = today_str + notes_str + CHAT_PROMPT + f"Your query is '{query}'"
    response = completion(
        messages=[
            {"content": history, "role": "user"},
        ],
        **CONFIG.litellm_config,
    )
    print(response["choices"][0]["message"].content)
    history += response["choices"][0]["message"].content
    # continue chat until Ctrl + C
    while True:
        next_question = input("> ")
        history += next_question
        # would be good to have better method for history, but I don't know if there are for APIs
        response = completion(
            messages=[
                {"content": history, "role": "user"},
            ],
            **CONFIG.litellm_config,
        )
        print(response["choices"][0]["message"].content)
        history += response["choices"][0]["message"].content


@cli.command()
@click.option("--n", type=int, help="number of notes to show", default=0)
@click.option(
    "--ascending", help="show messages sorted by date in ascending order", is_flag=True
)
def show(
    n,
    ascending,
):
    notes = list(sorted(glob(f"{CONFIG.notes_storage}/*.json"), reverse=not ascending))[
        :n
    ]
    for n in notes:
        print("-" * 80)
        Note.from_json(n).show()


@cli.command()
@click.option(
    "--show",
    type=str,
    help="Shows notes with a particular tag. If not given, all the tags are shown",
    default="",
)
def tag(show):
    if not show:
        tags = create_tags(f"{CONFIG.notes_storage}/tags")
        # show all available tags
        print(tags)
    else:
        notes = sorted(glob(f"{CONFIG.notes_storage}/*.json"))
        for n in notes:
            note = Note.from_json(n)
            if show in note.tags:
                print("-" * 80)
                note.show()


@cli.command()
def clear_all():
    shutil.rmtree(CONFIG.notes_storage)


if __name__ == "__main__":
    cli()
