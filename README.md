# App architecture
At this point app is a very simple set of scripts. All the magic is happening in `note_app.py`. This script has functions for main features:

  * `add` to add new note. New notes are processed with an LLM to try to convert it to more general note. For example, if the note was given in imperative case ("remind me/write down/etc."), LLM is asked to process this note to be a simple clause. Also if there are dates or time reference, LLM is asked to convert those to precise dates, given today's date and day of the week. Also, LLM is asked to assign some tags automatically, so it would be possible to search by tags later.
  * `search` is used to search across all the notes using natural language. It is done by using sentence embeddings of `sentence-transformers/all-mpnet-base-v2`. The sentence embedding is applied to concatenation of current date, original message, LLM processed message and LLM automatic tags. Also search can be done with the use of LLM. To be honest, I personally prefer approach with sentence embeddings, because it is done once and it's much cheaper then LLM, and you don't have to worry about context length. Though it is possible to unite these approaches in RAG-style approach, where you first select candidates with sentence embedding, and then do precise search with an LLM. Also, it can be a separate feature, where you can have a "chat" with your notes. 
  * `show` simply shows you n messages sorted by date.
  * `clear-all` removes folder with notes. It is not a proper deletion method, it simply removes folder with notes (which is by default `~/notes_storage`)

Notes are saved in a very simple structure. Each note is saved in `json` file with fields `creation_date`, `initial_message`, `processed_message`, `tags`. 
Also the name of each `json` is it's creation date. It allows to easily sort notes by date and it is possible to use index of the note as its id. It also easy to look at what results the program got. Proper database would be more robust, but also more complex. 

# Libraries
The main libraries used are:
  * [ollama](https://github.com/ollama/ollama) - for running open sourced LLMs locally. I used `llama3 8B` model. It might be worse than using OpenAI's or Anthropic's API, but to me the possibility to run things locally is always a plus. The model type can be changed in `config.json`. One of the disadvantages is that it is distributed as a binary files and models are quite heavy (`llama3 8B` is more that 4Gbs)
  * [litellm](https://github.com/BerriAI/litellm) - for calling `ollama`. This library provides a unified interface for a lot of different APIs and for `ollama` local models, which would allow to easily switch LLM provider in the future. Also, [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter) uses `litellm` under the hood and allows to run code, generated by LLM. While I didn't use open interpreter here, it could be a good extension of this app (to set reminders at the very least)
  * [sentence-transformers](https://www.sbert.net/) - for embedding notes with `all-mpnet-base-v2` for semantic search
  * [faiss](https://github.com/facebookresearch/faiss) for searching across sentence embeddings. I used simple index, but there are different kind of them, for faster search at scale. Even better option would be to use a proper database. From what I researched, [Milvus](https://milvus.io/) is a very good option for such tasks. Another good option would be `ElasticSearch`. I decided to stick with `faiss` for simplicity.
  * [click](https://github.com/pallets/click) for simple command line interface

  # Usage

 1. Installing `ollama` (not handled by pip) or changing LLM provider in `config.json`
 2. Installing package `pip install .` or `pip install -e .` for editable install. The later will allow changing LLMs later more easily
 3. You can run `bash note_generation.sh`, it contains notes that ChatGPT generated. Or simply run commands by hand:
 ```
 (test_task_2) [alex@bd2 notes_test_task]$ note add "Remind me about meeting with Bob this Saturday"

Creation date: 2024-05-24 15:10:17.604452
Original message: Remind me about meeting with Bob this Saturday
LLM processed message: Meeting with Bob on 2024-05-26
Automatic tags: ['appointment', 'meeting', 'person-contact']
 ```
 The date is wrong by the way, which is unfortunately quite frequent

 Show notes:
 ```
 (test_task_2) [alex@bd2 notes_test_task]$ note show --n 5
--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:18:10
Original message: Set up a meeting with the marketing team next Tuesday
LLM processed message: Meeting with marketing team on 2024-05-30
Automatic tags: ['appointment', 'marketing']
--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:18:01
Original message: Read the article on artificial intelligence advancements
LLM processed message: Article on artificial intelligence advancements
Automatic tags: ['study', 'learning']
--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:17:51
Original message: Chinese: Zàijiàn means Goodbye
LLM processed message: Goodbye
Automatic tags: ['phrase', 'language']
--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:17:43
Original message: Plant flowers in the garden
LLM processed message: plant flowers
Automatic tags: []
--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:17:34
Original message: History: The Renaissance was a period of cultural revival in Europe
LLM processed message: The Renaissance was a period of cultural revival in Europe
Automatic tags: ['history', 'culture']
```
Search with sentence embeddings
```
note search "Show me my appointments" --n 5

--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:16:38
Original message: Pick up dry cleaning on Friday
LLM processed message: Pick up dry cleaning on 2024-05-24
Automatic tags: ['appointment']
--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:12:34
Original message: Meeting with Bob on Friday at 2 PM
LLM processed message: Meeting with Bob on 2024-05-24 14:00:00.000000
Automatic tags: ['appointment', 'meeting', 'contact']
--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:15:43
Original message: Yoga class every Wednesday at 6 PM
LLM processed message: appointment
Automatic tags: ['appointment', 'recurring', 'fitness']
--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:11:38
Original message: Doctor's appointment on June 5th at 3 PM
LLM processed message: appointment on 2024-06-05
Automatic tags: ['appointment', 'medical']
--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:16:13
Original message: Coffee date with Emily on Thursday at 10 AM
LLM processed message: Coffee date with Emily on 2024-05-30 at 10:00:00
Automatic tags: ['appointment', 'social']
```
Search with LLM
```
(test_task_2) [alex@bd2 notes_test_task]$ note search "Show me my appointments" --n 5 --type llm

--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:15:43
Original message: Yoga class every Wednesday at 6 PM
LLM processed message: appointment
Automatic tags: ['appointment', 'recurring', 'fitness']
--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:16:13
Original message: Coffee date with Emily on Thursday at 10 AM
LLM processed message: Coffee date with Emily on 2024-05-30 at 10:00:00
Automatic tags: ['appointment', 'social']
--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:16:38
Original message: Pick up dry cleaning on Friday
LLM processed message: Pick up dry cleaning on 2024-05-24
Automatic tags: ['appointment']
```
For some reason LLM dropped some obvious results. I guess further prompt-engineering is needed, or RAG.

Usage of tags:
```
(test_task_2) [alex@bd2 notes_test_task]$ note tag
['appointment', 'bill', 'biology', 'birthday', 'book', 'cafe', 'climatechange', 'contact', 'culture', 'data science', 'deadline', 'economics', 'education', 'equation', 'event', 'fact', 'finance', 'fitness', 'forecast', 'foreign_word', 'formula', 'geography', 'geometry', 'gift', 'greeting', 'groceries', 'history', 'important date', 'interesting_fact', 'language', 'language learning', 'learning', 'marketing', 'math', 'medical', 'meeting', 'minimalism', 'payment', 'phone', 'phrase', 'phrase translation', 'physics', 'portuguese', 'reading', 'recommendation', 'recurring', 'research', 'science', 'self-improvement', 'shopping', 'social', 'study', 'task', 'translation', 'trip planning', 'weather', 'wedding']

(test_task_2) [alex@bd2 notes_test_task]$ note tag --show "self-improvement"

--------------------------------------------------------------------------------
Creation date: 2024-05-24 15:15:59
Original message: Write a blog post on minimalism
LLM processed message: Blog post about minimalism. Embrace simplicity, let go of unnecessary items and focus on what truly adds value to your life.
Automatic tags: ['minimalism', 'self-improvement']
```
I like how LLM processed note about minimalism is not minimalistic.

Also there's a possibility to chat with an LLM based on the notes collection:
```
(test_task_2) [alex@bd2 notes_test_task]$ note chat "what are my interests?"
Based on the provided JSON data, I can extract the tags associated with each note. Then, I'll count the frequency of each tag to determine your interests.

After analyzing the data, here's what I found:

* The most frequent tags (with 2 occurrences each) are:
        + appointment
        + translation
* The next most frequent tags (with 1 occurrence each) are:
        + finance
        + fitness
        + history
        + language learning
        + marketing
        + minimalism
        + self-improvement
        + study

Based on this analysis, it appears that your interests include:

1. Scheduling and planning appointments
2. Learning languages and phrases
3. Financial management and planning
4. Fitness and wellness
5. History and cultural studies
6. Marketing and business development
7. Minimalism and simplicity
8. Self-improvement and personal growth
9. Education and learning

Please note that these are just inferences based on the provided data, and your actual interests may be different.
> What are some languages that I learn?
A fun question!

Based on the JSON data, it looks like you're interested in learning several languages. Here are the language-related tags and phrases I found:

1. Italian: Grazie means Thank you
2. French: Au revoir means Goodbye
3. Portuguese: Obrigado means Thank you
4. Chinese: Zōijiān means Goodbye

These suggest that you're interested in learning Italian, French, Portuguese, and Chinese languages!

Please keep in mind that these are just inferred from the provided data, and your actual language interests may be different.
>
```

# Conclusions
Some thoughts about current implementation

* it would be better to use database
* both sentence embedding and LLM based search have their weak points. Embeddings are not always precise enough for correct retrieval and can't answer some particular requests and LLM often makes mistakes and selects less data. And you need to prompt it with all the notes and I feel like it hurts the performance. I think RAG should help with both approaches.
* LLM makes mistakes with dates and sometimes with tags. Also processed LLM note is not always what you expect, so better model and further prompt engineering are required.
* Chat with model also can be optimized with RAG.
