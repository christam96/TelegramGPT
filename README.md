# TelegramGPT

V0: A ChatGPT-like interface for Telegram conversations.

V1: Indexing [WIP] (ref: https://python.langchain.com/docs/modules/data_connection/indexing)
- Avoid loading and recomputing embeddings over unchanged content
- Strategy:
    - Make use of a `RecordManager` to keep track of document writes into the vector store.
    - Need to re-populate `vectorestore` with documents since it doesn't have context for previously inserted embeddings
