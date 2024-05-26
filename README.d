# LLM for Galaxy Clusters
In this repository we train a LLM to be an expert on galaxy clusters using a curated set of scientific articles on galaxy clusters. 

## How to use:
- Create a `.env` file containing `OPENAI_API_KEY`.
- Run `streamlit run LLM-GalaxyClusters.py`

## How does it work?
This code has two main components:
1. The code ingests a curated set of scientific articles on galaxy clusters in the form of PDFs. Then, it creates vector embeddings for these articles and saves them in a `chromadb.sqlite3` database.
2. The code uses `LangChain` to call a LLM (`chatgpt-3.5-turbo`) to generate answers based off the users questions. The responses are augmented using the RAG technique that concentrates the LLM's answer based off the ingested PDFs.


## Contributing
If you are interested in contributing to the code or the base of PDFs, please contact me via `carterrhea93@gmail.com` or leave a GitHub issue.

The list of scientific articles used to train the LLM can be found in `LLM-GalaxyClusters.bib`.

