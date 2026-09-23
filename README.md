# AI-Powered Healthcare IT Support System

AI-powered IT support system for healthcare environment using LangChain and OpenAI API for intelligent ticket classification, automated responses, and knowledge base search with FastAPI backend.

Personal project, built to explore retrieval-augmented ticket triage with LangChain. It is not production software — see **Status** below for exactly what is and isn't implemented.

## Status

**Implemented**

- FastAPI service with chat and ticket endpoints
- Ticket classifier and knowledge-base retrieval over Chroma
- Prompt configuration split out into `config/prompts.yml`
- A unit test suite for the classifier
- Dockerfile and Compose setup

**Not implemented / known limitations**

- Requires an OpenAI API key; no offline/local model path
- Classifier is not evaluated against a labelled dataset — accuracy is unmeasured
- No auth on the API

## Built with

- **Python** — fastapi, uvicorn, langchain, openai, chromadb, PyYAML

## Running it

```bash
pip install -r requirements.txt
python src/main.py
```

## Layout

```
Dockerfile
config/
  prompts.yml
  settings.yml
docker-compose.yml
requirements.txt
src/
  ai_engine.py
  api/
    models.py
    routes.py
  chat_handler.py
  knowledge_base.py
  main.py
  ticket_classifier.py
tests/
  test_classifier.py
```

