# HMO Chatbot System

This project is a microservice-based chatbot system that helps users inquire about services provided by Israeli health funds (HMOs), based on their personal details and membership tier.

## Overview

The system is composed of two main services:

- **Backend**: A FastAPI server that handles LLM calls, user data collection, and RAG-based answering using context from pre-embedded HTML documents.
- **Frontend**: A Streamlit app that provides an interactive chat interface for users.

## Features

- Collects user details interactively (name, ID, gender, age, HMO, card number, tier)
- Confirms details before answering questions
- Uses vector-based retrieval to find relevant answers from HMO service documents
- Supports both Hebrew and English inputs
- Dockerized and can run locally with `docker-compose`

Yes — based on your updated file structure (in the image), the `README.md` needs a small correction under **Project Structure** to reflect the actual folder layout and filenames. Here's what should be changed:


## Project Structure

```
phase2_app/
├── backend/
│   ├── main.py
│   ├── openai_utils.py
│   ├── retriever.py
│   ├── models.py
│   ├── html_loader.py
│   ├── prompts/
│   │   ├── collect_info.txt
│   │   └── answer_question.txt
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   └── embeddings.json
├── phase2_data/
│   └── *.html (source documents)
├── scripts/
│   └── embed_documents.py
├── docker-compose.yml
├── .env.example
└── README.md
```


## Running Locally

1. Make sure you have Docker and Docker Compose installed.

2. Add your Azure OpenAI credentials to `.env`:

```
AZURE_OPENAI_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
```

3. Build and run:

```bash
docker-compose up --build
```

4. Visit the frontend:

```
http://localhost:8501
```

The backend runs at:

```
http://localhost:8000
```

## Notes

- The backend expects the `embedded_docs.jsonl` file to exist in `phase2_data/`
- All prompts are located in `backend/prompts/`
- The frontend communicates with the backend using the `/ask` route

## License

This project is intended for educational or internal use. Please check with your organization before deploying it to production.
```
