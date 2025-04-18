# HMO Chatbot System

A microservice-based chatbot system for Israeli HMO (Health Fund) service inquiries, using RAG (Retrieval Augmented Generation) and LLM for providing accurate, personalized responses about health services.

The system uses:
- Azure OpenAI (GPT-4o) for chat interactions
- FastAPI for backend services
- Streamlit for frontend interface
- Vector-based document retrieval
- Docker and Docker Compose for deployment
- Enhanced logging and monitoring

## Features

- Bilingual support (Hebrew/English)
- Two-phase interaction:
  1. User details collection
  2. Service inquiries
- RAG-powered answers from HMO documents
- Real-time performance monitoring
- Comprehensive logging system
- Language-aware response formatting
- Docker containerization

## Tech Stack

- Python 3.11
- Azure OpenAI (GPT-4o)
- FastAPI + Uvicorn
- Streamlit
- Docker and Docker Compose
- Vector embeddings for RAG
- Enhanced logging with metrics tracking

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/EitanBakirov/health-maintenance-chatbot.git
cd health-maintenance-chatbot
```

### 2. Configure environment
Copy `.env.example` to `.env` and add your Azure OpenAI credentials:
```env
AZURE_OPENAI_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
```

## Project Structure

```
app/
├── backend/          # FastAPI service
│   ├── main.py      # API endpoints
│   ├── openai_utils.py
│   ├── retriever.py # RAG implementation
│   ├── initialize.py
│   └── prompts/     # System prompts
├── frontend/        # Streamlit interface
│   └── app.py
├── shared/          # Common utilities
│   ├── logger_config.py
│   └── monitoring.py
├── scripts/         # Data processing
│   └── embed_documents.py
├── phase2_data/     # Source documents
└── docker-compose.yml
```

## Running Options

### Docker Compose

1. Build and run containers:
```bash
docker compose up --build
```

Access:
- Frontend: http://localhost:8501
- Backend: http://localhost:8000

## API Endpoints

- `POST /ask` - Main chat endpoint
- `GET /metrics` - System monitoring
- `GET /health` - Service health check

## Monitoring & Metrics

The system tracks:
- LLM Operations
  - Success/failure rates
  - Response times
  - Token usage
- RAG Performance
  - Query similarity scores 
  - Document retrieval stats
  - No-match rates
- Conversation Flow
  - Language statistics
  - Phase completion rates
  - User interaction patterns

## Environment Variables

Required in `.env`:
```env
AZURE_OPENAI_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
```

## Example Interaction

1. Language Selection
2. User Information Collection
3. Service Inquiries with RAG
4. Personalized Responses

## Notes

- Requires Azure OpenAI access
- Supports both RTL and LTR text
- Auto-generates embeddings on first run
- Comprehensive logging system

## Potential Upgrades

- Utilize a dedicated VectorDB to save the files
- Add analytics dashboard for monitoring
- Support additional document formats (PDF, DOC)
