# Multi-Agent Research System

## Overview

This project implements a sophisticated Multi-Agent System (MAS) architecture featuring a **Research Coordinator** that orchestrates specialized AI agents to analyze academic research. The system transforms academic papers into a searchable, structured, and semantically rich knowledge base through intelligent agent coordination.

##  Key Features

### Core Capabilities
- **Intelligent Agent Orchestration**: Research Coordinator routes queries to specialized agents based on analysis needs
- **Multi-Database Architecture**: Integrates Neo4j (graph), MongoDB (documents), and ChromaDB (vectors) for comprehensive data storage
- **Academic Paper Processing**: Automated ingestion pipeline that extracts metadata, entities, topics, and relationships from PDFs
- **Semantic Search & Retrieval**: Hybrid search combining vector similarity, graph traversal, and document analysis
- **Real-time Research Analysis**: Dynamic routing between relationship analysis and thematic analysis based on query type

### Agent Specializations
- **Research Coordinator**: Central supervisor that classifies queries and delegates to appropriate specialists
- **Relationship Analyst**: Maps connections between papers, authors, concepts, and research lineages using Neo4j
- **Theme Analyst**: Identifies patterns, topics, and trends across research literature using MongoDB
- **Entity Extraction**: Automated identification of key concepts, methodologies, and research entities
- **Topic Modeling**: Latent theme discovery and research domain classification

## ️ Architecture

### System Workflow
```
PDF Ingestion ➜ Entity Extraction ➜ Topic Modeling ➜ Graph Construction ➜ Vector Embedding ➜ Agent Analysis ➜ User Interaction
```

### Directory Structure
```
supervisor-multi-agent-system/
├── src/
│   ├── main.py                    # FastAPI application entry point
│   ├── api/v1/endpoints/          # API endpoints
│   │   ├── status.py              # Health check endpoints
│   │   └── agent.py               # Main agent interaction endpoint
│   ├── domain/
│   │   ├── agents/                # Specialized AI agents
│   │   │   ├── research_coordinator.py   # Central orchestration agent
│   │   │   ├── relationship_analyst.py  # Graph relationship analysis
│   │   │   └── theme_analyst.py         # Topic and theme analysis
│   │   └── formatters/            # Response formatting utilities
│   ├── databases/                 # Database configurations
│   │   ├── graph/                 # Neo4j configuration
│   │   ├── document/              # MongoDB configuration
│   │   └── vector/                # ChromaDB configuration
│   ├── services/                  # Database service layers
│   │   ├── graph_service.py       # Neo4j operations
│   │   ├── document_service.py    # MongoDB operations
│   │   └── vector_service.py      # ChromaDB operations
│   └── utils/                     # Utilities and tools
│       ├── ingestion_pipeline.py  # PDF processing pipeline
│       ├── model_init.py          # LLM initialization
│       └── agent_wrapper.py       # Agent response utilities
├── cli.py                         # Professional CLI interface
├── docker-compose.yml             # Multi-service orchestration
├── requirements.txt               # Python dependencies
└── sources/                       # PDF documents for ingestion
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | FastAPI | High-performance web API with automatic documentation |
| **Agent Framework** | LangChain + LangGraph | Multi-agent orchestration and workflow management |
| **LLM Integration** | OpenAI GPT-4 | Natural language processing and analysis |
| **Graph Database** | Neo4j | Knowledge graph for entity relationships |
| **Document Database** | MongoDB | Structured document storage and topic modeling |
| **Vector Database** | ChromaDB | Semantic search and similarity matching |
| **Containerization** | Docker + Docker Compose | Consistent deployment and scaling |
| **CLI Interface** | Click | Professional command-line management |

##  Prerequisites

- **Python**: 3.11+
- **Docker**: Latest version with Docker Compose
- **OpenAI API Key**: Required for LLM operations
- **Memory**: Minimum 8GB RAM recommended for all services

##  Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd supervisor-multi-agent-system
cp .env.defaults .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Start the System
```bash
# Using the CLI (recommended)
python cli.py start

# Or using Docker Compose directly
docker compose up --build
```

### 3. Add Academic Papers
```bash
# Create sources directory and add PDF files
mkdir sources
# Copy your academic PDF files to sources/
```

### 4. Test the System
```bash
# Quick test
python cli.py test --query "machine learning applications"

# Interactive test
curl -X POST "http://localhost:8000/api/v1/agent" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do neural networks relate to computer vision?"}'
```

##  CLI Commands

The system includes a professional CLI for easy management:

```bash
# System Management
python cli.py start           # Start all services
python cli.py stop            # Stop all services  
python cli.py restart         # Restart system
python cli.py status          # Check service status

# Development & Testing
python cli.py test            # Test system functionality
python cli.py health          # Run health checks
python cli.py logs            # View system logs

# Database Management
python cli.py start --databases-only  # Start only databases
```

##  Usage Examples

### Research Query Examples

**Relationship Analysis:**
```json
{
  "query": "How do transformer architectures connect to natural language processing?"
}
```

**Thematic Analysis:**
```json
{
  "query": "What are the main themes in climate change adaptation research?"
}
```

**Cross-Disciplinary Analysis:**
```json
{
  "query": "Show me connections between machine learning and medical diagnosis"
}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/status` | GET | System health check |
| `/api/v1/agent` | POST | Main research analysis endpoint |
| `/api/v1/agent/raw` | POST | Debug endpoint with raw outputs |

### Response Format
```json
{
  "status": "success",
  "message": "Formatted research analysis...",
  "query": "Original query",
  "system_health": {
    "relationship_analyst": "✅ Active",
    "theme_analyst": "✅ Active", 
    "database_usage": "✅ High",
    "response_quality": "Database-driven"
  }
}
```

##  Data Processing Pipeline

### 1. PDF Ingestion
- Extracts text content using PyMuPDF
- Generates metadata (title, authors, year, abstract)
- Creates document chunks for vector embedding

### 2. Entity Extraction
- Identifies key concepts, methodologies, datasets
- Extracts research relationships and dependencies
- Maps author and institutional connections

### 3. Topic Modeling
- Discovers latent themes across documents
- Clusters research by domain and approach
- Identifies emerging research trends

### 4. Knowledge Graph Construction
- Creates nodes for papers, authors, concepts
- Establishes relationships between entities
- Enables traversal and network analysis

### 5. Vector Embedding
- Generates semantic embeddings for all text chunks
- Enables similarity search and retrieval
- Supports hybrid search strategies

##  Agent Workflow

### Research Coordinator Decision Flow
```
Query Input
    ↓
Query Classification
    ↓
┌─────────────┬─────────────┬─────────────┐
│   Greeting  │   Simple    │  Research   │
│             │  Question   │   Query     │
└─────────────┴─────────────┴─────────────┘
                                  ↓
                          Analysis Planning
                                  ↓
                    ┌─────────────┬─────────────┐
                    │ Relationship│   Theme     │
                    │  Analysis   │  Analysis   │
                    └─────────────┴─────────────┘
                                  ↓
                            Synthesis & Response
```

### Specialist Agent Responsibilities

**Relationship Analyst:**
- Queries Neo4j for entity connections
- Maps citation networks and research lineages
- Identifies influential papers and authors
- Analyzes collaborative patterns

**Theme Analyst:**
- Queries MongoDB for topic patterns
- Identifies research themes and trends
- Extracts key terminology and concepts
- Analyzes methodological approaches

## ️ Database Schema

### Neo4j Graph Schema
```cypher
// Nodes
(:Paper {id, title, year, source})
(:Author {name})
(:Concept {name, category, description})

// Relationships
(:Author)-[:AUTHORED]->(:Paper)
(:Paper)-[:CONTAINS]->(:Concept)
(:Concept)-[:RELATES_TO]->(:Concept)
```

### MongoDB Collections
```javascript
// papers collection
{
  paper_id: String,
  metadata: {title, authors, year, abstract, keywords},
  content: [{page, text}],
  entities: {concepts, relationships},
  processed_at: Date
}

// topics collection  
{
  paper_id: String,
  category: String,
  terms: [{term, weight}],
  created_at: Date
}
```

### ChromaDB Schema
```python
# Collection: academic_papers
{
  documents: [text_chunks],
  embeddings: [vector_embeddings],
  metadatas: [{
    paper_id, page, source, title, 
    authors, year, research_field
  }],
  ids: [unique_chunk_ids]
}
```

##  Access Points

After starting the system:

- **API Documentation**: http://localhost:8000/docs
- **API Status**: http://localhost:8000/api/v1/status
- **Neo4j Browser**: http://localhost:7474 (neo4j/password)
- **MongoDB Express**: http://localhost:8081
- **ChromaDB**: http://localhost:8001

##  Configuration

### Environment Variables (.env)
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration (defaults provided)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DB=neo4j

MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USER=user
MONGODB_PASSWORD=password
MONGODB_DB=research_db

CHROMA_HOST=localhost
CHROMA_PORT=8001
```

## 🧪 Testing & Development

### Run System Tests
```bash
# Full system test
python cli.py test

# Specific functionality test
python cli.py test --query "transformer models"

# Health check
python cli.py health --detailed

# Database ingestion test
python src/utils/ingestion_pipeline.py --test
```

### Development Workflow
```bash
# Start databases only for development
python cli.py start --databases-only

# Run API in development mode
python -m uvicorn src.main:app --reload

# Monitor logs
python cli.py logs --follow
```

##  References & Citations

- [LangChain Multi-Agent Systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [Neo4j Graph Database](https://neo4j.com/docs/)
- [ChromaDB Vector Database](https://docs.trychroma.com/)
- [FastAPI Framework](https://fastapi.tiangolo.com/)

