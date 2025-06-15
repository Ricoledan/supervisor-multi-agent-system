# Multi-Agent Research System

## Overview

This project implements a sophisticated Multi-Agent System (MAS) architecture featuring a **Research Coordinator** that
orchestrates specialized AI agents to analyze academic research. The system transforms academic papers into a
searchable, structured, and semantically rich knowledge base through intelligent agent coordination powered by *
*LangGraph** state management and **Command-based routing**.

## Key Features

### Core Capabilities

- **Intelligent Agent Orchestration**: Research Coordinator uses LangGraph StateGraph with Command-based routing to
  dynamically delegate queries to specialized agents
- ️**Multi-Database Architecture**: Integrates Neo4j (graph), MongoDB (documents), and ChromaDB (vectors) for
  comprehensive data storage
- **Automated PDF Processing**: Advanced ingestion pipeline with entity extraction, topic modeling, and metadata
  enrichment using OpenAI GPT-4
- **Semantic Search & Retrieval**: Hybrid search combining vector similarity, graph traversal, and document analysis
- **Real-time Research Analysis**: Dynamic routing between relationship analysis and thematic analysis based on query
  classification
- ️**Professional CLI Interface**: Comprehensive command-line management with health checks, logging, and testing
  capabilities

### LangGraph Workflow Architecture

```
Research Coordinator (LangGraph StateGraph)
    ↓
Query Classification Node
    ↓
┌─────────────┬─────────────┬─────────────┐
│   Greeting  │   Simple    │  Research   │
│   Handler   │  Question   │   Query     │
└─────────────┴─────────────┴─────────────┘
                                  ↓
                          Planning Node
                                  ↓
                    ┌─────────────┬─────────────┐
                    │Relationship │   Theme     │
                    │  Analyst    │  Analyst    │
                    │   Node      │    Node     │
                    └─────────────┴─────────────┘
                                  ↓
                           Synthesis Node
                                  ↓
                           Final Response
```

### Agent Specializations

- **🎯 Research Coordinator**: Central supervisor using LangGraph Commands for intelligent query classification and agent
  delegation
- **🔗 Relationship Analyst**: Maps connections between papers, authors, concepts, and research lineages using Neo4j
  graph queries
- **📊 Theme Analyst**: Identifies patterns, topics, and trends across research literature using MongoDB document
  analysis
- **🏷️ Entity Extraction**: Automated identification of key concepts, methodologies, and research entities via LLM
  processing
- **📈 Topic Modeling**: Latent theme discovery and research domain classification with weighted term extraction

## ️ System Architecture

### Complete Workflow Pipeline

```
PDF Ingestion ➜ Entity Extraction ➜ Topic Modeling ➜ Graph Construction ➜ Vector Embedding ➜ Agent Analysis ➜ User Interaction
```

### Advanced Ingestion Pipeline

The system features a sophisticated **multi-stage ingestion pipeline** that processes academic PDFs:

1. **📄 PDF Text Extraction**: Uses PyMuPDF for robust text and metadata extraction
2. **🧠 LLM-Powered Analysis**: OpenAI GPT-4 extracts entities, relationships, and topics
3. **🔗 Knowledge Graph Construction**: Builds Neo4j nodes and relationships for papers, authors, and concepts
4. **📊 Topic Modeling**: Discovers research themes and categorizes content in MongoDB
5. **🎯 Vector Embeddings**: Creates semantic embeddings for similarity search in ChromaDB
6. **✅ Quality Validation**: Tests data integrity across all databases

```bash
# Run complete ingestion pipeline
python src/utils/ingestion_pipeline.py

# Test with a single PDF
python src/utils/ingestion_pipeline.py --test
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
│   │   │   ├── research_coordinator.py   # LangGraph orchestration agent
│   │   │   ├── relationship_analyst.py  # Neo4j graph analysis
│   │   │   └── theme_analyst.py         # MongoDB topic analysis
│   ├── databases/                 # Database configurations
│   │   ├── graph/                 # Neo4j configuration
│   │   ├── document/              # MongoDB configuration
│   │   └── vector/                # ChromaDB configuration
│   ├── services/                  # Database service layers
│   │   ├── graph_service.py       # Neo4j operations
│   │   ├── document_service.py    # MongoDB operations
│   │   └── vector_service.py      # ChromaDB operations
│   └── utils/                     # Utilities and tools
│       ├── ingestion_pipeline.py  # Comprehensive PDF processing
│       ├── model_init.py          # LLM initialization
│       └── agent_wrapper.py       # Agent response utilities
├── cli.py                         # Professional CLI interface
├── docker-compose.yml             # Multi-service orchestration
├── requirements.txt               # Python dependencies
└── sources/                       # PDF documents for ingestion
```

### Technology Stack

| Component             | Technology              | Purpose                                               |
|-----------------------|-------------------------|-------------------------------------------------------|
| **Agent Framework**   | LangGraph + LangChain   | Modern state-based multi-agent orchestration          |
| **LLM Integration**   | OpenAI GPT-4            | Entity extraction, topic modeling, and analysis       |
| **API Framework**     | FastAPI                 | High-performance web API with automatic documentation |
| **Graph Database**    | Neo4j                   | Knowledge graph for entity relationships              |
| **Document Database** | MongoDB                 | Structured document storage and topic modeling        |
| **Vector Database**   | ChromaDB                | Semantic search and similarity matching               |
| **Containerization**  | Docker + Docker Compose | Consistent deployment and scaling                     |
| **CLI Interface**     | Click                   | Professional command-line management                  |

## Prerequisites

- **Python**: 3.11+
- **Docker**: Latest version with Docker Compose
- **OpenAI API Key**: Required for LLM operations
- **System Requirements**: 8GB RAM minimum, 16GB recommended
- **Storage**: 20GB minimum, 50GB recommended for large document collections

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Ricoledan/supervisor-multi-agent-system
cd supervisor-multi-agent-system
cp .env.defaults .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Start the System

```bash
# Using the CLI (recommended)
python cli.py start

# Quick start with minimal health checks
python cli.py quick-start

# Start only databases for development
python cli.py start --databases-only
```

### 3. Add Research Papers

```bash
# Create sources directory and add PDFs
mkdir -p sources
# Copy your academic PDF files to sources/
```

### 4. Test the System

```bash
# Quick test with clean output
python cli.py test --simple

# Detailed system test
python cli.py test --query "machine learning applications"

# Test specific functionality
curl -X POST "http://localhost:8000/api/v1/agent" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do neural networks relate to computer vision?"}'
```

## ️ CLI Commands

The system includes a comprehensive CLI for professional management:

### System Management

```bash
python cli.py start           # Start all services
python cli.py stop            # Stop all services  
python cli.py restart         # Restart system
python cli.py status          # Check service status
```

### Development & Testing

```bash
python cli.py test            # Test system functionality
python cli.py test --simple   # Clean, formatted output
python cli.py health          # Run health checks
python cli.py health --detailed  # Comprehensive health analysis
python cli.py logs            # View system logs
python cli.py logs --follow   # Follow logs in real-time
```

### Database Management

```bash
python cli.py start --databases-only  # Start only databases
python cli.py restart --service neo4j  # Restart specific service
```

## Research Applications & Use Cases

### Literature Review Automation

```json
{
  "query": "What are the main approaches to transformer architectures in natural language processing?"
}
```

### Research Gap Identification

```json
{
  "query": "How do computer vision techniques connect to medical diagnosis research?"
}
```

### Trend Analysis

```json
{
  "query": "What themes are emerging in climate change adaptation research over the past 5 years?"
}
```

### Citation Network Analysis

```json
{
  "query": "Show me the research lineage and evolution of BERT language models"
}
```

### Cross-Disciplinary Discovery

```json
{
  "query": "How does reinforcement learning apply to robotics and autonomous systems?"
}
```

## API Documentation

### Core Research Endpoint

**POST /api/v1/agent**

```bash
curl -X POST "http://localhost:8000/api/v1/agent" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do neural networks relate to computer vision?"}'
```

### Response Structure

```json
{
  "status": "success",
  "message": "# 🎯 Research Analysis Results\n\n**Query:** How do neural networks relate to computer vision?\n\n## 🔗 Relationship Analysis\n\nBased on the knowledge graph analysis...",
  "query": "How do neural networks relate to computer vision?",
  "query_type": "RESEARCH_QUERY",
  "specialists_used": {
    "relationship_analyst": true,
    "theme_analyst": true
  },
  "system_health": {
    "relationship_analyst": "✅ Active",
    "theme_analyst": "✅ Active",
    "database_usage": "✅ High",
    "response_quality": "Database-driven"
  }
}
```

### Additional Endpoints

| Endpoint                 | Method | Description                     |
|--------------------------|--------|---------------------------------|
| `/api/v1/status`         | GET    | System health check             |
| `/api/v1/agent`          | POST   | Main research analysis endpoint |
| `/api/v1/agent/detailed` | POST   | Full conversation state         |
| `/api/v1/agent/raw`      | POST   | Debug endpoint with raw outputs |
| `/api/v1/agent/health`   | GET    | Agent system health check       |

## 🗄️ Database Schema & Architecture

### Neo4j Graph Schema

```cypher
// Nodes
(:Paper {id, title, year, source, research_field, methodology})
(:Author {name})
(:Concept {name, category, description})

// Relationships
(:Author)-[:AUTHORED]->(:Paper)
(:Paper)-[:CONTAINS]->(:Concept)
(:Concept)-[:RELATES_TO {type, description}]->(:Concept)
```

### MongoDB Collections

```javascript
// papers collection
{
    paper_id: String,
        metadata
:
    {
        title, authors, year, abstract, keywords,
            journal, doi, research_field, methodology
    }
,
    content: [{page, text}],
        entities
:
    {
        concepts, relationships
    }
,
    processed_at: Date
}

// topics collection  
{
    paper_id: String,
        category
:
    String,
        terms
:
    [{term, weight}],
        source
:
    String,
        created_at
:
    Date
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
        authors, year, research_field,
        chunk_id, chunk_total
    }],
    ids: [unique_chunk_ids]
}
```

## ️ Configuration & Environment

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

### Advanced Configuration

```bash
# LLM Model Selection
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo for faster responses

# Ingestion Pipeline Settings
CHUNK_SIZE=1000        # Text chunk size for embeddings
CHUNK_OVERLAP=200      # Overlap between chunks
MAX_CONCEPTS=15        # Maximum concepts per paper

# Performance Tuning
NEO4J_POOL_SIZE=10
MONGODB_POOL_SIZE=10
```

## Agent Tool System

### Relationship Analyst Tools

- **`analyze_research_relationships()`**: Queries Neo4j for entity connections
    - Paper lineages and citation networks
    - Author collaboration patterns
    - Cross-disciplinary concept relationships
    - Research influence patterns

### Theme Analyst Tools

- **`analyze_research_themes()`**: Queries MongoDB for topic patterns
    - Latent theme discovery across document collections
    - Research trend identification and evolution
    - Methodological approach analysis
    - Domain-specific terminology extraction

## Performance & Scalability

### Performance Characteristics

- **Query Response Time**: 15-45 seconds (depends on database size and complexity)
- **PDF Processing Speed**: 2-3 minutes per paper (including all extractions)
- **Concurrent Users**: Supports 5-10 simultaneous research queries
- **Database Storage**: ~500MB per 100 research papers

### System Requirements

- **Minimum**: 8GB RAM, 4 CPU cores, 20GB storage
- **Recommended**: 16GB RAM, 8 CPU cores, 50GB storage
- **Production**: 32GB RAM, 8+ CPU cores, 100GB+ storage

### Scalability Options

- **Horizontal Scaling**: Docker Compose replicas for API services
- **Database Optimization**: Connection pooling and memory tuning
- **Caching**: Redis integration for frequent queries (future enhancement)

## Troubleshooting

### Common Issues & Solutions

#### Database Connection Failures

```bash
# Check service status
python cli.py status

# View detailed logs
python cli.py logs --service neo4j
python cli.py logs --service mongodb
python cli.py logs --service chromadb

# Restart specific service
python cli.py restart --service neo4j
```

#### Empty Database Results

```bash
# Verify data ingestion completed
python cli.py test --query "machine learning"

# Check ingestion quality
python src/utils/ingestion_pipeline.py --test

# Re-run full ingestion if needed
python src/utils/ingestion_pipeline.py
```

#### API Timeout Issues

```bash
# Increase timeout for complex queries
python cli.py test --timeout 120

# Check database performance
python cli.py health --detailed

# Monitor system resources
python cli.py logs --follow
```

#### Source Directory Issues

```bash
# Verify sources directory exists
ls -la sources/

# Check PDF file permissions
python cli.py test --query "test"
```

## Development & Extension

### Adding New Specialist Agents

1. **Create Agent File**: `src/domain/agents/new_specialist.py`

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


@tool
def analyze_custom_data(query: str) -> str:
    """Custom analysis tool"""
    # Your custom database queries here
    return analysis_result


specialist_agent = create_react_agent(
    model=model,
    tools=[analyze_custom_data],
    prompt=SYSTEM_PROMPT
)
```

2. **Update Coordinator**: Add routing logic in `research_coordinator.py`
3. **Add API Endpoints**: Update `agent.py` if needed

### Custom Database Queries

```python
# Example: Custom Neo4j analysis
def analyze_author_networks(query: str):
    with driver.session() as session:
        result = session.run("""
            MATCH (a1:Author)-[:AUTHORED]->(p)<-[:AUTHORED]-(a2:Author)
            WHERE a1.name CONTAINS $query
            RETURN a1.name, a2.name, count(p) as collaborations
            ORDER BY collaborations DESC LIMIT 10
        """, query=query)
        return result.data()
```

### Development Workflow

```bash
# Start databases only for development
python cli.py start --databases-only

# Run API in development mode
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Monitor logs in separate terminal
python cli.py logs --follow

# Test changes
python cli.py test --simple
```

## Access Points

After starting the system, access these interfaces:

- **API Documentation**: http://localhost:8000/docs
- **API Status**: http://localhost:8000/api/v1/status
- **Neo4j Browser**: http://localhost:7474 (neo4j/password)
- **MongoDB Express**: http://localhost:8081
- **ChromaDB**: http://localhost:8001

## 🧪 Testing & Quality Assurance

### Comprehensive Testing

```bash
# Full system test with clean output
python cli.py test --simple

# Test with specific queries
python cli.py test --query "transformer models" --timeout 60

# Health check all components
python cli.py health --detailed

# Test ingestion pipeline
python src/utils/ingestion_pipeline.py --test

# API endpoint testing
curl -X GET "http://localhost:8000/api/v1/agent/health"
```

### Quality Validation

The system includes built-in quality checks:

- **Data Integrity**: Validates cross-database consistency
- **Response Quality**: Monitors agent specialist usage
- **Performance Metrics**: Tracks query response times
- **Database Health**: Monitors connection status and query performance

## Supported Document Types & Sources

### Input Formats

- **Primary**: PDF research papers with text content
- **Secondary**: Text files (.txt, .md) for preprocessing
- **Future**: DOI-based ingestion, arXiv API integration

### Recommended Paper Sources

- Academic conferences (NeurIPS, ICML, ACL, ICLR, etc.)
- Journal articles from major publishers (IEEE, ACM, Springer, Elsevier)
- Preprint servers (arXiv, bioRxiv, medRxiv)
- Technical reports and white papers

## References & Resources

- [LangGraph Multi-Agent Documentation](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [Neo4j Graph Database Documentation](https://neo4j.com/docs/)
- [ChromaDB Vector Database Documentation](https://docs.trychroma.com/)
- [FastAPI Framework Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Multi-Agent Systems](https://langchain-ai.lang.chat/langgraph/concepts/multi_agent/)

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## License

This project is licensed under the MIT License—see the LICENSE file for details.

---