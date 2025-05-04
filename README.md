# supervisor-multi-agent-system

## Overview

This project implements a Multi-Agent System (MAS) architecture featuring a single supervisor agent that orchestrates
which agent should be called, served through a FastAPI interface.

## Key Features

- **Task Delegation & Coordination**: Assigns tasks to subordinate AI agents, manages inter-agent communication, and
  ensures efficient task execution.

- **Tool-Calling & API Integration**: Directly interacts with external APIs, databases, and computational tools, serving
  as an interface between agents and external services.

- **Monitoring & Control**: Evaluates agent outputs, detects errors, and can override or redirect actions to ensure
  accurate and efficient operations.

- **Decision-Making & Adaptive Workflow**: Adjusts workflows dynamically based on agent responses, reassigning tasks or
  retrying operations when necessary.

- **Multi-Step Planning & Execution**: Structures complex workflows into sequential steps, ensuring dependencies between
  tasks are handled correctly.

- **Error Handling & Recovery**: Detects failures or inconsistencies in agent outputs, implementing retry mechanisms or
  alternative strategies when needed.

- **Hierarchical Control**: Manages multiple layers of agents, including other supervisors, to scale AI-driven workflows
  effectively.

- **Policy & Rule Enforcement**: Enforces compliance with business logic, security constraints, and ethical guidelines,
  filtering or modifying responses accordingly.

## Architecture

### Directory Structure

```text
supervisor-multi-agent-system/
├── Dockerfile
├── README.md
├── requirements.txt
├── .env.defaults
├── .env
├── src/
│   ├── main.py
│   └── api/
│       └── v1/
│           ├── endpoints/
│           │   ├── status.py
│           │   └── agent.py
└── docker-compose.yml
```

### Components

- **FastAPI**: a modern, high-performance web framework for building APIs with Python.

- **Langchain**: a framework for developing applications powered by large language models.

- **LangGraph**: a library designed for building stateful, multi-actor applications with LLMs, facilitating the creation
  of agent and multi-agent workflows.

- **Neo4j**: a leading open-source graph database management system implemented in Java.

- **Faiss**: a library for efficient similarity search and clustering of dense vectors.

- **ChromaDB**: an open-source vector database optimized for applications utilizing large language models.

- **PostgresSQL**: a powerful, open-source object-relational database management system known for
  its robustness, extensibility, and compliance with SQL standards.

- **Docker**: an open-source platform that automates the deployment, scaling, and management of applications using
  containerization.

### System Workflow Overview

#### Objective

A multi-agent system that transforms academic papers into a searchable, structured, and semantically rich knowledge
graph
through these high-level stages:

**PDFs/Docs ➜ Extraction ➜ Topic Modeling ➜ Summarization ➜ Graph Construction ➜ Semantic Retrieval ➜ User Interaction**

### Step-by-Step Agent-Orchestrated Flow

| **Step**                          | **Agent**                  | **Function**                                                                                 |
|-----------------------------------|----------------------------|----------------------------------------------------------------------------------------------|
| 1. Ingest and parse PDFs          | **Supervisor**             | Coordinates all downstream tasks and delegates actions to appropriate agents                 |
| 2. Extract entities and events    | **EntityExtractor**        | Identifies key concepts, people, organizations, and temporal events from documents           |
| 3. Generate paper/topic summaries | **Summarizer**             | Produces concise summaries highlighting methods, arguments, and findings                     |
| 4. Perform topic modeling         | **TopicModel**             | Analyzes document content to detect latent themes and assigns topic labels                   |
| 5. Store in vector database       | **Retriever**              | Saves documents and associated metadata as embeddings into ChromaDB                          |
| 6. Build knowledge graph          | **Neo4jWriter**            | Constructs and maintains a graph linking documents, entities, and topics                     |
| 7. Serve user queries             | **Retriever + Supervisor** | Combines semantic search with graph traversal to answer user prompts and support exploration |

### Workflow Goals

1. Automated Research Understanding
   • Summarize complex academic texts into digestible formats
   • Highlight ideological perspectives, methods, and critique patterns

2. Knowledge Graph Construction
   • Create structured relationships between entities, topics, and documents
   • Support visual and logical querying (e.g., via Neo4j or GraphQL)

3. Topic-Aware Semantic Retrieval
   • Let users search documents by meaning and topic (e.g., “LLMs + critique + ethics”)
   • Leverage hybrid retrieval: vector search and graph constraints

4. Agentic Reasoning Framework
   • Enable agents to reason across documents over time (via Memory and Supervisor)
   • Compose multi-hop inferences across extracted knowledge (e.g., cause-effect chains)

5. Foundation for New Dataset Creation
   • Output labeled topic-document pairs
   • Auto-generate critique prompts or RLHF-ready data

#### Agent Responsibilities

- **Supervisor**: The main agent that orchestrates the workflow, managing the interaction between subordinate
  agents and external systems.

- **Entity Extractor**  
  Identifies and extracts named entities, key phrases, authors, institutions, methods, datasets, and concepts from raw
  academic text or PDFs. This structured data is used as input for linking and graph construction.

- **Entity Linker**  
  Resolves and standardizes entities by mapping them to canonical identifiers in external knowledge bases (e.g.,
  Wikidata, DBLP) or internal ontologies. Helps ensure consistent references across papers.

- **Graph Writer**  
  Builds and updates a knowledge graph representing relationships between papers, concepts, methodologies, and authors.
  The graph can be stored in Neo4j and queried for semantic search and visualization.

- **Memory Agent**  
  Maintains context, historical interactions, and learned facts across user sessions or agent invocations. This enables
  long-term reasoning and personalized interactions over time.

- **Summarizer**  
  Generates concise, human-readable summaries of academic papers or agent responses. Useful for building digestible
  outputs and feeding downstream agents with abstracted information.

- **Topic Model Agent**  
  Analyzes the latent themes across papers using unsupervised learning. Clusters documents into
  interpretable topics, aiding in organization, research gap discovery, and query routing.

## Pre-requisites

- Python Version: **3.11+**
- Dependencies: `requirements.txt`
- Docker / Docker Compose

## Setup

Create a `.env` file by copying the `.env.defaults` file:

```bash
cp .env.defaults .env
```

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

Run the following command to start the application:

```bash
docker compose up --build
```

The FastAPI server will be accessible at `http://localhost:8000`

## References

- [Multi-agent Systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)