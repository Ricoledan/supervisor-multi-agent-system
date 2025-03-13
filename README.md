# supervisor-multi-agent-system

## Overview

This project implements a Hierarchical Multi-Agent System (MAS) architecture featuring a central supervisor agent that
orchestrates specialized subordinate agents, served through a FastAPI interface.

## Key Features

- **Task Delegation and Coordination**: Supports architectures where a central supervisor agent assigns tasks to
  specialized subordinate agents, monitors their progress, and ensures alignment with overarching objectives.

- **Performance Monitoring**: By evaluating the outputs and behaviors of subordinate agents, a supervisor agent can
  maintain quality and efficiency, intervening when necessary to correct deviations.

- **Decision Support**: Integrating insights from multiple agents, the supervisor provides high-level decision-making
  assistance, optimizing workflow execution and improving overall system intelligence.

- **Communication Management**: Facilitates structured communication flows between agents, enabling smooth
  inter-agent collaboration.

- **Hierarchical Structuring**: Supports multi-level hierarchies, allowing a supervisor to manage other
  supervisors, facilitating scalable, distributed, and complex AI-driven workflows.

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

- **Langchain**: a framework for developing applications powered by large language models. It simplifies the
  entire LLM application lifecycle by providing a standard interface for models, embeddings, vector stores, and more,
  allowing
  developers to build context-aware reasoning applications.

- **LangGraph**: a library designed for building stateful, multi-actor applications with LLMs, facilitating the creation
  of agent and multi-agent workflows. It offers fine-grained control over application flow and state, enabling features
  like memory persistence and human-in-the-loop interactions.

- **Neo4j**: a leading open-source graph database management system implemented in Java. It stores data as nodes,
  relationships, and properties, making it ideal for representing complex, interconnected information.

- **Faiss**: a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search
  in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for
  evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. Some of the most
  useful algorithms are implemented on the GPU. It is developed primarily at Meta's Fundamental AI Research group.

- **ChromaDB**: an open-source vector database optimized for applications utilizing large language models. It
  efficiently stores and retrieves vector embeddings, facilitating tasks like semantic search and similarity matching.

- **PostgresSQL**: a powerful, open-source object-relational database management system known for
  its robustness, extensibility, and compliance with SQL standards. It supports advanced data types, indexing methods,
  and offers features like transactions, foreign keys, views, triggers, and stored procedures.

- **Docker**: an open-source platform that automates the deployment, scaling, and management of applications using
  containerization. Containers are lightweight, portable units that package an application and its dependencies,
  ensuring consistent behavior across different environments.

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