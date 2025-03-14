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