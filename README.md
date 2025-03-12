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