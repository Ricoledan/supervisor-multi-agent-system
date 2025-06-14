#!/bin/bash
# Advanced Test Queries for Your Multi-Agent System

echo "🧪 Testing Multi-Agent System with Advanced Queries"
echo "================================================"

# Test 1: Graph-focused query
echo ""
echo "🔍 Test 1: Graph Relationships Query"
curl -s -X POST http://localhost:8000/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me the connections between reinforcement learning and critic models in the research papers"}' | jq -r '.message'

echo ""
echo "---"

# Test 2: Concept exploration
echo ""
echo "🔍 Test 2: Deep Learning Concepts"
curl -s -X POST http://localhost:8000/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main deep learning approaches mentioned in the papers?"}' | jq -r '.message'

echo ""
echo "---"

# Test 3: Author and methodology focus
echo ""
echo "🔍 Test 3: Research Methodology Analysis"
curl -s -X POST http://localhost:8000/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze the different critique methodologies across the research papers"}' | jq -r '.message'

echo ""
echo "---"

# Test 4: Comparative analysis
echo ""
echo "🔍 Test 4: Comparative Research Analysis"
curl -s -X POST http://localhost:8000/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare the approaches to language model training across different papers"}' | jq -r '.message'

echo ""
echo "================================="
echo "✅ Multi-Agent System Test Complete"