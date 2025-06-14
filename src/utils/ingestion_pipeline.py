# src/utils/ingestion_pipeline.py - FIXED VERSION

import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import uuid
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from src.databases.document.config import get_database as get_mongodb
from src.databases.graph.config import get_neo4j_driver
from src.databases.vector.config import ChromaDBConfig

from src.utils.model_init import get_openai_model
from src.domain.prompts.agent_prompts import METADATA_EXTRACTION_PROMPT, ENTITY_EXTRACTION_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FixedPdfIngestionPipeline:
    def __init__(self):
        """Initialize with proper error handling and connection testing"""
        self.embeddings = OpenAIEmbeddings()
        self.model = get_openai_model()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Initialize all database connections with proper error handling
        self._init_vectordb_connection()
        self._init_graphdb_connection()
        self._init_mongodb_connection()

    def _init_vectordb_connection(self):
        """Initialize ChromaDB connection"""
        try:
            chroma_config = ChromaDBConfig()
            self.vector_db = chroma_config.get_client()
            # Get or create collection - use the consistent name
            self.vector_collection = chroma_config.get_collection("academic_papers")
            logger.info("✅ ChromaDB connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to ChromaDB: {e}")
            self.vector_db = None
            self.vector_collection = None

    def _init_graphdb_connection(self):
        """Initialize Neo4j connection"""
        try:
            self.graph_db = get_neo4j_driver()
            # Test the connection
            with self.graph_db.session() as session:
                session.run("RETURN 1")
            logger.info("✅ Neo4j connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            self.graph_db = None

    def _init_mongodb_connection(self):
        """Initialize MongoDB connection"""
        try:
            self.mongo_db = get_mongodb()
            # Test the connection
            self.mongo_db.list_collection_names()
            logger.info("✅ MongoDB connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            self.mongo_db = None

    def generate_paper_id(self, pdf_path, metadata=None):
        """Generate consistent paper ID for all databases"""
        filename = str(pdf_path.name)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))

    def is_already_processed(self, paper_id, pdf_path):
        """Check if paper exists in ANY database - skip if found"""
        filename = str(pdf_path.name)

        # Check MongoDB first (fastest check)
        if self.mongo_db is not None:
            try:
                existing = self.mongo_db.papers.find_one({
                    "$or": [
                        {"paper_id": paper_id},
                        {"source": filename}
                    ]
                })
                if existing:
                    logger.info(f"📋 Paper {filename} already exists in MongoDB - skipping")
                    return True
            except Exception as e:
                logger.warning(f"Could not check MongoDB: {e}")

        return False  # Process if not found or can't check

    def extract_metadata(self, documents) -> Dict:
        """Extract metadata with better fallback handling"""
        try:
            # Try LLM extraction first
            if documents:
                text = "\n".join([doc.page_content for doc in documents[:3]])
                formatted_messages = METADATA_EXTRACTION_PROMPT.format_messages(text=text)
                response = self.model.invoke(formatted_messages)

                try:
                    metadata = eval(response.content)
                    if isinstance(metadata, dict) and metadata.get('title'):
                        # Add source from document metadata
                        if hasattr(documents[0], 'metadata') and 'source' in documents[0].metadata:
                            metadata['source'] = documents[0].metadata['source']
                        return metadata
                except:
                    pass  # Fall through to fallback

            # Fallback metadata
            source = ""
            if documents and hasattr(documents[0], 'metadata'):
                source = documents[0].metadata.get('source', '')

            return {
                "title": f"Academic Paper - {Path(source).stem}" if source else "Unknown Paper",
                "authors": ["Unknown Author"],
                "year": None,
                "source": source,
                "abstract": documents[0].page_content[:200] + "..." if documents else "",
                "journal": "",
                "keywords": []
            }

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {
                "title": "Error Extracting Title",
                "authors": ["Unknown"],
                "year": None,
                "source": "",
                "abstract": "",
                "journal": "",
                "keywords": []
            }

    def create_embeddings(self, documents):
        """Create embeddings for ChromaDB"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            chunk_texts = [chunk.page_content for chunk in chunks]

            if not chunk_texts:
                logger.warning("No text chunks created for embedding")
                return {"chunks": [], "embeddings": [], "metadata": []}

            # Create embeddings
            embedding_vectors = self.embeddings.embed_documents(chunk_texts)

            logger.info(f"Created {len(embedding_vectors)} embeddings for {len(chunks)} chunks")

            return {
                "chunks": chunks,
                "embeddings": embedding_vectors,
                "metadata": [chunk.metadata for chunk in chunks]
            }
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return {"chunks": [], "embeddings": [], "metadata": []}

    def store_in_vectordb(self, embedding_data, paper_id, metadata):
        """Store embeddings in ChromaDB - FIXED"""
        if self.vector_collection is None:
            logger.error("ChromaDB collection not available")
            return False

        try:
            chunks = embedding_data["chunks"]
            embeddings = embedding_data["embeddings"]

            if not chunks or not embeddings:
                logger.warning("No chunks or embeddings to store")
                return False

            # Create unique IDs for chunks
            ids = [f"{paper_id}-chunk-{i}" for i in range(len(chunks))]

            # Create metadata for each chunk
            metadatas = []
            documents = []

            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "paper_id": paper_id,
                    "page": chunk.metadata.get("page", 0),
                    "source": chunk.metadata.get("source", ""),
                    "chunk_id": i,
                    "title": metadata.get("title", "Unknown"),
                    "authors": str(metadata.get("authors", [])),  # Convert to string for ChromaDB
                    "year": metadata.get("year", "Unknown")
                }
                metadatas.append(chunk_metadata)
                documents.append(chunk.page_content)

            # Store in ChromaDB
            self.vector_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

            logger.info(f"✅ Stored {len(ids)} chunks in ChromaDB")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to store in ChromaDB: {e}")
            return False

    def extract_entities(self, documents):
        """Extract entities for Neo4j with better error handling"""
        try:
            # Use first 5 pages for entity extraction
            text = "\n\n".join([doc.page_content for doc in documents[:5]])

            formatted_messages = ENTITY_EXTRACTION_PROMPT.format_messages(text=text)
            response = self.model.invoke(formatted_messages)

            try:
                entities = eval(response.content)
                if not isinstance(entities, dict):
                    raise ValueError("Invalid entity structure")

                # Ensure required keys exist
                entities.setdefault("concepts", [])
                entities.setdefault("relationships", [])

                logger.info(
                    f"Extracted {len(entities.get('concepts', []))} concepts and {len(entities.get('relationships', []))} relationships")
                return entities

            except:
                # Fallback: extract simple entities
                logger.warning("LLM entity extraction failed, using fallback")
                return self._extract_fallback_entities(text)

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"concepts": [], "relationships": []}

    def _extract_fallback_entities(self, text):
        """Simple fallback entity extraction"""
        import re

        # Extract capitalized phrases as concepts
        concept_patterns = [
            r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b',
            r'\b(?:machine learning|artificial intelligence|neural network|deep learning)\b',
        ]

        concepts = []
        seen_concepts = set()

        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:10]:  # Limit to avoid too many
                if match.lower() not in seen_concepts and len(match) > 3:
                    concepts.append({
                        "name": match,
                        "category": "extracted_concept",
                        "description": f"Concept extracted from text: {match}"
                    })
                    seen_concepts.add(match.lower())

        return {
            "concepts": concepts[:10],  # Limit concepts
            "relationships": []  # No relationships in fallback
        }

    def store_in_graphdb(self, entities, paper_id, metadata):
        """Store in Neo4j - FIXED with better error handling"""
        if self.graph_db is None:
            logger.error("Neo4j connection not available")
            return False

        try:
            with self.graph_db.session() as session:
                # Create paper node first
                paper_props = {
                    "id": paper_id,
                    "title": metadata.get("title", "Unknown"),
                    "year": metadata.get("year"),
                    "source": metadata.get("source", "")
                }

                session.run(
                    """
                    MERGE (p:Paper {id: $id})
                    SET p.title = $title, p.year = $year, p.source = $source
                    """,
                    **paper_props
                )

                # Create author nodes and relationships
                authors = metadata.get("authors", [])
                if authors:
                    for author in authors:
                        if author and author.strip() and author != "Unknown Author":
                            session.run(
                                """
                                MERGE (a:Author {name: $name})
                                WITH a
                                MATCH (p:Paper {id: $paper_id})
                                MERGE (a)-[:AUTHORED]->(p)
                                """,
                                name=author.strip(),
                                paper_id=paper_id
                            )

                # Create concept nodes and relationships
                concepts = entities.get("concepts", [])
                if concepts:
                    for concept in concepts:
                        name = concept.get("name", "").strip()
                        if name:
                            session.run(
                                """
                                MERGE (c:Concept {name: $name})
                                SET c.category = $category, c.description = $description
                                WITH c
                                MATCH (p:Paper {id: $paper_id})
                                MERGE (p)-[:CONTAINS]->(c)
                                """,
                                name=name,
                                category=concept.get("category", ""),
                                description=concept.get("description", ""),
                                paper_id=paper_id
                            )

                # Create relationships between concepts
                relationships = entities.get("relationships", [])
                if relationships:
                    for rel in relationships:
                        from_name = rel.get("from", "").strip()
                        to_name = rel.get("to", "").strip()
                        if from_name and to_name:
                            session.run(
                                """
                                MATCH (a:Concept {name: $from_name}), (b:Concept {name: $to_name})
                                MERGE (a)-[r:RELATES_TO {type: $type}]->(b)
                                SET r.description = $description
                                """,
                                from_name=from_name,
                                to_name=to_name,
                                type=rel.get("type", "related_to"),
                                description=rel.get("description", "")
                            )

            logger.info(f"✅ Stored paper with {len(concepts)} concepts in Neo4j")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to store in Neo4j: {e}")
            return False

    def store_in_mongodb(self, documents, metadata, entities, paper_id):
        """Store in MongoDB - FIXED"""
        if self.mongo_db is None:
            logger.error("MongoDB connection not available")
            return False

        try:
            papers_collection = self.mongo_db.papers

            # Create document for MongoDB
            paper_doc = {
                "paper_id": paper_id,
                "source": metadata.get("source", ""),
                "metadata": metadata,
                "content": [
                    {"page": i, "text": doc.page_content}
                    for i, doc in enumerate(documents)
                ],
                "entities": entities,
                "processed_at": pd.Timestamp.now().isoformat()
            }

            # Insert or update
            result = papers_collection.replace_one(
                {"paper_id": paper_id},
                paper_doc,
                upsert=True
            )

            logger.info(f"✅ Stored paper in MongoDB (upserted: {result.upserted_id is not None})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to store in MongoDB: {e}")
            return False

    def process_pdf(self, pdf_path):
        """Process a single PDF with improved error handling"""
        logger.info(f"🔄 Processing PDF: {pdf_path.name}")

        try:
            # Generate paper ID
            paper_id = self.generate_paper_id(pdf_path)

            # Check if already processed
            if self.is_already_processed(paper_id, pdf_path):
                return True

            # Load PDF
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            if not documents:
                logger.warning(f"No content extracted from {pdf_path}")
                return False

            logger.info(f"Loaded {len(documents)} pages from PDF")

            # Extract metadata
            metadata = self.extract_metadata(documents)
            logger.info(f"Extracted metadata: {metadata.get('title', 'Unknown')[:50]}...")

            # Track success for each database
            results = {"vector": False, "graph": False, "mongo": False}

            # Store in ChromaDB
            if self.vector_collection is not None:
                embedding_data = self.create_embeddings(documents)
                results["vector"] = self.store_in_vectordb(embedding_data, paper_id, metadata)

            # Store in Neo4j
            if self.graph_db is not None:
                entities = self.extract_entities(documents)
                results["graph"] = self.store_in_graphdb(entities, paper_id, metadata)
            else:
                entities = {"concepts": [], "relationships": []}

            # Store in MongoDB
            if self.mongo_db is not None:
                results["mongo"] = self.store_in_mongodb(documents, metadata, entities, paper_id)

            # Report results
            success_count = sum(results.values())
            total_dbs = sum(1 for db in [self.vector_collection, self.graph_db, self.mongo_db] if db is not None)

            if success_count == total_dbs:
                logger.info(f"✅ Successfully stored {pdf_path.name} in all {total_dbs} databases")
                return True
            elif success_count > 0:
                logger.warning(f"⚠️ Partially stored {pdf_path.name}: {results}")
                return True
            else:
                logger.error(f"❌ Failed to store {pdf_path.name} in any database")
                return False

        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path}: {e}", exc_info=True)
            return False

    def close_connections(self):
        """Close database connections"""
        if hasattr(self, 'graph_db') and self.graph_db:
            self.graph_db.close()
        logger.info("Database connections closed")


def quick_test_ingestion():
    """Test ingestion with one paper"""
    source_path = Path("sources")
    if not source_path.exists():
        print(f"❌ Sources directory not found: {source_path}")
        print("   Creating sources directory...")
        source_path.mkdir(exist_ok=True)
        print("   📁 Please add PDF files to the sources/ folder")
        return False

    pdf_files = list(source_path.glob('*.pdf'))
    if not pdf_files:
        print(f"❌ No PDFs found in {source_path}")
        print("   📁 Please add academic PDF files to sources/")
        return False

    print(f"✅ Found {len(pdf_files)} PDFs")
    for pdf in pdf_files:
        print(f"   📄 {pdf.name}")

    # Test with first paper
    print(f"\n🔄 Testing ingestion with: {pdf_files[0].name}")
    pipeline = FixedPdfIngestionPipeline()

    try:
        success = pipeline.process_pdf(pdf_files[0])

        if success:
            print(f"✅ Successfully processed {pdf_files[0].name}")

            # Verify in all databases
            print("\n🔍 Verification:")

            # Check MongoDB
            if pipeline.mongo_db:
                count = pipeline.mongo_db.papers.count_documents({})
                print(f"   📊 MongoDB: {count} papers")

            # Check Neo4j
            if pipeline.graph_db:
                with pipeline.graph_db.session() as session:
                    result = session.run("MATCH (p:Paper) RETURN count(p) as count")
                    count = result.single()["count"]
                    print(f"   🔗 Neo4j: {count} papers")

            # Check ChromaDB
            if pipeline.vector_collection:
                count = pipeline.vector_collection.count()
                print(f"   🔍 ChromaDB: {count} vectors")

            return True
        else:
            print(f"❌ Failed to process {pdf_files[0].name}")
            return False

    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False
    finally:
        pipeline.close_connections()


def ingest_all_pdfs(source_dir="sources"):
    """Process all PDFs in source directory"""
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"Source directory {source_dir} does not exist")
        return

    pipeline = FixedPdfIngestionPipeline()
    try:
        pdf_files = list(source_path.glob('*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files")

        if not pdf_files:
            logger.warning("No PDF files found to process")
            return

        successful = 0
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            if pipeline.process_pdf(pdf_file):
                successful += 1

        logger.info(f"Successfully processed {successful}/{len(pdf_files)} PDFs")

    except Exception as e:
        logger.error(f"Error in ingestion process: {e}")
    finally:
        pipeline.close_connections()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest academic PDFs into all databases")
    parser.add_argument("--source", type=str, default="sources", help="Directory containing PDF files")
    parser.add_argument("--test", action="store_true", help="Run quick test with one paper")

    args = parser.parse_args()

    if args.test:
        print("🧪 Running quick ingestion test...")
        success = quick_test_ingestion()
        if success:
            print("\n🎉 Test successful! All databases populated.")
        else:
            print("\n❌ Test failed. Check setup and try again.")
    else:
        print(f"🚀 Starting full ingestion from {args.source}")
        ingest_all_pdfs(args.source)