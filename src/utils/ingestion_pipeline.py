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


class PdfIngestionPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.model = get_openai_model()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        self._init_vectordb_connection()
        self._init_graphdb_connection()
        self._init_mongodb_connection()

    def _init_vectordb_connection(self):
        try:
            chroma_config = ChromaDBConfig()
            self.vector_db = chroma_config.get_client()
            self.vector_collection = self.vector_db.get_or_create_collection("academic_papers")
            logger.info("Vector DB connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Vector DB: {e}")
            self.vector_db = None

    def _init_graphdb_connection(self):
        try:
            self.graph_db = get_neo4j_driver()
            logger.info("Graph DB connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.graph_db = None

    def _init_mongodb_connection(self):
        try:
            self.mongo_db = get_mongodb()
            logger.info("MongoDB connection established")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.mongo_db = None

    def generate_paper_id(self, pdf_path, metadata=None):
        """Generate consistent paper ID for all databases"""
        # Use filename as primary identifier for consistency
        filename = str(pdf_path.name)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))

    def is_already_processed(self, paper_id, pdf_path):
        """Check if paper is already processed in ANY database"""
        filename = str(pdf_path.name)

        # Check MongoDB first (fastest)
        if self.mongo_db is not None:
            existing_mongo = self.mongo_db.papers.find_one({
                "$or": [
                    {"paper_id": paper_id},
                    {"source": filename}
                ]
            })
            if existing_mongo:
                logger.info(f"Paper {filename} already exists in MongoDB")
                return True

        # Check Neo4j
        if self.graph_db is not None:
            try:
                with self.graph_db.session() as session:
                    result = session.run("MATCH (p:Paper {id: $paper_id}) RETURN p", paper_id=paper_id)
                    if result.single():
                        logger.info(f"Paper {filename} already exists in Neo4j")
                        return True
            except Exception as e:
                logger.warning(f"Could not check Neo4j for duplicates: {e}")

        # Check ChromaDB
        if self.vector_db is not None:
            try:
                # Query for existing chunks with this paper_id
                existing_chunks = self.vector_collection.get(
                    where={"paper_id": paper_id},
                    limit=1
                )
                if existing_chunks['ids']:
                    logger.info(f"Paper {filename} already exists in ChromaDB")
                    return True
            except Exception as e:
                logger.warning(f"Could not check ChromaDB for duplicates: {e}")

        return False

    def extract_metadata(self, documents) -> Dict:
        """Extract structured metadata from document using LLM"""
        try:
            if documents and hasattr(documents[0], 'metadata'):
                pdf_metadata = documents[0].metadata
                # Use PDF metadata if available
                if pdf_metadata.get('title') and pdf_metadata.get('author'):
                    return {
                        'title': pdf_metadata.get('title'),
                        'authors': [pdf_metadata.get('author')],
                        'year': pdf_metadata.get('creationDate', '').split('-')[0] if '-' in pdf_metadata.get(
                            'creationDate', '') else None,
                        'source': pdf_metadata.get('source', ''),
                        'doi': pdf_metadata.get('doi', ''),
                        'abstract': '',
                        'journal': '',
                        'keywords': []
                    }

            # Use LLM to extract metadata from text
            text = "\n".join([doc.page_content for doc in documents[:3]])
            formatted_messages = METADATA_EXTRACTION_PROMPT.format_messages(text=text)
            response = self.model.invoke(formatted_messages)

            try:
                # Try to parse LLM response as dict
                metadata = eval(response.content)
            except:
                # Fallback: create basic metadata from filename
                logger.warning("Failed to parse LLM metadata, using fallback")
                metadata = {
                    "title": "Unknown Paper",
                    "authors": ["Unknown Author"],
                    "year": None,
                    "journal": "",
                    "abstract": text[:200] + "..." if len(text) > 200 else text,
                    "doi": "",
                    "keywords": []
                }

            # Always add source from document metadata
            if documents and hasattr(documents[0], 'metadata') and 'source' in documents[0].metadata:
                metadata['source'] = documents[0].metadata['source']

            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            # Return safe fallback
            source = ""
            if documents and hasattr(documents[0], 'metadata'):
                source = documents[0].metadata.get('source', '')
            return {
                "title": f"Paper from {Path(source).name}" if source else "Unknown Paper",
                "authors": ["Unknown Author"],
                "year": None,
                "source": source,
                "abstract": "",
                "journal": "",
                "keywords": []
            }

    def create_embeddings(self, documents):
        try:
            chunks = self.text_splitter.split_documents(documents)
            chunk_texts = [chunk.page_content for chunk in chunks]
            embedding_vectors = self.embeddings.embed_documents(chunk_texts)
            return {
                "chunks": chunks,
                "embeddings": embedding_vectors,
                "metadata": [chunk.metadata for chunk in chunks]
            }
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return {"chunks": [], "embeddings": [], "metadata": []}

    def store_in_vectordb(self, embedding_data, paper_id):
        if self.vector_db is None:
            logger.error("Vector DB connection not available")
            return False

        try:
            chunks = embedding_data["chunks"]
            embeddings = embedding_data["embeddings"]
            if not chunks:
                logger.warning("No chunks to store in vector DB")
                return False

            # Create consistent chunk IDs to prevent duplicates
            ids = [f"{paper_id}-chunk-{i}" for i in range(len(chunks))]

            # Check if any chunks already exist
            try:
                existing = self.vector_collection.get(ids=ids[:1])  # Check first ID
                if existing['ids']:
                    logger.info("Chunks already exist in ChromaDB, skipping vector storage")
                    return True
            except:
                pass  # Continue if check fails

            metadatas = []
            documents = []
            for i, chunk in enumerate(chunks):
                metadatas.append({
                    "paper_id": paper_id,
                    "page": chunk.metadata.get("page", 0),
                    "source": chunk.metadata.get("source", ""),
                    "chunk_id": i,
                    "title": embedding_data.get("metadata", {}).get("title", "")
                })
                documents.append(chunk.page_content)

            # Use upsert to handle potential duplicates
            self.vector_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Stored/updated {len(ids)} chunks in Vector DB")
            return True
        except Exception as e:
            logger.error(f"Failed to store in Vector DB: {e}")
            return False

    def extract_entities(self, documents):
        try:
            # Combine first 10 pages of text for entity extraction
            text = "\n\n".join([doc.page_content for doc in documents[:10]])
            formatted_messages = ENTITY_EXTRACTION_PROMPT.format_messages(text=text)
            response = self.model.invoke(formatted_messages)

            try:
                entities = eval(response.content)
                # Ensure proper structure
                if not isinstance(entities, dict):
                    raise ValueError("Invalid entity structure")
                if "concepts" not in entities:
                    entities["concepts"] = []
                if "relationships" not in entities:
                    entities["relationships"] = []
            except:
                logger.warning("Failed to parse entity extraction, using fallback")
                # Create simple fallback entities from text
                entities = self._extract_fallback_entities(text)

            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"concepts": [], "relationships": []}

    def _extract_fallback_entities(self, text):
        """Simple keyword-based entity extraction as fallback"""
        import re

        # Simple regex patterns for common academic concepts
        concept_patterns = [
            r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b',  # Capitalized phrases
            r'\b(?:machine learning|artificial intelligence|neural network|deep learning)\b',
            r'\b(?:algorithm|model|method|approach|technique)\b',
        ]

        concepts = []
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:10]:  # Limit to 10 concepts
                concepts.append({
                    "name": match,
                    "category": "extracted_concept",
                    "description": f"Concept extracted from text: {match}"
                })

        return {
            "concepts": concepts[:10],  # Limit concepts
            "relationships": []  # Keep relationships empty for fallback
        }

    def store_in_graphdb(self, entities, paper_id, metadata=None):
        if self.graph_db is None:
            logger.error("Graph DB connection not available")
            return False
        try:
            with self.graph_db.session() as session:
                # Check if paper already exists
                result = session.run("MATCH (p:Paper {id: $paper_id}) RETURN p", paper_id=paper_id)
                if result.single():
                    logger.info("Paper already exists in Neo4j, skipping graph storage")
                    return True

                # Create paper node
                paper_props = {
                    "id": paper_id,
                    "title": metadata.get("title", "Unknown") if metadata else "Unknown",
                    "year": metadata.get("year") if metadata else None
                }

                session.run(
                    """
                    MERGE (p:Paper {id: $id})
                    SET p.title = $title, p.year = $year
                    RETURN p
                    """,
                    **paper_props
                )

                # Create author nodes and relationships
                if metadata and "authors" in metadata:
                    for author in metadata["authors"]:
                        if author and author.strip():  # Only if author is not empty
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
                for concept in entities.get("concepts", []):
                    if concept.get("name"):  # Only if concept has a name
                        session.run(
                            """
                            MERGE (c:Concept {name: $name})
                            SET c.category = $category, c.description = $description
                            WITH c
                            MATCH (p:Paper {id: $paper_id})
                            MERGE (p)-[:CONTAINS]->(c) 
                            RETURN c
                            """,
                            name=concept.get("name"),
                            category=concept.get("category", ""),
                            description=concept.get("description", ""),
                            paper_id=paper_id
                        )

                # Create relationships between concepts
                for rel in entities.get("relationships", []):
                    if rel.get("from") and rel.get("to"):  # Only if both ends exist
                        session.run(
                            """
                            MATCH (a:Concept {name: $from_name}), (b:Concept {name: $to_name})
                            MERGE (a)-[r:RELATES_TO {type: $type}]->(b)
                            SET r.description = $description
                            """,
                            from_name=rel.get("from"),
                            to_name=rel.get("to"),
                            type=rel.get("type", "related_to"),
                            description=rel.get("description", "")
                        )

            logger.info(
                f"Stored {len(entities.get('concepts', []))} concepts and {len(entities.get('relationships', []))} relationships in Neo4j")
            return True
        except Exception as e:
            logger.error(f"Failed to store in Neo4j: {e}")
            return False

    def store_in_mongodb(self, documents, metadata, entities, paper_id):
        if self.mongo_db is None:
            logger.error("MongoDB connection not available")
            return False
        try:
            paper_collection = self.mongo_db.papers

            # Check if already exists
            existing = paper_collection.find_one({"paper_id": paper_id})
            if existing:
                logger.info("Paper already exists in MongoDB, skipping document storage")
                return True

            source = ""
            if documents and hasattr(documents[0], 'metadata') and 'source' in documents[0].metadata:
                source = documents[0].metadata['source']

            paper_doc = {
                "paper_id": paper_id,
                "source": source,
                "metadata": metadata or {},
                "content": [{"page": i, "text": doc.page_content} for i, doc in enumerate(documents)],
                "entities": entities or {},
                "processed_at": pd.Timestamp.now()
            }

            result = paper_collection.insert_one(paper_doc)
            logger.info(f"Stored document in MongoDB with ID: {result.inserted_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store in MongoDB: {e}")
            return False

    def process_pdf(self, pdf_path):
        logger.info(f"Processing PDF: {pdf_path}")

        try:
            # Generate consistent paper ID
            paper_id = self.generate_paper_id(pdf_path)

            # Check if already processed in ANY database
            if self.is_already_processed(paper_id, pdf_path):
                logger.info(f"✅ Skipping {pdf_path.name} - already processed in databases")
                return True

            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            if not documents:
                logger.warning(f"No content extracted from {pdf_path}")
                return False

            # Extract metadata with better error handling
            metadata = self.extract_metadata(documents)
            logger.info(f"Extracted metadata: {metadata.get('title', 'Unknown')}")

            # Create embeddings
            embedding_data = self.create_embeddings(documents)
            embedding_data["metadata"] = metadata

            # Store in databases
            vector_success = self.store_in_vectordb(embedding_data, paper_id)

            entities = self.extract_entities(documents)
            graph_success = self.store_in_graphdb(entities, paper_id, metadata)

            mongo_success = self.store_in_mongodb(documents, metadata, entities, paper_id)

            if vector_success or graph_success or mongo_success:
                logger.info(f"✅ Successfully processed {pdf_path}")
                return True
            else:
                logger.error(f"❌ Failed to store {pdf_path} in any database")
                return False

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return False

    def close_connections(self):
        """Close any open connections"""
        if self.graph_db is not None:
            self.graph_db.close()
        logger.info("Database connections closed")


def quick_test_ingestion():
    """Test ingestion with just one paper - for article demo"""
    source_path = Path("sources")
    if not source_path.exists():
        print(f"❌ Sources directory not found: {source_path}")
        return False

    pdf_files = list(source_path.glob('*.pdf'))

    if not pdf_files:
        print(f"❌ No PDFs found in {source_path}")
        return False

    print(f"✅ Found {len(pdf_files)} PDFs:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    # Test with first paper only
    print(f"\n🔄 Testing ingestion with: {pdf_files[0].name}")
    pipeline = PdfIngestionPipeline()

    try:
        success = pipeline.process_pdf(pdf_files[0])

        if success:
            print(f"✅ Successfully processed {pdf_files[0].name}")

            # Verify it's in MongoDB
            if pipeline.mongo_db is not None:
                papers = list(pipeline.mongo_db.papers.find().limit(1))
                if papers:
                    title = papers[0].get('metadata', {}).get('title', 'Unknown')
                    print(f"✅ Found in MongoDB: {title}")
                else:
                    print("⚠️  Not found in MongoDB")

            # Check Neo4j
            if pipeline.graph_db is not None:
                with pipeline.graph_db.session() as session:
                    result = session.run("MATCH (p:Paper) RETURN count(p) as count")
                    count = result.single()["count"]
                    print(f"✅ Found {count} papers in Neo4j")

            # Check ChromaDB
            if pipeline.vector_db is not None:
                collection = pipeline.vector_collection
                result = collection.count()
                print(f"✅ Found {result} chunks in ChromaDB")

            return True
        else:
            print(f"❌ Failed to process {pdf_files[0].name}")
            return False

    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False
    finally:
        pipeline.close_connections()


def ingest_pdfs(source_dir):
    """Process all PDFs in source directory"""
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"Source directory {source_dir} does not exist")
        return

    pipeline = PdfIngestionPipeline()
    try:
        pdf_files = list(source_path.glob('*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files in {source_dir}")

        if not pdf_files:
            logger.warning("No PDF files found to process")
            return

        successful = 0
        skipped = 0
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            if pipeline.process_pdf(pdf_file):
                successful += 1
            else:
                skipped += 1

        logger.info(f"Successfully processed {successful}/{len(pdf_files)} PDFs ({skipped} skipped)")

    except Exception as e:
        logger.error(f"Error in ingestion process: {e}")
    finally:
        pipeline.close_connections()

    logger.info("PDF ingestion complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest academic PDFs into databases")
    parser.add_argument("--source", type=str, default="sources", help="Directory containing PDF files")
    parser.add_argument("--test", action="store_true", help="Run quick test with one paper")

    args = parser.parse_args()

    if args.test:
        print("🧪 Running quick ingestion test...")
        success = quick_test_ingestion()
        if success:
            print("\n🎉 Test successful! Ready for article demo.")
        else:
            print("\n❌ Test failed. Check your setup.")
    else:
        ingest_pdfs(args.source)