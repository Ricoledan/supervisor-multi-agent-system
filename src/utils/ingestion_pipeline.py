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

            text = "\n".join([doc.page_content for doc in documents[:3]])
            formatted_messages = METADATA_EXTRACTION_PROMPT.format_messages(text=text)
            response = self.model.invoke(formatted_messages)
            metadata = eval(response.content)
            if documents and hasattr(documents[0], 'metadata') and 'source' in documents[0].metadata:
                metadata['source'] = documents[0].metadata['source']
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {"title": "Unknown", "authors": [], "year": None, "source": ""}

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
            ids = [f"{paper_id}-chunk-{i}" for i in range(len(chunks))]
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
            self.vector_collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Stored {len(ids)} chunks in Vector DB")
            return True
        except Exception as e:
            logger.error(f"Failed to store in Vector DB: {e}")
            return False

    def extract_entities(self, documents):
        try:
            text = "\n\n".join([doc.page_content for doc in documents[:10]])
            formatted_messages = ENTITY_EXTRACTION_PROMPT.format_messages(text=text)
            response = self.model.invoke(formatted_messages)
            try:
                entities = eval(response.content)
            except:
                entities = {"concepts": [], "relationships": []}
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"concepts": [], "relationships": []}

    def store_in_graphdb(self, entities, paper_id, metadata=None):
        if self.graph_db is None:
            logger.error("Graph DB connection not available")
            return False
        try:
            with self.graph_db.session() as session:
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

                if metadata and "authors" in metadata:
                    for author in metadata["authors"]:
                        session.run(
                            """
                            MERGE (a:Author {name: $name})
                            WITH a
                            MATCH (p:Paper {id: $paper_id})
                            MERGE (a)-[:AUTHORED]->(p)
                            """,
                            name=author,
                            paper_id=paper_id
                        )

                for concept in entities.get("concepts", []):
                    session.run(
                        """
                        MERGE (c:Concept {name: $name})
                        SET c.category = $category, c.description = $description
                        WITH c
                        MATCH (p:Paper {id: $paper_id})
                        MERGE (p)-[:CONTAINS]->(c) RETURN c
                        """,
                        name=concept.get("name"),
                        category=concept.get("category", ""),
                        description=concept.get("description", ""),
                        paper_id=paper_id
                    )

                for rel in entities.get("relationships", []):
                    session.run(
                        """
                        MATCH (a:Concept {name: $from}), (b:Concept {name: $to})
                        MERGE (a)-[r:RELATES_TO {type: $type}]->(b)
                        SET r.description = $description
                        """,
                        from_=rel.get("from"),
                        to=rel.get("to"),
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

        if self.mongo_db is not None and self.mongo_db.papers.find_one({"source": str(pdf_path.name)}):
            logger.info(f"Skipping {pdf_path.name}, already ingested.")
            return True

        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            if not documents:
                logger.warning(f"No content extracted from {pdf_path}")
                return False

            metadata = self.extract_metadata(documents)

            # Generate a unique ID based on filename/title instead of using PostgreSQL ID
            title = metadata.get("title", "")
            filename = metadata.get("source", "")
            paper_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{title}:{filename}"))

            embedding_data = self.create_embeddings(documents)
            embedding_data["metadata"] = metadata
            self.store_in_vectordb(embedding_data, paper_id)

            entities = self.extract_entities(documents)
            self.store_in_graphdb(entities, paper_id, metadata)

            self.store_in_mongodb(documents, metadata, entities, paper_id)

            logger.info(f"Successfully processed {pdf_path}")
            return True
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return False

    def close_connections(self):
        """Close any open connections"""
        if self.graph_db is not None:
            self.graph_db.close()
        logger.info("Database connections closed")


def ingest_pdfs(source_dir):
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"Source directory {source_dir} does not exist")
        return
    pipeline = PdfIngestionPipeline()
    try:
        pdf_files = list(source_path.glob('*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files in {source_dir}")
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pipeline.process_pdf(pdf_file)
    except Exception as e:
        logger.error(f"Error in ingestion process: {e}")
    finally:
        pipeline.close_connections()
    logger.info("PDF ingestion complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest academic PDFs into databases")
    parser.add_argument("--source", type=str, required=True, help="Directory containing PDF files")
    args = parser.parse_args()
    ingest_pdfs(args.source)