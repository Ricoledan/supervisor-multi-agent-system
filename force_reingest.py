#!/usr/bin/env python3
"""
Force re-ingestion to populate MongoDB and ChromaDB
"""

import logging
from pathlib import Path
from src.databases.graph.config import get_neo4j_driver
from src.databases.vector.config import ChromaDBConfig
from src.databases.document.config import get_database
from src.utils.ingestion_pipeline import PdfIngestionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_chromadb():
    """Clear ChromaDB collections"""
    try:
        config = ChromaDBConfig()
        client = config.get_client()
        
        # Get existing collections
        try:
            collection = client.get_collection("academic_papers")
            client.delete_collection("academic_papers")
            logger.info("Cleared ChromaDB academic_papers collection")
        except:
            logger.info("ChromaDB collection didn't exist")
        
        # Create fresh collection
        collection = client.create_collection("academic_papers")
        logger.info("Created fresh ChromaDB collection")
        
    except Exception as e:
        logger.error(f"Error clearing ChromaDB: {e}")

def clear_mongodb():
    """Clear MongoDB collections"""
    try:
        db = get_database()
        
        # Drop existing collections
        collections = db.list_collection_names()
        for collection_name in collections:
            db.drop_collection(collection_name)
            logger.info(f"Dropped MongoDB collection: {collection_name}")
        
        logger.info("Cleared MongoDB database")
        
    except Exception as e:
        logger.error(f"Error clearing MongoDB: {e}")

def force_reingest():
    """Force re-ingestion of all papers"""
    logger.info("Starting forced re-ingestion...")
    
    # Clear existing data in MongoDB and ChromaDB
    clear_mongodb()
    clear_chromadb()
    
    # Process all PDFs
    source_path = Path("/app/sources")
    if not source_path.exists():
        logger.error("Sources directory not found")
        return
    
    pdf_files = list(source_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    pipeline = PdfIngestionPipeline()
    
    try:
        for pdf_file in pdf_files:
            logger.info(f"Force processing: {pdf_file.name}")
            
            # Generate paper ID
            paper_id = pipeline.generate_paper_id(pdf_file)
            
            # Process the PDF (this will skip Neo4j but populate MongoDB and ChromaDB)
            success = pipeline.process_pdf(pdf_file)
            
            if success:
                logger.info(f"✅ Successfully processed {pdf_file.name}")
            else:
                logger.error(f"❌ Failed to process {pdf_file.name}")
    
    except Exception as e:
        logger.error(f"Error during re-ingestion: {e}")
    finally:
        pipeline.close_connections()
    
    logger.info("Force re-ingestion complete")

if __name__ == "__main__":
    force_reingest()
