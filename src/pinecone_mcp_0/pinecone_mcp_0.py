from fastmcp import FastMCP
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import uuid
import os
from pydantic import BaseModel, Field
from typing import Optional, Dict, Type

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

mcp = FastMCP("Pinecone")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("text-embedding-3-small-index")

class FirstNamespaceSchema(BaseModel):
    author: str = Field(description="Author of the document")
    file_id: int = Field(description="Unique identifier for the file")
    industry: str = Field(description="Industry category the document belongs to")
    original_text: str = Field(description="The original text content of the document")

    class Config:
        extra = "forbid"
# Add additional schemas here
NAMESPACE_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "first_namespace": FirstNamespaceSchema
    # Add more namespaces as needed
}

def get_namespace_schema(namespace: str) -> Type[BaseModel]:
    try:
        return NAMESPACE_SCHEMAS[namespace]
    except KeyError:
        raise ValueError(f"Unknown namespace: {namespace}")

@mcp.tool()
def embed(query_text: str) -> list[float] | None:
    client = OpenAI()

    # Specify the model (can also be done once if always using the same model)
    model_name = "text-embedding-3-small"
    """
    Generates an embedding vector for the given text using the specified OpenAI model.

    Args:
        query_text: The text to embed.

    Returns:
        A list of floats representing the embedding vector, or None if an error occurs.
    """
    try:
        # Call the OpenAI Embeddings API
        response = client.embeddings.create(
            input=query_text,
            model=model_name
        )
        # Extract the embedding vector
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        return None
    
@mcp.tool()
def search_pinecone(query_text: str, namespace: str, filter: dict) -> list[str] | None:
# def search_pinecone(query_text: str, namespace: str) -> list[str] | None:
    """
    Searches Pinecone for the most relevant documents based on the given query text within a specific namespace and applying a metadata filter 
    (if there is no filter, input the filter as an empty dictionary).

    Args:
        query_text: The text to search for.
        namespace: The namespace to search within.
        filter: A dictionary representing the metadata filter to apply (optional).

    Returns:
        A list of strings representing the most relevant documents, or None if an error occurs.
    """
    try:
        # Embed the query text
        query_embedding = embed(query_text)
        if query_embedding is None:
            return ["no query embedding"]

        results = index.query(
            namespace=namespace,
            top_k=3,
            include_metadata=True,
            vector=query_embedding,
            filter=filter
        )

        key_data = []
        for match in results.get('matches', []):
            key_data.append({
                'id': match.get('id'),
                'score': match.get('score'),
                'metadata': match.get('metadata', {})
            })

        return key_data if key_data else "No matches found"
    except Exception as e:
        print(f"An error occurred during Pinecone search: {e}")
        return [f"An error occurred during Pinecone search: {e}"]

@mcp.tool()
def insert_text(namespace: str, data: FirstNamespaceSchema) -> bool:
    """
    Inserts a text into a specific namespace with the given metadata.

    Args:
        text: The text to insert.
        namespace: The namespace to insert the text into.
        metadata: A dictionary containing the metadata for the text.

    Returns:
        True if the text was inserted successfully, False otherwise.
    """
    try:
        # Embed the text
        text_embedding = embed(data.original_text)
        if text_embedding is None:
            return False

        # Insert the text into Pinecone
        index.upsert(
            vectors=[{
                "id": str(uuid.uuid4()),
                "values": text_embedding,
                "metadata": data.model_dump()
            }],
            namespace=namespace
        )
        return True
    except Exception as e:
        print(f"An error occurred during text insertion: {e}")
        return False
    
    