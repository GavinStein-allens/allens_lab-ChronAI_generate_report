import azure.functions as func
import numpy as np
from langchain.embeddings import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

"""
request will receive a record list in the body. Each record will have the below structure:
{
    "id": String (GUID)
    "originalText": String
    "updatedText": String
}

The function will return a record list where each record will have the below structure:
{
    "id": String (GUID)
    "originalEmbedding: []float
    "updatedEmbedding": []float
    "comparison": float
    "originalText": String
    "updatedText": String

}
"""

@app.route(route="generateReport")
def generateReport(req: func.HttpRequest) -> func.HttpResponse:
    # Check if the request method is POST, if not, return an error response
    if req.method != "POST":
        return func.HttpResponse(
            "Method not supported",
            status_code=599
        )
    try:
        # Load environment variables from the .env file
        load_dotenv()
        API_ENDPOINT = os.environ.get('API_ENDPOINT')
        API_VERSION = os.environ.get('API_VERSION')
        API_KEY = os.environ.get('API_KEY')

        # Create an instance of AzureOpenAIEmbeddings for text embedding
        EMBEDDING_MODEL = AzureOpenAIEmbeddings(
            azure_deployment="ada-embedding-002",
            azure_endpoint=API_ENDPOINT,
            openai_api_version=API_VERSION,
            openai_api_key=API_KEY
        )

        # Get the JSON body from the request
        body = req.get_json()
        records = body.get('records')
        return_records = []

        # Iterate over each record in the request
        for record in records:
            original_text = record["originalText"]
            updated_text = record["updatedText"]
            id = record["id"]

            # Embed the original and updated texts
            original_embedding = EMBEDDING_MODEL.embed_query(original_text)
            updated_embedding = EMBEDDING_MODEL.embed_query(updated_text)

            # Calculate the cosine similarity between the two embeddings
            dot_product = np.dot(original_embedding, updated_embedding)
            original_norm = np.linalg.norm(original_embedding)
            updated_norm = np.linalg.norm(updated_embedding)

            similarity_comparison = dot_product / (original_norm * updated_norm)

            # Append the result to the return_records list
            return_records.append({
                "id": id,
                "comparison": similarity_comparison,
                "originalText": original_text,
                "updatedText": updated_text
            })

        # Convert the return_records list to JSON and return it as a response
        return_json = json.dumps(return_records)
        return func.HttpResponse(
            return_json,
            status_code=200
        )

    except Exception as e:
        # Return an error response if an exception occurs
        return func.HttpResponse(
            f"An error occurred while trying to process your request: {e}",
            status_code=598
        )