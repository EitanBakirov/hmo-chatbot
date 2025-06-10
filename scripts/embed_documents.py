import os
import json
from bs4 import BeautifulSoup
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load secrets
load_dotenv()

# Initialize Azure OpenAI client
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
if azure_endpoint is None:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set.")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=azure_endpoint
)

# Define files and domain labels
html_files = {
    "optometry": "phase2_data/optometry_services.html",
    "pregnancy": "phase2_data/pregnancy_services.html",
    "workshops": "phase2_data/workshops_services.html",
    "alternative": "phase2_data/alternative_services.html",
    "communication_clinic": "phase2_data/communication_clinic_services.html",
    "dental": "phase2_data/dental_services.html"
}

# Function to extract visible text
def extract_text_from_html(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

def generate_embeddings() -> str:
    """Generate embeddings for HTML files"""
    embedded_docs = []
    for domain, filepath in html_files.items():
        print(f"Embedding: {domain}")
        text = extract_text_from_html(filepath)
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        embedded_docs.append({
            "domain": domain,
            "text": text,
            "embedding": response.data[0].embedding
        })

    # Save to file
    output_path = "phase2_data/embedded_docs.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in embedded_docs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return output_path

if __name__ == "__main__":
    generate_embeddings()
