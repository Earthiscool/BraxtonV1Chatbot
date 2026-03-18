import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'service_account.json'
DOCS_DIR = './downloaded_docs'
CHROMA_DIR = './chroma_db'
GEMINI_API_KEY = 'AIzaSyBcuW9eWhqLq5D3Yvr1q8jR6aWxbLH0BaU'

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def download_documents(folder_id):
    service = get_drive_service()
    os.makedirs(DOCS_DIR, exist_ok=True)
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get('files', [])

    print(f"Downloading {len(files)} files...")
    for f in files:
        file_id = f['id']
        name = f['name']
        mime = f['mimeType']

        try:
            if mime == 'application/vnd.google-apps.document':
                request = service.files().export_media(fileId=file_id, mimeType='text/plain')
                filepath = os.path.join(DOCS_DIR, f"{name}.txt")
            elif mime == 'application/vnd.google-apps.spreadsheet':
                request = service.files().export_media(fileId=file_id, mimeType='text/csv')
                filepath = os.path.join(DOCS_DIR, f"{name}.csv")
            else:
                print(f"  Skipping unsupported type: {name}")
                continue

            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            with open(filepath, 'wb') as out:
                out.write(buffer.getvalue())
            print(f"  Downloaded: {name}")

        except Exception as e:
            print(f"  Failed: {name} — {e}")

def build_index():
    print("\nBuilding embeddings and storing in ChromaDB...")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print(f"Loaded {len(documents)} document chunks")


    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("client_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)


    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("\nIndex saved to ./chroma_db")
    return index

if __name__ == "__main__":
    folder_id = input("Paste your Google Drive folder ID: ")
    download_documents(folder_id)
    build_index()