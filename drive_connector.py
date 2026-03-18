import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'service_account.json'

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def list_files(folder_id):
    service = get_drive_service()
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get('files', [])
    if not files:
        print("No files found. Make sure you shared the folder with the service account email.")
    for f in files:
        print(f"  {f['name']}  ({f['mimeType']})")
    return files

if __name__ == "__main__":
    folder_id = input("Paste your Google Drive folder ID: ")
    print("\nFiles found:")
    list_files(folder_id)
