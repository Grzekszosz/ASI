import os
import zipfile

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()


def download_model_from_azure(container_name: str, blob_name: str, local_path: str):
    conn_str = os.getenv("CONNSTR")
    if not conn_str:
        raise RuntimeError("Brak zmiennej środowiskowej AZURE_STORAGE_CONNECTION_STRING. Ustaw ją tak, aby wskazywała na twój Storage Account.")
    local_zip = "ag.zip"
    extract_dir = "autogluon_model"
    # 2. Inicjalizacja klienta Azure Blob Service
    blob_service_client = BlobServiceClient.from_connection_string(conn_str)

    # 3. Pobranie klienta kontenera
    container_client = blob_service_client.get_container_client(container_name)

    # 4. Pobranie klienta konkretnego bloba
    blob_client = container_client.get_blob_client(blob_name)

    # 5. Pobranie zawartości i zapis do pliku lokalnego
    with open(local_path, "wb") as file:
        download_stream = blob_client.download_blob()
        file.write(download_stream.readall())

    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Pobrano model: '{blob_name}' do '{local_path}'")
