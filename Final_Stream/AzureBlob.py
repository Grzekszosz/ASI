import os
from azure.storage.blob import BlobServiceClient

def download_model_from_azure(container_name: str, blob_name: str, local_path: str):
    """
    Pobiera plik (np. model.joblib) z Azure Blob Storage do lokalnego systemu plików.

    container_name  – nazwa kontenera (czyli odpowiednik bucketa)
    blob_name       – nazwa pliku w kontenerze (np. "model.joblib")
    local_path      – ścieżka, gdzie zapisać pobrany plik lokalnie
    """

    # 1. Sprawdź, czy jest ustawiona zmienna środowiskowa z connection string
    conn_str = os.getenv("CONNSTR")
    if not conn_str:
        raise RuntimeError("Brak zmiennej środowiskowej AZURE_STORAGE_CONNECTION_STRING. Ustaw ją tak, aby wskazywała na twój Storage Account.")

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

    print(f"Pobrano model: '{blob_name}' do '{local_path}'")
