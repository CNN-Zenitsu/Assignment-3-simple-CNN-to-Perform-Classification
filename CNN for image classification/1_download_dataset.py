# Download and extract RealWaste dataset for waste classification
import requests
import zipfile
import os
import shutil

# Dataset URL from UCI ML Repository
url = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"

# Download dataset
file_name = "realwaste_dataset.zip"
response = requests.get(url, stream=True)
response.raise_for_status()
with open(file_name, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print(f"Dataset downloaded successfully and saved as {file_name}")

def unzip(zip_file_path, extract_to):
    """Extract zip file to specified directory"""
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Dataset extracted successfully to {extract_to}")

# Extract and organize dataset
os.makedirs("realwaste_dataset_extracted", exist_ok=True)
unzip("realwaste_dataset.zip", "realwaste_dataset_extracted")
shutil.move("/content/realwaste_dataset_extracted/realwaste-main/RealWaste", "/content")
# Clean up temporary files
os.remove("/content/realwaste_dataset.zip")
shutil.rmtree("/content/realwaste_dataset_extracted")
