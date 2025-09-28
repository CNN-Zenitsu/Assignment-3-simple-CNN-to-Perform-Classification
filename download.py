import zipfile
import os

def extract_zip(zip_path, extract_to=None):
    """
    Extracts a zip file to a specified directory.
    
    :param zip_path: Path to the zip file
    :param extract_to: Directory to extract files into (default: same folder as zip)
    """
    if extract_to is None:
        extract_to = os.path.splitext(zip_path)[0]  # create folder with same name
    
    # Create output directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")

# Example usage
if __name__ == "__main__":
    zip_file = "vine.zip"   # replace with your zip file path
    
    extract_zip(zip_file, "extracted_files")  # you can leave second arg None if you want default
