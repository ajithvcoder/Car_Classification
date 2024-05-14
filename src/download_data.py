
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to move kaggle JSON file to home")
    
    parser.add_argument('--json_file', required=True, help="Path to the JSON file")
    
    args = parser.parse_args()
    
    return args

import os
import shutil

def move_kaggle_json(kaggle_json):
    # Check the operating system
    if os.name == 'posix':  # Linux/Mac OS
        dest_dir = os.path.join('/root', '.kaggle')
    elif os.name == 'nt':   # Windows
        dest_dir = os.path.join(os.environ['USERPROFILE'], '.kaggle')
    else:
        print("Unsupported operating system")
        return False

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # Move the kaggle.json file to the appropriate folder
    shutil.move(kaggle_json, dest_dir)
    print("kaggle.json file moved successfully.")

    return True

if __name__ == "__main__":
    args = parse_arguments()
    json_file_path = args.json_file
    
    if move_kaggle_json(json_file_path): 
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('kshitij192/cars-image-dataset', path='./data', unzip=True)
