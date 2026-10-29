import os
import re
import glob

def cleanup_temp_files(current_script_dir):
    # Get the directory where the script is located
    script_dir = current_script_dir

    # Delete files with specific extensions in the script's directory
    for ext in ('*.sol', '*.res', '*.sav', '*.pul'):
        for file in glob.glob(os.path.join(script_dir, ext)):
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error deleting {file}: {e}")

    # Regex pattern for files like a7f9c2.1
    pattern = re.compile(r'^[0-9a-f]+\.[0-9]+$')

    # Delete matching files in the script's directory (excluding .py files)
    for file in os.listdir(script_dir):
        path = os.path.join(script_dir, file)
        if os.path.isfile(path) and not file.endswith('.py') and pattern.match(file):
            try:
                os.remove(path)
            except OSError as e:
                print(f"Error deleting {file}: {e}")
