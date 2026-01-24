import os

# Folders to completely ignore
IGNORE_DIRS = {'data', '__pycache__', '.git', 'venv', 'env', '.vscode', 'weights'}
# Extensions to ignore (so we don't list 33k images)
IGNORE_EXTS = {'.mpg', '.avi', '.mp4', '.png', '.jpg', '.jpeg', '.npy', '.pt', '.zip', '.pyc'}

def print_tree(startpath):
    print(f"Project Structure: {os.path.basename(os.path.abspath(startpath))}/")
    
    for root, dirs, files in os.walk(startpath):
        # Modify dirs in-place to skip ignored folders
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = '    ' * level
        print(f'{indent}{os.path.basename(root)}/')
        
        subindent = '    ' * (level + 1)
        for f in files:
            if not any(f.endswith(ext) for ext in IGNORE_EXTS):
                print(f'{subindent}{f}')

if __name__ == "__main__":
    print_tree(".")