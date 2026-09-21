import os
import re

def strip_all_comments_and_docstrings(text):
    """Aggressively removes all # comments and docstrings via precise regex."""
    # 1. Remove multi-line triple-quoted docstrings (""" ... """)
    text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)

    # 2. Remove inline # comments, making sure we don't clear URLs or strings
    # Looks for a # that is NOT inside a string quote
    text = re.sub(r'(?m)^([^\n\'"]*?)#.*$', r'\1', text)

    return text

def flatten_codebase(root_dir, output_file):
    # Folders to skip
    # exclude_dirs = {'.git', '__pycache__', 'venv', 'node_modules', '.vscode', 'dist', 'build', 'data'}

    exclude_dirs = {'.git', '__pycache__', 'venv', 'node_modules', '.vscode', 'dist', 'build', 'data','tests'}

    # File extensions to skip
    exclude_exts = {'.pyc', '.exe', '.bin', '.pdf', '.jpg', '.png', '.zip', '.lock'}

    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in exclude_exts):
                    continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)

                f.write(f"\n{'='*60}\nPATH: {rel_path}\n{'='*60}\n")

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as code_file:
                        content = code_file.read()

                        # Apply aggressive cleaning to Python files
                        if file.endswith('.py'):
                            content = strip_all_comments_and_docstrings(content)

                        # REMOVE ALL BLANK LINES (including lines left empty by deleted comments)
                        content = re.sub(r'^\s*$\n', '', content, flags=re.MULTILINE)

                        f.write(content)
                    f.write("\n")
                except Exception as e:
                    f.write(f"[Error reading file: {e}]\n")

if __name__ == "__main__":
    flatten_codebase('.', 'full_codebase_dump.txt')
    print("Done! Real comment-free codebase dump generated.")
