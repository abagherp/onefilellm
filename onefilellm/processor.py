import os
import sys
import requests
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader
import tiktoken
import nltk
from nltk.corpus import stopwords
import re
from pathlib import Path
import nbformat
from nbconvert import PythonExporter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import pyperclip
import wget
from tqdm import tqdm
from time import sleep
import xml.etree.ElementTree as ET
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

# Download NLTK data and initialize stop words
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words("english"))

# Constants
DEFAULT_EXCLUDED_DIRS = {'.venv', '__pycache__', '.git', 'node_modules', '.pytest_cache', '.idea', '.vs', '.next'}

TOKEN = os.getenv('GITHUB_TOKEN', 'default_token_here')
if TOKEN == 'default_token_here':
    raise EnvironmentError("GITHUB_TOKEN environment variable not set.")

headers = {"Authorization": f"token {TOKEN}"}

def safe_file_read(filepath, fallback_encoding='latin1'):
    """
    Safely read a file with UTF-8 encoding, falling back to latin1 if needed.
    
    Args:
        filepath: Path to the file to read
        fallback_encoding: Encoding to try if UTF-8 fails
    
    Returns:
        str: Contents of the file
    """
    try:
        with open(filepath, "r", encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding=fallback_encoding) as file:
            return file.read()

# Copy all the helper functions from onefilellm.py
def download_file(url, target_path):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    with open(target_path, "wb") as f:
        f.write(response.content)

def is_allowed_filetype(filename):
    allowed_extensions = ['.py', '.txt', '.js', '.tsx', '.ts', '.md', '.cjs', '.html', '.json', '.ipynb', '.h', '.localhost', '.sh', '.yaml', '.example']
    exluded_files = ['compressed_output.txt', 'uncompressed_output.txt', 'processed_urls.txt', 'instruction.md', 'instructions.md']
    return any(filename.endswith(ext) for ext in allowed_extensions) and not any(filename.endswith(file) for file in exluded_files)

def process_ipynb_file(temp_file):
    with open(temp_file, "r", encoding='utf-8', errors='ignore') as f:
        notebook_content = f.read()

    exporter = PythonExporter()
    python_code, _ = exporter.from_notebook_node(nbformat.reads(notebook_content, as_version=4))
    return python_code

def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def process_local_folder(local_path):
    content = [f'<source type="local_directory" path="{escape_xml(local_path)}">']
    for root, dirs, files in os.walk(local_path):
        # Exclude unwanted directories
        dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDED_DIRS]
        
        for file in files:
            if is_allowed_filetype(file):
                print(f"Processing {os.path.join(root, file)}...")

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, local_path)
                content.append(f'<file name="{escape_xml(relative_path)}">')

                if file.endswith(".ipynb"):
                    content.append(escape_xml(process_ipynb_file(file_path)))
                else:
                    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                        content.append(escape_xml(f.read()))

                content.append('</file>')

    content.append('</source>')
    print("All files processed.")
    return '\n'.join(content)

def get_token_count(text, disallowed_special=[], chunk_size=1000):
    enc = tiktoken.get_encoding("cl100k_base")

    # Remove XML tags
    text_without_tags = re.sub(r'<[^>]+>', '', text)

    # Split the text into smaller chunks
    chunks = [text_without_tags[i:i+chunk_size] for i in range(0, len(text_without_tags), chunk_size)]
    total_tokens = 0

    for chunk in chunks:
        tokens = enc.encode(chunk, disallowed_special=disallowed_special)
        total_tokens += len(tokens)
    
    return total_tokens

def preprocess_text(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as input_file:
        input_text = input_file.read()

    def process_text(text):
        text = re.sub(r"[\n\r]+", "\n", text)
        text = re.sub(r"[^a-zA-Z0-9\s_.,!?:;@#$%^&*()+\-=[\]{}|\\<>`~'\"/]+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    try:
        # Try to parse the input as XML
        root = ET.fromstring(input_text)

        # Process text content while preserving XML structure
        for elem in root.iter():
            if elem.text:
                elem.text = process_text(elem.text)
            if elem.tail:
                elem.tail = process_text(elem.tail)

        # Write the processed XML to the output file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print("Text preprocessing completed with XML structure preserved.")
    except ET.ParseError:
        # If XML parsing fails, process the text without preserving XML structure
        processed_text = process_text(input_text)
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(processed_text)
        print("XML parsing failed. Text preprocessing completed without XML structure.")

def process_input(input_path, working_dir, console, custom_excluded_dirs=None):
    """
    Process the input and generate output files in the working directory
    
    Args:
        input_path: Path or URL to process
        working_dir: Directory where output files should be created
        console: Rich console instance for output
        custom_excluded_dirs: Set of additional directories to exclude
    """
    # Combine default and custom excluded directories
    excluded_dirs = DEFAULT_EXCLUDED_DIRS.union(custom_excluded_dirs or set())
    
    # Define output files with full paths
    output_file = os.path.join(working_dir, "uncompressed_output.txt")
    processed_file = os.path.join(working_dir, "compressed_output.txt")
    urls_list_file = os.path.join(working_dir, "processed_urls.txt")

    console.print(f"\n[bold bright_green]You entered:[/bold bright_green] [bold bright_yellow]{input_path}[/bold bright_yellow]\n")

    with Progress(
        TextColumn("[bold bright_blue]{task.description}"),
        BarColumn(bar_width=None),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[bright_blue]Processing...", total=100)

        try:
            if "github.com" in input_path:
                if "/pull/" in input_path:
                    final_output = process_github_pull_request(input_path)
                elif "/issues/" in input_path:
                    final_output = process_github_issue(input_path)
                else:
                    final_output = process_github_repo(input_path)
            elif urlparse(input_path).scheme in ["http", "https"]:
                if "youtube.com" in input_path or "youtu.be" in input_path:
                    final_output = fetch_youtube_transcript(input_path)
                elif "arxiv.org" in input_path:
                    final_output = process_arxiv_pdf(input_path)
                else:
                    crawl_result = crawl_and_extract_text(input_path, max_depth=2, include_pdfs=True, ignore_epubs=True)
                    final_output = crawl_result['content']
                    with open(urls_list_file, 'w', encoding='utf-8') as urls_file:
                        urls_file.write('\n'.join(crawl_result['processed_urls']))
            elif input_path.startswith("10.") and "/" in input_path or input_path.isdigit():
                final_output = process_doi_or_pmid(input_path)
            else:
                final_output = process_local_folder(input_path)

            progress.update(task, advance=50)

            # Write the uncompressed output
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(final_output)

            # Process the compressed output
            preprocess_text(output_file, processed_file)

            progress.update(task, advance=50)

            compressed_text = safe_file_read(processed_file)
            compressed_token_count = get_token_count(compressed_text)
            console.print(f"\n[bright_green]Compressed Token Count:[/bright_green] [bold bright_cyan]{compressed_token_count}[/bold bright_cyan]")

            uncompressed_text = safe_file_read(output_file)
            uncompressed_token_count = get_token_count(uncompressed_text)
            console.print(f"[bright_green]Uncompressed Token Count:[/bright_green] [bold bright_cyan]{uncompressed_token_count}[/bold bright_cyan]")

            console.print(f"\n[bold bright_yellow]{processed_file}[/bold bright_yellow] and [bold bright_blue]{output_file}[/bold bright_blue] have been created in the working directory.")

            pyperclip.copy(uncompressed_text)
            console.print(f"\n[bright_white]The contents of [bold bright_blue]{output_file}[/bold bright_blue] have been copied to the clipboard.[/bright_white]")

        except Exception as e:
            console.print(f"\n[bold red]An error occurred:[/bold red] {str(e)}")
            console.print("\nPlease check your input and try again.")
            raise