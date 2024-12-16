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

# Import existing constants and helper functions from onefilellm.py
EXCLUDED_DIRS = {'.venv', '__pycache__', '.git', 'node_modules', '.pytest_cache', '.idea', '.vs', '.next'}

TOKEN = os.getenv('GITHUB_TOKEN', 'default_token_here')
if TOKEN == 'default_token_here':
    raise EnvironmentError("GITHUB_TOKEN environment variable not set.")

headers = {"Authorization": f"token {TOKEN}"}

# Copy all the helper functions from onefilellm.py
def download_file(url, target_path):
    # ... (copy the function implementation)
    pass

def is_allowed_filetype(filename):
    # ... (copy the function implementation)
    pass

# Copy all other helper functions...

# Update the default excluded dirs
DEFAULT_EXCLUDED_DIRS = {'.venv', '__pycache__', '.git', 'node_modules', '.pytest_cache', '.idea', '.vs', '.next'}

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