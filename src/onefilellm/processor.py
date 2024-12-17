import os
from urllib.parse import urlparse
import tiktoken
import nltk
from nltk.corpus import stopwords
import re
import pyperclip
import xml.etree.ElementTree as ET
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

from onefilellm.utils import (
    process_github_pull_request, process_github_issue, process_github_repo, fetch_youtube_transcript, process_arxiv_pdf, 
    crawl_and_extract_text, process_doi_or_pmid, escape_xml, truncate_text_to_tokens, should_exclude_path, is_allowed_filetype, 
    process_ipynb_file, get_token_count, preprocess_text, safe_file_read
)

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

def get_source_name(input_path):
    """Generate a name for the output files based on the input source"""
    if "github.com" in input_path:
        parts = input_path.split("github.com/")[-1].split("/")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    elif os.path.isdir(input_path):
        return os.path.basename(input_path)
    return "output"

def process_input(input_path, working_dir, console, custom_excluded_dirs=None, max_tokens=None, output_dir=None):
    """
    Process the input and generate output files
    
    Args:
        input_path: Path or URL to process
        working_dir: Directory where output files should be created
        console: Rich console instance for output
        custom_excluded_dirs: Set of additional directories to exclude
        max_tokens: Maximum number of tokens per file
        output_dir: Optional directory to store output files
    """
    # Generate source name for output files
    source_name = get_source_name(input_path)
    
    # Determine output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, source_name)
    else:
        output_base = os.path.join(os.getcwd(), source_name)
    
    # Define output files with full paths
    output_file = f"{output_base}_uncompressed.txt"
    processed_file = f"{output_base}_compressed.txt"
    urls_list_file = f"{output_base}_urls.txt"

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
                    final_output = process_github_pull_request(input_path, custom_excluded_dirs)
                elif "/issues/" in input_path:
                    final_output = process_github_issue(input_path, custom_excluded_dirs)
                else:
                    final_output = process_github_repo(input_path, custom_excluded_dirs)
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
                final_output = process_local_folder(input_path, max_tokens, custom_excluded_dirs)

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

            console.print(f"\n[bold bright_yellow]{processed_file}[/bold bright_yellow] and [bold bright_blue]{output_file}[/bold bright_blue] have been created in {output_dir}")

            pyperclip.copy(uncompressed_text)
            console.print(f"\n[bright_white]The contents of [bold bright_blue]{output_file}[/bold bright_blue] have been copied to the clipboard.[/bright_white]")

        except Exception as e:
            console.print(f"\n[bold red]An error occurred:[/bold red] {str(e)}")
            console.print("\nPlease check your input and try again.")
            raise