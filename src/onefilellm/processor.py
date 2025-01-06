import os
from urllib.parse import urlparse
import nltk
from nltk.corpus import stopwords
import re
import pyperclip
import xml.etree.ElementTree as ET
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

from onefilellm.utils import (
    process_github_pull_request, process_github_issue, process_github_repo, fetch_youtube_transcript, process_arxiv_pdf, 
    crawl_and_extract_text, process_doi_or_pmid, escape_xml, should_exclude_path, is_allowed_filetype, 
    process_ipynb_file, preprocess_text, safe_file_read, process_local_folder, token_manager
)

# Download NLTK data and initialize stop words
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words("english"))

# Constants
TOKEN = os.getenv('GITHUB_TOKEN', 'default_token_here')
if TOKEN == 'default_token_here':
    raise EnvironmentError("GITHUB_TOKEN environment variable not set.")

headers = {"Authorization": f"token {TOKEN}"}

def process_input(input_path, working_dir, console, custom_excluded_dirs=None, max_tokens=None, output_dir=None, excluded_exts=None):
    """
    Process the input and generate output files
    
    Args:
        input_path: Path or URL to process
        working_dir: Directory where output files should be created
        console: Rich console instance for output
        custom_excluded_dirs: Set of additional directories to exclude
        max_tokens: Maximum number of tokens per file
        output_dir: Optional directory to store output files
        excluded_exts: Set of file extensions to exclude
    """
    # Initialize token counter per extension
    extension_tokens = {}
    total_tokens = 0
    
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
                    final_output, extension_tokens = process_github_pull_request(input_path, custom_excluded_dirs, max_tokens, excluded_exts)
                elif "/issues/" in input_path:
                    final_output, extension_tokens = process_github_issue(input_path, custom_excluded_dirs, max_tokens, excluded_exts)
                else:
                    final_output, extension_tokens = process_github_repo(input_path, custom_excluded_dirs, max_tokens, excluded_exts)
            elif urlparse(input_path).scheme in ["http", "https"]:
                if "youtube.com" in input_path or "youtu.be" in input_path:
                    final_output = fetch_youtube_transcript(input_path, max_tokens)
                elif "arxiv.org" in input_path:
                    final_output = process_arxiv_pdf(input_path, max_tokens)
                else:
                    crawl_result = crawl_and_extract_text(input_path, max_depth=2, include_pdfs=True, ignore_epubs=True, max_tokens=max_tokens)
                    final_output = crawl_result['content']
                    with open(urls_list_file, 'w', encoding='utf-8') as urls_file:
                        urls_file.write('\n'.join(crawl_result['processed_urls']))
            elif input_path.startswith("10.") and "/" in input_path or input_path.isdigit():
                final_output = process_doi_or_pmid(input_path, max_tokens)
            else:
                final_output, extension_tokens = process_local_folder(input_path, max_tokens, custom_excluded_dirs, excluded_exts)

            progress.update(task, advance=50)

            # Write the uncompressed output
            with open(output_file, "w", encoding='utf-8') as file:
                file.write(final_output)

            # Process the compressed output
            preprocess_text(output_file, processed_file)

            progress.update(task, advance=50)

            # Get token counts using TokenManager
            compressed_text = safe_file_read(processed_file)
            compressed_token_count = token_manager.count_tokens(compressed_text)
            console.print(f"\n[bright_green]Compressed Token Count:[/bright_green] [bold bright_cyan]{compressed_token_count}[/bold bright_cyan]")

            uncompressed_text = safe_file_read(output_file)
            uncompressed_token_count = token_manager.count_tokens(uncompressed_text)
            console.print(f"[bright_green]Uncompressed Token Count:[/bright_green] [bold bright_cyan]{uncompressed_token_count}[/bold bright_cyan]")

            # Display extension summary if we have extension data
            if extension_tokens:
                total_tokens = sum(extension_tokens.values())
                console.print("\n[bold bright_green]Token Summary by Extension:[/bold bright_green]")
                # Sort extensions by token count in descending order
                sorted_extensions = sorted(extension_tokens.items(), key=lambda x: x[1], reverse=True)
                for ext, tokens in sorted_extensions:
                    percentage = (tokens / total_tokens) * 100
                    console.print(f"[bright_white]{ext}:[/bright_white] [bold bright_cyan]{tokens:,}[/bold bright_cyan] tokens ([bold bright_yellow]{percentage:.1f}%[/bold bright_yellow])")

            console.print(f"\n[bold bright_yellow]{processed_file}[/bold bright_yellow] and [bold bright_blue]{output_file}[/bold bright_blue] have been created in {output_dir}")

            pyperclip.copy(uncompressed_text)
            console.print(f"\n[bright_white]The contents of [bold bright_blue]{output_file}[/bold bright_blue] have been copied to the clipboard.[/bright_white]")

        except Exception as e:
            console.print(f"\n[bold red]An error occurred:[/bold red] {str(e)}")
            console.print("\nPlease check your input and try again.")
            raise

def get_source_name(input_path):
    """Generate a name for the output files based on the input source"""
    if "github.com" in input_path:
        parts = input_path.split("github.com/")[-1].split("/")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    elif os.path.isdir(input_path):
        return os.path.basename(input_path)
    return "output"