import requests
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader
import os
import tiktoken
import nltk
from nltk.corpus import stopwords
import re
import nbformat
from nbconvert import PythonExporter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import wget
from rich import print
import xml.etree.ElementTree as ET
from fnmatch import fnmatch
from .constants import (
    DEFAULT_EXCLUDED_DIRS,
    ALLOWED_EXTENSIONS,
    EXCLUDED_FILES,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_ENCODING
)

DEFAULT_EXCLUDED_DIRS = {'.venv', '__pycache__', '.git', 'node_modules', '.pytest_cache', '.idea', '.vs', '.next'}

def get_github_headers():
    """Get GitHub API headers with token if available"""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise EnvironmentError("GITHUB_TOKEN environment variable not set. This is required for GitHub operations.")
    return {"Authorization": f"token {token}"}

def safe_file_read(filepath, fallback_encoding='latin1'):
    """Safely read a file with UTF-8 encoding, falling back to latin1"""
    try:
        with open(filepath, "r", encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding=fallback_encoding) as file:
            return file.read()

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

def download_file(url, target_path, headers=None):
    """Download a file from URL to target path"""
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    with open(target_path, "wb") as f:
        f.write(response.content)

def get_token_count(text, disallowed_special=[], chunk_size=1000):
    """Count tokens in text, removing XML tags first."""
    text_without_tags = re.sub(r'<[^>]+>', '', text)
    chunks = [text_without_tags[i:i+chunk_size] for i in range(0, len(text_without_tags), chunk_size)]
    enc = tiktoken.get_encoding(DEFAULT_ENCODING)
    return sum(len(enc.encode(chunk, disallowed_special=disallowed_special)) for chunk in chunks)

def is_allowed_filetype(filename, excluded_exts=None):
    """Check if a file should be processed based on extension"""
    # First check if the file is in the excluded patterns
    if any(fnmatch(filename, pattern) for pattern in EXCLUDED_FILES):
        return False
        
    # Get the file extension
    _, ext = os.path.splitext(filename)
    
    # If there are excluded extensions, check those first
    if excluded_exts and ext in excluded_exts:
        return False
        
    # Finally check if it's in the allowed extensions
    return ext in ALLOWED_EXTENSIONS

def remove_base64_content(text):
    """Remove base64 encoded content from text."""
    # Remove markdown image with base64
    text = re.sub(r'!\[.*?\]\(data:image/[^;]+;base64,[^)]+\)', '[BASE64_IMAGE_REMOVED]', text)
    # Remove HTML img with base64
    text = re.sub(r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>', '[BASE64_IMAGE_REMOVED]', text)
    # Remove raw base64 strings
    text = re.sub(r'data:image/[^;]+;base64,[a-zA-Z0-9+/=\n]+', '[BASE64_IMAGE_REMOVED]', text)
    return text

def clean_notebook_output(text):
    """Clean notebook output by removing In[] lines and consolidating empty lines."""
    # Remove "# In[X]:" lines (with any number or empty brackets)
    text = re.sub(r'#\s*In\s*\[\s*\d*\s*\]:\s*\n', '', text)
    # Consolidate multiple empty lines into a single empty line
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text

def process_ipynb_file(temp_file):
    """Convert Jupyter notebook to Python code, excluding cell outputs"""
    try:
        with open(temp_file, "r", encoding='utf-8', errors='ignore') as f:
            notebook_content = f.read()

        # Remove base64 content before parsing
        notebook_content = remove_base64_content(notebook_content)
        
        notebook = nbformat.reads(notebook_content, as_version=4)
        # Remove all outputs from cells
        for cell in notebook.cells:
            if 'outputs' in cell:
                cell.outputs = []
            # Clean markdown cells
            if cell.get('cell_type') == 'markdown' and 'source' in cell:
                cell.source = remove_base64_content(cell.source)

        exporter = PythonExporter()
        python_code, _ = exporter.from_notebook_node(notebook)
        
        # Clean up the exported code
        python_code = clean_notebook_output(python_code)
        
        return python_code
    except Exception as e:
        print(f"Warning: Error processing notebook {temp_file}: {str(e)}")
        return f"# Error processing notebook: {str(e)}"

def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

class TokenManager:
    """Centralized class for managing token-related operations."""
    
    def __init__(self, encoding=DEFAULT_ENCODING, chunk_size=DEFAULT_CHUNK_SIZE):
        """Initialize the TokenManager with encoding and chunk size."""
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.enc = tiktoken.get_encoding(encoding)
        # Allow all special tokens to avoid errors
        self.disallowed_special = ()
    
    def count_tokens(self, text, disallowed_special=None):
        """Count tokens in text, removing XML tags first."""
        text_without_tags = re.sub(r'<[^>]+>', '', text)
        chunks = [text_without_tags[i:i+self.chunk_size] for i in range(0, len(text_without_tags), self.chunk_size)]
        disallowed = self.disallowed_special if disallowed_special is None else disallowed_special
        return sum(len(self.enc.encode(chunk, disallowed_special=disallowed)) for chunk in chunks)
    
    def truncate_text(self, text, max_tokens):
        """
        Truncate text to max tokens and return truncation info.
        
        Returns:
            tuple: (truncated_text, was_truncated, truncation_percentage)
        """
        if not max_tokens:
            return text, False, 100
            
        tokens = self.enc.encode(text, disallowed_special=self.disallowed_special)
        if len(tokens) <= max_tokens:
            return text, False, 100
            
        truncated_tokens = tokens[:max_tokens]
        return (
            self.enc.decode(truncated_tokens),
            True,
            100 * round(len(truncated_tokens)/len(tokens), 2)
        )
    
    def process_content(self, content, max_tokens=None):
        """
        Process content with token counting and optional truncation.
        
        Returns:
            tuple: (processed_content, token_info_dict)
        """
        original_tokens = self.count_tokens(content)
        token_info = {
            'original_tokens': original_tokens,
            'was_truncated': False,
            'final_tokens': original_tokens,
            'truncation_percentage': 100
        }
        
        if max_tokens:
            content, was_truncated, truncation_percentage = self.truncate_text(content, max_tokens)
            final_tokens = self.count_tokens(content)
            token_info.update({
                'was_truncated': was_truncated,
                'final_tokens': final_tokens,
                'truncation_percentage': truncation_percentage
            })
            
        return content, token_info

# Create a global instance for convenience
token_manager = TokenManager()

def should_exclude_path(path, base_path, excluded_dirs):
    """
    Check if a path should be excluded based on excluded_dirs.
    Handles both absolute and relative paths.
    
    Args:
        path: Path to check
        base_path: Base directory path for relative path calculation
        excluded_dirs: Set of directory patterns to exclude
    """
    if not excluded_dirs:
        return False
        
    # Convert path to relative path if it's absolute
    try:
        rel_path = os.path.relpath(path, base_path)
    except ValueError:
        # If paths are on different drives (Windows), use absolute path
        rel_path = path
    
    # Check if any excluded pattern matches the path
    for excluded in excluded_dirs:
        # Handle both relative and absolute patterns
        if excluded.startswith('/'):
            # Absolute path pattern
            if path.startswith(excluded) or path.startswith(excluded.lstrip('/')):
                return True
        else:
            # Relative path pattern - check if it matches any part of the path
            path_parts = rel_path.split(os.sep)
            if excluded in path_parts:
                return True
            # Also check if pattern matches with wildcards
            if any(part.startswith(excluded) for part in path_parts):
                return True
    
    return False    

def process_directory(url, output, excluded_exts=None):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    files = response.json()

    for file in files:
        if file["type"] == "file" and is_allowed_filetype(file["name"], excluded_exts):
            print(f"Processing {file['path']}...")

            temp_file = f"temp_{file['name']}"
            download_file(file["download_url"], temp_file)

            output.write(f"# {'-' * 3}\n")
            output.write(f"# Filename: {file['path']}\n")
            output.write(f"# {'-' * 3}\n\n")

            if file["name"].endswith(".ipynb"):
                output.write(process_ipynb_file(temp_file))
            else:
                with open(temp_file, "r", encoding='utf-8', errors='ignore') as f:
                    output.write(f.read())

            output.write("\n\n")
            os.remove(temp_file)
        elif file["type"] == "dir":
            process_directory(file["url"], output, excluded_exts)

def process_local_directory(local_path, output, excluded_exts=None):
    for root, dirs, files in os.walk(local_path):
        for file in files:
            if is_allowed_filetype(file, excluded_exts):
                print(f"Processing {os.path.join(root, file)}...")

                output.write(f"# {'-' * 3}\n")
                output.write(f"# Filename: {os.path.join(root, file)}\n")
                output.write(f"# {'-' * 3}\n\n")

                file_path = os.path.join(root, file)

                if file.endswith(".ipynb"):
                    output.write(process_ipynb_file(file_path))
                else:
                    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                        output.write(f.read())

                output.write("\n\n")

def process_github_repo(repo_url, excluded_dirs=None, max_tokens=None, excluded_exts=None):
    try:
        headers = get_github_headers()
    except EnvironmentError as e:
        raise EnvironmentError(f"Cannot process GitHub repository: {str(e)}")

    api_base_url = "https://api.github.com/repos/"
    repo_url_parts = repo_url.split("https://github.com/")[-1].split("/")
    repo_name = "/".join(repo_url_parts[:2])

    subdirectory = ""
    if len(repo_url_parts) > 4 and repo_url_parts[2] == "tree":
        subdirectory = "/".join(repo_url_parts[4:])

    contents_url = f"{api_base_url}{repo_name}/contents"
    if subdirectory:
        contents_url = f"{contents_url}/{subdirectory}"

    repo_content = [f'<source type="github_repository" url="{repo_url}">']
    extension_tokens = {}

    def process_directory(url, repo_content, max_tokens=None, excluded_exts=None):
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        files = response.json()

        for file in files:
            # Skip excluded directories
            if file["type"] == "dir" and excluded_dirs and should_exclude_path(file["path"], "", excluded_dirs):
                print(f"Skipping excluded directory: {file['path']}")
                continue
                
            if file["type"] == "file" and is_allowed_filetype(file["name"], excluded_exts):
                # Skip files in excluded directories
                if excluded_dirs and should_exclude_path(file["path"], "", excluded_dirs):
                    print(f"Skipping file in excluded directory: {file['path']}")
                    continue
                    
                print(f"Processing {file['path']}...")

                temp_file = f"temp_{file['name']}"
                download_file(file["download_url"], temp_file, headers)

                repo_content.append(f'<file name="{escape_xml(file["path"])}">') 

                if file["name"].endswith(".ipynb"):
                    file_content = process_ipynb_file(temp_file)
                else:
                    with open(temp_file, "r", encoding='utf-8', errors='ignore') as f:
                        file_content = f.read()

                # Process content with token management
                file_content, token_info = token_manager.process_content(file_content, max_tokens)
                
                # Track tokens by extension
                _, ext = os.path.splitext(file["name"])
                extension_tokens[ext] = extension_tokens.get(ext, 0) + token_info['final_tokens']
                
                if token_info['was_truncated']:
                    print(f"Token count: {token_info['final_tokens']} (truncated from {token_info['original_tokens']}, {token_info['truncation_percentage']}%)")
                    repo_content.append(f'<!-- Content truncated from {token_info["original_tokens"]} to {token_info["final_tokens"]} tokens ({token_info["truncation_percentage"]}%) -->')
                else:
                    print(f"Token count: {token_info['final_tokens']}")

                repo_content.append(escape_xml(file_content))
                repo_content.append('</file>')
                os.remove(temp_file)
            elif file["type"] == "dir":
                process_directory(file["url"], repo_content, max_tokens, excluded_exts)

    process_directory(contents_url, repo_content, max_tokens, excluded_exts)
    repo_content.append('</source>')
    print("All files processed.")

    return "\n".join(repo_content), extension_tokens

def process_local_folder(local_path, max_tokens=None, excluded_dirs=None, excluded_exts=None):
    """Process a local directory and its files"""
    content = [f'<source type="local_directory" path="{escape_xml(local_path)}">']
    excluded_dirs = set(excluded_dirs or []).union(DEFAULT_EXCLUDED_DIRS)
    extension_tokens = {}
    
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if not should_exclude_path(
            os.path.join(root, d), 
            local_path, 
            excluded_dirs
        )]
        
        if should_exclude_path(root, local_path, excluded_dirs):
            continue
            
        for file in files:
            if is_allowed_filetype(file, excluded_exts):
                file_path = os.path.join(root, file)
                if should_exclude_path(file_path, local_path, excluded_dirs):
                    continue
                    
                print(f"Processing {file_path}...")

                relative_path = os.path.relpath(file_path, local_path)
                content.append(f'<file name="{escape_xml(relative_path)}">')

                if file.endswith(".ipynb"):
                    file_content = process_ipynb_file(file_path)
                else:
                    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                        file_content = f.read()
                
                # Process content with token management
                file_content, token_info = token_manager.process_content(file_content, max_tokens)
                
                # Track tokens by extension
                _, ext = os.path.splitext(file)
                extension_tokens[ext] = extension_tokens.get(ext, 0) + token_info['final_tokens']
                
                if token_info['was_truncated']:
                    print(f"Token count: {token_info['final_tokens']} (truncated from {token_info['original_tokens']}, {token_info['truncation_percentage']}%)")
                    content.append(f'<!-- Content truncated from {token_info["original_tokens"]} to {token_info["final_tokens"]} tokens ({token_info["truncation_percentage"]}%) -->')
                else:
                    print(f"Token count: {token_info['final_tokens']}")

                content.append(escape_xml(file_content))
                content.append('</file>')

    content.append('</source>')
    print("All files processed.")
    return '\n'.join(content), extension_tokens

def process_arxiv_pdf(arxiv_abs_url, max_tokens=None):
    pdf_url = arxiv_abs_url.replace("/abs/", "/pdf/") + ".pdf"
    response = requests.get(pdf_url)
    pdf_content = response.content

    with open('temp.pdf', 'wb') as pdf_file:
        pdf_file.write(pdf_content)

    text = []
    with open('temp.pdf', 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in range(len(pdf_reader.pages)):
            text.append(pdf_reader.pages[page].extract_text())

    content = ' '.join(text)
    
    # Process content with token management
    content, token_info = token_manager.process_content(content, max_tokens)

    formatted_text = f'<source type="arxiv_paper" url="{escape_xml(arxiv_abs_url)}">\n'
    formatted_text += '<paper>\n'
    if token_info['was_truncated']:
        print(f"Token count: {token_info['final_tokens']} (truncated from {token_info['original_tokens']}, {token_info['truncation_percentage']}%)")
        formatted_text += f'<!-- Content truncated from {token_info["original_tokens"]} to {token_info["final_tokens"]} tokens ({token_info["truncation_percentage"]}%) -->\n'
    else:
        print(f"Token count: {token_info['final_tokens']}")
    formatted_text += escape_xml(content)
    formatted_text += '\n</paper>\n'
    formatted_text += '</source>'

    os.remove('temp.pdf')
    print("ArXiv paper processed successfully.")

    return formatted_text

def extract_links(input_file, output_file):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
        urls = re.findall(url_pattern, content)
    
    with open(output_file, 'w', encoding='utf-8') as output:
        for url in urls:
            output.write(url + '\n')

def fetch_youtube_transcript(url, max_tokens=None):
    def extract_video_id(url):
        pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None

    video_id = extract_video_id(url)
    if not video_id:
        return f'<source type="youtube_transcript" url="{escape_xml(url)}">\n<error>Could not extract video ID from URL.</error>\n</source>'

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        transcript = formatter.format_transcript(transcript_list)
        
        # Process content with token management
        transcript, token_info = token_manager.process_content(transcript, max_tokens)
        
        formatted_text = f'<source type="youtube_transcript" url="{escape_xml(url)}">\n'
        formatted_text += '<transcript>\n'
        formatted_text += escape_xml(transcript)
        
        if token_info['was_truncated']:
            print(f"YouTube transcript tokens: {token_info['final_tokens']} (truncated from {token_info['original_tokens']}, {token_info['truncation_percentage']}%)")
            formatted_text += f'\n<!-- Content truncated from {token_info["original_tokens"]} to {token_info["final_tokens"]} tokens ({token_info["truncation_percentage"]}%) -->'
        else:
            print(f"YouTube transcript tokens: {token_info['final_tokens']}")
            
        formatted_text += '\n</transcript>\n'
        formatted_text += '</source>'
        
        return formatted_text
    except Exception as e:
        return f'<source type="youtube_transcript" url="{escape_xml(url)}">\n<error>{escape_xml(str(e))}</error>\n</source>'

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

def is_same_domain(base_url, new_url):
    return urlparse(base_url).netloc == urlparse(new_url).netloc

def is_within_depth(base_url, current_url, max_depth):
    base_parts = urlparse(base_url).path.rstrip('/').split('/')
    current_parts = urlparse(current_url).path.rstrip('/').split('/')

    if current_parts[:len(base_parts)] != base_parts:
        return False

    return len(current_parts) - len(base_parts) <= max_depth

def process_pdf(url, max_tokens=None):
    response = requests.get(url)
    response.raise_for_status()

    with open('temp.pdf', 'wb') as pdf_file:
        pdf_file.write(response.content)

    text = []
    with open('temp.pdf', 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in range(len(pdf_reader.pages)):
            text.append(pdf_reader.pages[page].extract_text())

    os.remove('temp.pdf')
    return ' '.join(text)

def crawl_and_extract_text(base_url, max_depth, include_pdfs, ignore_epubs, max_tokens=None):
    visited_urls = set()
    urls_to_visit = [(base_url, 0)]
    processed_urls = []
    all_text = [f'<source type="web_documentation" url="{escape_xml(base_url)}">']

    while urls_to_visit:
        current_url, current_depth = urls_to_visit.pop(0)
        clean_url = current_url.split('#')[0]

        if clean_url not in visited_urls and is_same_domain(base_url, clean_url) and is_within_depth(base_url, clean_url, max_depth):
            if ignore_epubs and clean_url.endswith('.epub'):
                continue

            try:
                response = requests.get(current_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                visited_urls.add(clean_url)

                if clean_url.endswith('.pdf') and include_pdfs:
                    text = process_pdf(clean_url, max_tokens=max_tokens)
                else:
                    # Remove script, style, etc.
                    for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
                        element.decompose()
                    
                    # Remove comments
                    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
                    for comment in comments:
                        comment.extract()
                    
                    # Remove base64 images
                    for img in soup.find_all('img'):
                        if img.get('src', '').startswith('data:image'):
                            img.replace_with('[BASE64_IMAGE_REMOVED]')
                    
                    text = soup.get_text(separator='\n', strip=True)
                    # Clean any remaining base64 content
                    text = remove_base64_content(text)
                    
                # Process content with token management
                text, token_info = token_manager.process_content(text, max_tokens)

                all_text.append(f'<page url="{escape_xml(clean_url)}">')
                if token_info['was_truncated']:
                    print(f"URL: {clean_url} - Tokens: {token_info['final_tokens']} (truncated from {token_info['original_tokens']}, {token_info['truncation_percentage']}%)")
                    all_text.append(f'<!-- Content truncated from {token_info["original_tokens"]} to {token_info["final_tokens"]} tokens ({token_info["truncation_percentage"]}%) -->')
                else:
                    print(f"URL: {clean_url} - Tokens: {token_info['final_tokens']}")

                all_text.append(escape_xml(text))
                all_text.append('</page>')
                processed_urls.append(clean_url)
                print(f"Processed: {clean_url}")

                if current_depth < max_depth:
                    for link in soup.find_all('a', href=True):
                        new_url = urljoin(current_url, link['href']).split('#')[0]
                        if new_url not in visited_urls and is_within_depth(base_url, new_url, max_depth) and (include_pdfs or not new_url.endswith('.pdf')) and not (ignore_epubs and new_url.endswith('.epub')):
                            urls_to_visit.append((new_url, current_depth + 1))

            except requests.RequestException as e:
                print(f"Failed to retrieve {clean_url}: {e}")

    all_text.append('</source>')
    formatted_content = '\n'.join(all_text)

    return {
        'content': formatted_content,
        'processed_urls': processed_urls
    }

def process_doi_or_pmid(identifier, max_tokens=None):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
        'Connection': 'keep-alive'
    }

    try:
        payload = {
            'sci-hub-plugin-check': '',
            'request': identifier
        }

        base_url = 'https://sci-hub.se/'
        response = requests.post(base_url, headers=headers, data=payload, timeout=60)
        soup = BeautifulSoup(response.content, 'html.parser')
        pdf_element = soup.find(id='pdf')

        if pdf_element is None:
            raise ValueError(f"No PDF found for identifier {identifier}. Sci-hub might be inaccessible or the document is not available.")

        content = pdf_element.get('src').replace('#navpanes=0&view=FitH', '').replace('//', '/')

        if content.startswith('/downloads'):
            pdf_url = 'https://sci-hub.se' + content
        elif content.startswith('/tree'):
            pdf_url = 'https://sci-hub.se' + content
        elif content.startswith('/uptodate'):
            pdf_url = 'https://sci-hub.se' + content
        else:
            pdf_url = 'https:/' + content

        pdf_filename = f"{identifier.replace('/', '-')}.pdf"
        wget.download(pdf_url, pdf_filename)

        with open(pdf_filename, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()

        # Process content with token management
        text, token_info = token_manager.process_content(text, max_tokens)

        formatted_text = f'<source type="sci_hub_paper" identifier="{escape_xml(identifier)}">\n'
        formatted_text += '<paper>\n'
        if token_info['was_truncated']:
            print(f"Token count: {token_info['final_tokens']} (truncated from {token_info['original_tokens']}, {token_info['truncation_percentage']}%)")
            formatted_text += f'<!-- Content truncated from {token_info["original_tokens"]} to {token_info["final_tokens"]} tokens ({token_info["truncation_percentage"]}%) -->\n'
        else:
            print(f"Token count: {token_info['final_tokens']}")
        formatted_text += escape_xml(text)
        formatted_text += '\n</paper>\n'
        formatted_text += '</source>'

        os.remove(pdf_filename)
        print(f"Identifier {identifier} processed successfully.")
        return formatted_text
    except (requests.RequestException, ValueError) as e:
        error_text = f'<source type="sci_hub_paper" identifier="{escape_xml(identifier)}">\n'
        error_text += f'<error>{escape_xml(str(e))}</error>\n'
        error_text += '</source>'
        print(f"Error processing identifier {identifier}: {str(e)}")
        print("Sci-hub appears to be inaccessible or the document was not found. Please try again later.")
        return error_text
        
def process_github_pull_request(pull_request_url, excluded_dirs=None, max_tokens=None, excluded_exts=None):
    try:
        headers = get_github_headers()
    except EnvironmentError as e:
        raise EnvironmentError(f"Cannot process GitHub pull request: {str(e)}")

    url_parts = pull_request_url.split("/")
    repo_owner = url_parts[3]
    repo_name = url_parts[4]
    pull_request_number = url_parts[-1]

    api_base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pull_request_number}"

    response = requests.get(api_base_url, headers=headers)
    pull_request_data = response.json()

    diff_url = pull_request_data["diff_url"]
    diff_response = requests.get(diff_url, headers=headers)
    pull_request_diff = diff_response.text

    diff_tokens = token_manager.count_tokens(pull_request_diff)
    if max_tokens:
        pull_request_diff, _ = token_manager.truncate_text(pull_request_diff, max_tokens // 2)
        print(f"PR diff tokens: {diff_tokens} (truncated to {max_tokens // 2})")
    
    comments_url = pull_request_data["comments_url"]
    review_comments_url = pull_request_data["review_comments_url"]
    comments_response = requests.get(comments_url, headers=headers)
    review_comments_response = requests.get(review_comments_url, headers=headers)
    comments_data = comments_response.json()
    review_comments_data = review_comments_response.json()

    all_comments = comments_data + review_comments_data
    all_comments.sort(key=lambda comment: comment.get("position") or float("inf"))

    formatted_text = f'<source type="github_pull_request" url="{pull_request_url}">\n'
    formatted_text += '<pull_request_info>\n'
    formatted_text += f'<title>{escape_xml(pull_request_data["title"])}</title>\n'
    
    description = pull_request_data["body"]
    desc_tokens = token_manager.count_tokens(description)
    if max_tokens:
        description, _ = token_manager.truncate_text(description, max_tokens // 4)
        if desc_tokens > max_tokens // 4:
            formatted_text += f'<description truncated="true" original_tokens="{desc_tokens}">'
        else:
            formatted_text += '<description>'
    else:
        formatted_text += '<description>'
    formatted_text += f'{escape_xml(description)}</description>\n'

    formatted_text += '<merge_details>\n'
    formatted_text += f'{escape_xml(pull_request_data["user"]["login"])} wants to merge {pull_request_data["commits"]} commit into {repo_owner}:{pull_request_data["base"]["ref"]} from {pull_request_data["head"]["label"]}\n'
    formatted_text += '</merge_details>\n'
    
    if max_tokens and diff_tokens > max_tokens // 2:
        formatted_text += f'<diff_and_comments truncated="true" original_tokens="{diff_tokens}">\n'
    else:
        formatted_text += '<diff_and_comments>\n'

    diff_lines = pull_request_diff.split("\n")
    comment_index = 0
    for line in diff_lines:
        formatted_text += f'{escape_xml(line)}\n'
        while comment_index < len(all_comments) and all_comments[comment_index].get("position") == diff_lines.index(line):
            comment = all_comments[comment_index]
            formatted_text += f'<review_comment>\n'
            formatted_text += f'<author>{escape_xml(comment["user"]["login"])}</author>\n'
            formatted_text += f'<content>{escape_xml(comment["body"])}</content>\n'
            formatted_text += f'<path>{escape_xml(comment["path"])}</path>\n'
            formatted_text += f'<line>{comment["original_line"]}</line>\n'
            formatted_text += '</review_comment>\n'
            comment_index += 1

    formatted_text += '</diff_and_comments>\n'
    formatted_text += '</pull_request_info>\n'

    repo_url = f"https://github.com/{repo_owner}/{repo_name}"
    remaining_tokens = max_tokens - token_manager.count_tokens(formatted_text) if max_tokens else None
    repo_content, extension_tokens = process_github_repo(repo_url, excluded_dirs, max_tokens=remaining_tokens, excluded_exts=excluded_exts)
    
    formatted_text += '<repository>\n'
    formatted_text += repo_content
    formatted_text += '</repository>\n'
    formatted_text += '</source>'

    print(f"Pull request {pull_request_number} and repository content processed successfully.")

    return formatted_text, extension_tokens

def process_github_issue(issue_url, excluded_dirs=None, max_tokens=None, excluded_exts=None):
    try:
        headers = get_github_headers()
    except EnvironmentError as e:
        raise EnvironmentError(f"Cannot process GitHub issue: {str(e)}")

    url_parts = issue_url.split("/")
    repo_owner = url_parts[3]
    repo_name = url_parts[4]
    issue_number = url_parts[-1]

    api_base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}"

    response = requests.get(api_base_url, headers=headers)
    issue_data = response.json()

    comments_url = issue_data["comments_url"]
    comments_response = requests.get(comments_url, headers=headers)
    comments_data = comments_response.json()

    formatted_text = f'<source type="github_issue" url="{issue_url}">\n'
    formatted_text += '<issue_info>\n'
    formatted_text += f'<title>{escape_xml(issue_data["title"])}</title>\n'
    formatted_text += f'<description>{escape_xml(issue_data["body"])}</description>\n'
    formatted_text += '<comments>\n'

    for comment in comments_data:
        formatted_text += '<comment>\n'
        formatted_text += f'<author>{escape_xml(comment["user"]["login"])}</author>\n'
        formatted_text += f'<content>{escape_xml(comment["body"])}</content>\n'

        code_snippets = re.findall(r'https://github.com/.*#L\d+-L\d+', comment['body'])
        for snippet_url in code_snippets:
            url_parts = snippet_url.split("#")
            file_url = url_parts[0].replace("/blob/", "/raw/")
            line_range = url_parts[1]
            start_line, end_line = map(int, line_range.split("-")[0][1:]), map(int, line_range.split("-")[1][1:])

            file_response = requests.get(file_url, headers=headers)
            file_content = file_response.text

            code_lines = file_content.split("\n")[start_line-1:end_line]
            code_snippet = "\n".join(code_lines)

            formatted_text += '<code_snippet>\n'
            formatted_text += f'<![CDATA[{code_snippet}]]>\n'
            formatted_text += '</code_snippet>\n'

        formatted_text += '</comment>\n'

    formatted_text += '</comments>\n'
    formatted_text += '</issue_info>\n'

    repo_url = f"https://github.com/{repo_owner}/{repo_name}"
    repo_content, extension_tokens = process_github_repo(repo_url, excluded_dirs, max_tokens=max_tokens, excluded_exts=excluded_exts)
    
    formatted_text += '<repository>\n'
    formatted_text += repo_content
    formatted_text += '</repository>\n'
    formatted_text += '</source>'

    print(f"Issue {issue_number} and repository content processed successfully.")

    return formatted_text, extension_tokens

def get_source_name(input_path):
    """Generate a name for the output files based on the input source"""
    if "github.com" in input_path:
        parts = input_path.split("github.com/")[-1].split("/")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    elif os.path.isdir(input_path):
        return os.path.basename(input_path)
    return "output"
