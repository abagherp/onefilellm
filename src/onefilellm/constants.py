"""Constants used throughout the onefilellm package"""

# Directory exclusions
DEFAULT_EXCLUDED_DIRS = {
    '.venv', '__pycache__', '.git', 'node_modules',
    '.pytest_cache', '.idea', '.vs', '.next'
}

# File types
ALLOWED_EXTENSIONS = [
    '.py', '.txt', '.js', '.tsx', '.ts', '.md', 
    '.cjs', '.html', '.json', '.ipynb', '.h',
    '.localhost', '.sh', '.yaml', '.example',
    '.jsx', '.csv', '.R', '.Rmd'
]

EXCLUDED_FILES = [
    'compressed_output.txt',
    'uncompressed_output.txt',
    'processed_urls.txt',
    'instruction.md',
    'instructions.md',
    'package-lock.json',
    '*_uncompressed.txt',
    '*_compressed.txt',
    '*.png',
    '*.jpg',
    '*.jpeg',
    '*.gif',
    '*.svg',
    '*.ico',
]

# Token settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_ENCODING = "cl100k_base" 