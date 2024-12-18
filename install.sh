#!/bin/bash

# Function to check if a Python package is installed
check_package() {
    python -c "import $1" 2>/dev/null
    return $?
}

# Install base requirements first
echo "Installing base requirements..."
pip install nltk requests beautifulsoup4 --quiet

# Now we can safely download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Install the rest of the package
echo "Installing onefilellm..."
pip install -e . --use-pep517 --no-cache-dir

echo "Installation complete!"