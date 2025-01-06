import os
import sys
from .processor import process_input
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt
import argparse

def main():
    console = Console()
    parser = argparse.ArgumentParser(description='Process files for LLM ingestion')
    parser.add_argument('input_path', nargs='?', help='Path or URL to process')
    parser.add_argument('--exclude', '-e', action='append', help='Directories to exclude (can be used multiple times)')
    parser.add_argument('--max-tokens', '-m', type=int, help='Maximum number of tokens per file')
    parser.add_argument('--output-dir', '-o', help='Directory to store output files')
    parser.add_argument('--exclude-ext', '-x', action='append', help='File extensions to exclude (can be used multiple times, e.g., -x .js -x .py)')
    parser.add_argument('--max-depth', '-d', type=int, default=2, help='Maximum depth for URL crawling (default: 2)')
    
    args = parser.parse_args()
    input_path = args.input_path
    excluded_dirs = set(args.exclude) if args.exclude else set()
    max_tokens = args.max_tokens
    output_dir = args.output_dir
    excluded_exts = set(args.exclude_ext) if args.exclude_ext else set()
    max_depth = args.max_depth

    if not input_path:
        intro_text = Text("\nInput Paths or URLs Processed:\n", style="dodger_blue1")
        input_types = [
            ("• Local folder path (flattens all files into text)", "bright_white"),
            ("• GitHub repository URL (flattens all files into text)", "bright_white"),
            ("• GitHub pull request URL (PR + Repo)", "bright_white"),
            ("• GitHub issue URL (Issue + Repo)", "bright_white"),
            ("• Documentation URL (base URL)", "bright_white"),
            ("• YouTube video URL (to fetch transcript)", "bright_white"),
            ("• ArXiv Paper URL", "bright_white"),
            ("• DOI or PMID to search on Sci-Hub", "bright_white"),
        ]

        for input_type, color in input_types:
            intro_text.append(f"\n{input_type}", style=color)

        intro_panel = Panel(
            intro_text,
            expand=False,
            border_style="bold",
            title="[bright_white]Copy to File and Clipboard[/bright_white]",
            title_align="center",
            padding=(1, 1),
        )
        console.print(intro_panel)
        input_path = Prompt.ask("\n[bold dodger_blue1]Enter the path or URL[/bold dodger_blue1]", console=console)

    # Get the current working directory
    working_dir = os.getcwd()
    
    try:
        process_input(input_path, working_dir, console, excluded_dirs, max_tokens, output_dir, excluded_exts, max_depth)
    except Exception as e:
        console.print(f"\n[bold red]An error occurred:[/bold red] {str(e)}")
        console.print("\nPlease check your input and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 