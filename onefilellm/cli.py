import os
import sys
from .processor import process_input
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt

def main():
    console = Console()

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

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = Prompt.ask("\n[bold dodger_blue1]Enter the path or URL[/bold dodger_blue1]", console=console)

    # Get the current working directory
    working_dir = os.getcwd()
    
    try:
        process_input(input_path, working_dir, console)
    except Exception as e:
        console.print(f"\n[bold red]An error occurred:[/bold red] {str(e)}")
        console.print("\nPlease check your input and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 