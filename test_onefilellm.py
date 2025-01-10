import unittest
import os
import tempfile
import shutil
from onefilellm import process_github_repo, process_arxiv_pdf, process_local_folder, fetch_youtube_transcript, crawl_and_extract_text, process_doi_or_pmid, process_github_pull_request, process_github_issue

class TestDataAggregation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_github_repo(self):
        print("\nTesting GitHub repository processing...")
        repo_url = "https://github.com/jimmc414/onefilellm"
        repo_content, _ = process_github_repo(repo_url)
        self.assertIsInstance(repo_content, str)
        self.assertGreater(len(repo_content), 0)
        self.assertIn('<source type="github_repository"', repo_content)
        print("GitHub repository processing test passed.")

    def test_arxiv_pdf(self):
        print("\nTesting arXiv PDF processing...")
        arxiv_url = "https://arxiv.org/abs/2401.14295"
        arxiv_content = process_arxiv_pdf(arxiv_url)
        self.assertIsInstance(arxiv_content, str)
        self.assertGreater(len(arxiv_content), 0)
        self.assertIn('<source type="arxiv_paper"', arxiv_content)
        print("arXiv PDF processing test passed.")

    def test_local_folder(self):
        print("\nTesting local folder processing...")
        local_path = os.path.dirname(os.path.abspath(__file__))  # Use the directory of the test file
        local_content, _ = process_local_folder(local_path)
        self.assertIsInstance(local_content, str)
        self.assertGreater(len(local_content), 0)
        self.assertIn('<source type="local_directory"', local_content)
        print("Local folder processing test passed.")

    def test_local_folder_without_github_token(self):
        print("\nTesting local folder processing without GitHub token...")
        # Store original token
        original_token = os.environ.get('GITHUB_TOKEN')
        
        try:
            # Remove GitHub token from environment
            if 'GITHUB_TOKEN' in os.environ:
                del os.environ['GITHUB_TOKEN']
            
            # Process a local folder
            local_path = os.path.dirname(os.path.abspath(__file__))
            from onefilellm.processor import process_input
            from rich.console import Console
            
            # Create a test output directory
            test_output_dir = os.path.join(self.temp_dir, "test_output")
            os.makedirs(test_output_dir, exist_ok=True)
            
            # Process the local folder
            process_input(
                local_path,
                os.getcwd(),
                Console(),
                output_dir=test_output_dir
            )
            
            # Check that output files were created
            output_base = os.path.join(test_output_dir, os.path.basename(local_path))
            self.assertTrue(os.path.exists(f"{output_base}_uncompressed.txt"))
            self.assertTrue(os.path.exists(f"{output_base}_compressed.txt"))
            
            print("Local folder processing without GitHub token test passed.")
        finally:
            # Restore original token
            if original_token is not None:
                os.environ['GITHUB_TOKEN'] = original_token

    def test_youtube_transcript(self):
        print("\nTesting YouTube transcript fetching...")
        video_url = "https://www.youtube.com/watch?v=KZ_NlnmPQYk"
        transcript = fetch_youtube_transcript(video_url)
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0)
        self.assertIn('<source type="youtube_transcript"', transcript)
        print("YouTube transcript fetching test passed.")

    def test_webpage_crawl(self):
        print("\nTesting webpage crawling and text extraction...")
        webpage_url = "https://llm.datasette.io/en/stable/"
        max_depth = 1
        include_pdfs = False
        ignore_epubs = True
        crawl_result = crawl_and_extract_text(webpage_url, max_depth, include_pdfs, ignore_epubs)
        self.assertIsInstance(crawl_result, dict)
        self.assertIn('content', crawl_result)
        self.assertIn('processed_urls', crawl_result)
        self.assertGreater(len(crawl_result['content']), 0)
        self.assertGreater(len(crawl_result['processed_urls']), 0)
        self.assertIn('<source type="web_documentation"', crawl_result['content'])
        print("Webpage crawling and text extraction test passed.")

    def test_process_doi(self):
        print("\nTesting DOI processing...")
        doi = "10.1053/j.ajkd.2017.08.002"
        doi_content = process_doi_or_pmid(doi)
        self.assertIsInstance(doi_content, str)
        self.assertGreater(len(doi_content), 0)
        self.assertIn('<source type="sci_hub_paper"', doi_content)
        print("DOI processing test passed.")

    def test_process_pmid(self):
        print("\nTesting PMID processing...")
        pmid = "29203127"
        pmid_content = process_doi_or_pmid(pmid)
        self.assertIsInstance(pmid_content, str)
        self.assertGreater(len(pmid_content), 0)
        self.assertIn('<source type="sci_hub_paper"', pmid_content)
        print("PMID processing test passed.")

    def test_process_github_pull_request(self):
        print("\nTesting GitHub pull request processing...")
        pull_request_url = "https://github.com/dear-github/dear-github/pull/102"
        pull_request_content, _ = process_github_pull_request(pull_request_url)
        self.assertIsInstance(pull_request_content, str)
        self.assertGreater(len(pull_request_content), 0)
        self.assertIn('<source type="github_pull_request"', pull_request_content)
        self.assertIn('<pull_request_info>', pull_request_content)
        self.assertIn('<repository>', pull_request_content)
        print("GitHub pull request processing test passed.")

    def test_process_github_issue(self):
        print("\nTesting GitHub issue processing...")
        issue_url = "https://github.com/isaacs/github/issues/1191"
        issue_content, _ = process_github_issue(issue_url)
        self.assertIsInstance(issue_content, str)
        self.assertGreater(len(issue_content), 0)
        self.assertIn('<source type="github_issue"', issue_content)
        self.assertIn('<issue_info>', issue_content)
        self.assertIn('<repository>', issue_content)
        print("GitHub issue processing test passed.")

    def test_token_count_consistency(self):
        print("\nTesting token count consistency...")
        # Process a local folder to get both extension tokens and total counts
        repo_url = "https://github.com/yunshun/HumanBreast10X" 
        content, extension_tokens = process_github_repo(repo_url)
        
        # Calculate total tokens from extensions (excluding XML overhead)
        content_tokens = sum(tokens for ext, tokens in extension_tokens.items() if ext != '.xml')
        xml_overhead_tokens = extension_tokens.get('.xml', 0)
        total_extension_tokens = content_tokens + xml_overhead_tokens
        
        # Get uncompressed count
        from onefilellm.utils import token_manager
        uncompressed_count = token_manager.count_tokens(content)
        
        # Calculate XML overhead percentage
        xml_overhead_percentage = (xml_overhead_tokens / total_extension_tokens) * 100 if total_extension_tokens > 0 else 0
        
        print(f"\nToken count analysis:")
        print("Extension tokens by type:", extension_tokens)
        print(f"Content tokens (excluding XML): {content_tokens:,}")
        print(f"XML overhead tokens: {xml_overhead_tokens:,}")
        print(f"Total tokens: {total_extension_tokens:,}")
        print(f"Uncompressed count: {uncompressed_count:,}")
        print(f"XML overhead: {xml_overhead_percentage:.1f}%")
        
        # The XML overhead should be reasonable (typically less than 25%)
        self.assertLess(xml_overhead_percentage, 25, 
                       "XML overhead is unusually high (>25%), might indicate an issue")
        
        # The XML overhead should be at least 1% (if it's less, we might be missing tags)
        self.assertGreater(xml_overhead_percentage, 1,
                          "XML overhead is unusually low (<1%), might be missing XML tags")
        
        # The total tokens should match the uncompressed count within a small margin
        token_diff_percentage = abs(total_extension_tokens - uncompressed_count) / total_extension_tokens * 100
        self.assertLess(token_diff_percentage, 1,
                       f"Token count mismatch: {token_diff_percentage:.1f}% difference between total tokens and uncompressed count")
        
        print("Token count consistency test passed.")

if __name__ == "__main__":
    unittest.main()
