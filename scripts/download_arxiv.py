import arxiv
import os
from loguru import logger

def download_papers(queries: list[str], max_results_per_query: int = 5):
    """
    Downloads arXiv papers as PDFs into the data/raw/ directory.
    """
    target_dir = os.path.join("data", "raw")
    os.makedirs(target_dir, exist_ok=True)
    
    logger.info(f"Starting arXiv paper download. Target directory: {target_dir}")
    
    client = arxiv.Client()
    
    for query in queries:
        logger.info(f"Searching for: '{query}'")
        search = arxiv.Search(
            query=query,
            max_results=max_results_per_query,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        try:
            for result in client.results(search):
                # Format a safe filename
                clean_title = "".join(c for c in result.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                clean_title = clean_title.replace(' ', '_')
                
                filename = f"{result.get_short_id()}_{clean_title}.pdf"
                filepath = os.path.join(target_dir, filename)
                
                if os.path.exists(filepath):
                    logger.info(f"Skipping (already exists): {filename}")
                    continue
                    
                logger.info(f"Downloading: {result.title}")
                try:
                    result.download_pdf(dirpath=target_dir, filename=filename)
                except Exception as dl_err:
                    logger.error(f"Failed to download {filename}: {dl_err}")
                    
        except Exception as search_err:
            logger.error(f"Search failed for query '{query}': {search_err}")

    logger.info("Finished downloading arXiv papers.")

if __name__ == "__main__":
    queries = [
        "cat:cs.CL AND embedding", 
        "retrieval augmented generation"
    ]
    download_papers(queries=queries, max_results_per_query=3)
