# import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from datetime import datetime
import re
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
import time
import requests

# Configure headers to mimic browser behavior
COMMON_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

def get_all_urls(sitemap_url):
    """Extract URLs from sitemap.xml with retry logic"""
    try:
        print(f"Fetching sitemap from {sitemap_url}")
        response = requests.get(sitemap_url, headers=COMMON_HEADERS, timeout=10)
        response.raise_for_status()
        
        # Verify valid XML content
        if 'xml' not in response.headers.get('Content-Type', ''):
            raise ValueError("Response is not XML format")
            
        soup = BeautifulSoup(response.content, 'lxml-xml')
        urls = [loc.text.strip() for loc in soup.find_all('loc') if loc.text.strip()]
        
        # Filter out non-content pages
        excluded_patterns = r'(wp-|feed|\.xml|\/category\/|\/tag\/|\.css|\.js|\.png|\.jpg|\.svg)'
        filtered_urls = [url for url in urls if not re.search(excluded_patterns, url)]
        
        print(f"Found {len(filtered_urls)} valid URLs after filtering")
        return filtered_urls
        
    except Exception as e:
        print(f"Error fetching sitemap: {str(e)}")
        print(f"HTTP Status Code: {response.status_code if 'response' in locals() else 'Unknown'}")
        return []

def scrape_and_store(sitemap_url, output_file):
    """Main scraping pipeline with data preservation"""
    urls = get_all_urls(sitemap_url)
    
    if not urls:
        print("No valid URLs found. Exiting.")
        return

    results = []
    failed_urls = []
    
    for url in tqdm(urls, desc="Scraping pages"):
        try:
            # Initialize loader PER URL
            loader = WebBaseLoader([url])
            
            # Load with retry logic
            docs = []
            for attempt in range(3):
                try:
                    docs = loader.load()
                    break
                except Exception as e:
                    if attempt < 2:
                        wait = 2 ** attempt
                        print(f"Retrying {url} in {wait} seconds...")
                        time.sleep(wait)
                    else:
                        raise
            
            if not docs:
                raise ValueError("No documents loaded")
                
            # Process document
            html2text = Html2TextTransformer()
            cleaned_docs = html2text.transform_documents(docs)
            
            metadata = {
                "source": url,
                "title": extract_title(docs[0].page_content),
                "scrape_date": datetime.now().isoformat(),
                "content_type": classify_content(url)
            }
            
            cleaned_content = post_process(cleaned_docs[0].page_content)
            
            results.append({
                "metadata": metadata,
                "raw_content": docs[0].page_content,
                "cleaned_content": cleaned_content
            })
            
        except Exception as e:
            print(f"Failed to scrape {url}: {str(e)}")
            failed_urls.append(url)
    
    
    # Add failure metadata
    if failed_urls:
        print(f"Failed to scrape {len(failed_urls)} URLs")
        results.append({
            "metadata": {
                "scrape_errors": failed_urls,
                "error_count": len(failed_urls)
            }
        })
    
    if validate_scrape(results):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "scrape_date": datetime.now().isoformat(),
                "sitemap_url": sitemap_url,
                "success_rate": f"{len(results)-len(failed_urls)}/{len(urls)}",
                "pages": results
            }, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(results)-len(failed_urls)} pages to {output_file}")
    else:
        print("Scraping failed validation - output not saved")

def validate_scrape(data):
    """Enhanced validation with error reporting"""
    if not data or len(data) == 0:
        print("Validation failed: No data collected")
        return False
        
    required_terms = ["occam", "advisory", "consulting"]
    min_content_length = 200
    valid_count = 0
    
    for item in data:
        if 'metadata' not in item:
            continue
            
        content = item.get('cleaned_content', '').lower()
        if len(content) < min_content_length:
            print(f"Validation warning: Short content in {item['metadata'].get('source', 'unknown')}")
            
        if any(term in content for term in required_terms):
            valid_count += 1
            
    if valid_count < 3:
        print(f"Validation failed: Only {valid_count} pages contain required terms")
        return False
        
    return True

def extract_title(html_content):
    """Extracts page title from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.find('title')
    return title.get_text().strip() if title else "No Title Found"

def classify_content(url):
    """Categorizes URL into content types"""
    if '/blog/' in url.lower():
        return 'blog_post'
    elif '/services/' in url.lower():
        return 'service'
    elif '/about/' in url.lower():
        return 'about_us'
    elif '/contact' in url.lower():
        return 'contact_info'
    else:
        return 'general_page'

def post_process(text):
    """Cleans and normalizes scraped text"""
    # Remove HTML entities and special characters
    text = re.sub(r'&\w+;', '', text)
    # Collapse repeated newlines
    text = re.sub(r'\n+', '\n', text)
    # Remove leading/trailing whitespace
    return text.strip()


# Rest of the helper functions remain the same

if __name__ == "__main__":
    SITEMAP_URL = "https://occamsadvisory.com/sitemap.xml"
    scrape_and_store(SITEMAP_URL, "occams_content.json")
