import requests
import time

class WikipediaAPI:
    """Handles interactions with the Wikipedia API"""
    
    def __init__(self, cache_manager):
        """Initialize with a cache manager"""
        self.cache_manager = cache_manager
        self.api_url = 'https://en.wikipedia.org/w/api.php'
    
    def get_pages_in_category(self, category):
        """Get all page titles in a given Wikipedia category."""
        # Check cache first
        cached_pages = self.cache_manager.get_category_pages(category)
        if cached_pages:
            print(f"Using cached list of pages for category '{category}'")
            return cached_pages
        
        print(f"Fetching pages for category '{category}' from Wikipedia API")
        pages = []
        continue_token = None
        
        while True:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': 500,
                'format': 'json'
            }
            
            if continue_token:
                params['cmcontinue'] = continue_token
            
            response = requests.get(self.api_url, params=params)
            data = response.json()
            
            if 'query' in data and 'categorymembers' in data['query']:
                for member in data['query']['categorymembers']:
                    if member['ns'] == 0:  # Only include regular articles (namespace 0)
                        pages.append(member['title'])
            
            if 'continue' in data and 'cmcontinue' in data['continue']:
                continue_token = data['continue']['cmcontinue']
            else:
                break
        
        # Store in cache
        self.cache_manager.cache_category_pages(category, pages)
        
        return pages
    
    def get_page_content(self, page_title):
        """Get the plain text content of a Wikipedia page."""
        # Check cache first
        cached_content = self.cache_manager.get_page_content(page_title)
        if cached_content is not None:
            return cached_content
        
        print(f"  Fetching content for '{page_title}' from Wikipedia API")
        params = {
            'action': 'query',
            'prop': 'extracts',
            'titles': page_title,
            'explaintext': True,
            'format': 'json'
        }
        
        response = requests.get(self.api_url, params=params)
        data = response.json()
        
        pages = data['query']['pages']
        content = ''
        for page_id in pages:
            content = pages[page_id].get('extract', '')
        
        # Store in cache
        self.cache_manager.cache_page_content(page_title, content)
        
        # Respect rate limits
        time.sleep(0.1)
        
        return content 