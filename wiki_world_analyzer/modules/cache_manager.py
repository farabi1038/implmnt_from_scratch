import os
import json
import time
from datetime import datetime

class CacheManager:
    """Manages caching of Wikipedia data and analysis results"""
    
    def __init__(self, cache_dir='cache', reset_cache=False):
        """Initialize the cache manager"""
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, 'wikipedia_cache.json')
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Reset cache if requested
        if reset_cache and os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            print("Cache has been reset")
        
        # Load or initialize cache
        self.cache = self.load_cache()
    
    def load_cache(self):
        """Load cache from file if it exists, otherwise return empty cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    print(f"Loaded cache with {len(cache.get('categories', {}))} categories and {len(cache.get('pages', {}))} pages")
                    if 'analysis_results' in cache:
                        print(f"Cache contains {len(cache['analysis_results'])} analysis results")
                    return cache
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Return empty cache structure
        return {
            "categories": {},       # category_name -> list of page titles
            "pages": {},            # page_title -> page content
            "analysis_results": {}, # category_name -> word frequency analysis results
            "last_updated": {}      # category_name or page_title -> timestamp
        }
    
    def save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_category_pages(self, category):
        """Get category pages from cache if available and not expired"""
        now = time.time()
        cache_expiry_days = 7  # Category lists expire after 7 days
        cache_expiry_seconds = cache_expiry_days * 24 * 60 * 60
        
        if (category in self.cache['categories'] and 
            category in self.cache['last_updated'] and
            now - self.cache['last_updated'][category] < cache_expiry_seconds):
            return self.cache['categories'][category]
        
        return None
    
    def cache_category_pages(self, category, pages):
        """Store category pages in cache"""
        self.cache['categories'][category] = pages
        self.cache['last_updated'][category] = time.time()
        self.save_cache()
    
    def get_page_content(self, page_title):
        """Get page content from cache if available and not expired"""
        now = time.time()
        cache_expiry_days = 30  # Page content expires after 30 days
        cache_expiry_seconds = cache_expiry_days * 24 * 60 * 60
        
        if (page_title in self.cache['pages'] and 
            page_title in self.cache['last_updated'] and
            now - self.cache['last_updated'][page_title] < cache_expiry_seconds):
            return self.cache['pages'][page_title]
        
        return None
    
    def cache_page_content(self, page_title, content):
        """Store page content in cache"""
        self.cache['pages'][page_title] = content
        self.cache['last_updated'][page_title] = time.time()
        
        # Save cache periodically (every 5 pages)
        if len(self.cache['pages']) % 5 == 0:
            self.save_cache()
    
    def get_cached_analysis(self, category):
        """Check if we have a valid cached analysis result for this category"""
        now = time.time()
        cache_expiry_days = 1  # Analysis results expire after 1 day
        cache_expiry_seconds = cache_expiry_days * 24 * 60 * 60
        
        analysis_key = f"analysis_{category}"
        
        if (analysis_key in self.cache['analysis_results'] and 
            analysis_key in self.cache['last_updated'] and 
            now - self.cache['last_updated'][analysis_key] < cache_expiry_seconds):
            
            print(f"Using cached analysis results for category '{category}'")
            cached_data = self.cache['analysis_results'][analysis_key]
            
            # Reconstruct full results from cached data
            return {
                "category": cached_data["category"],
                "page_count": cached_data["page_count"],
                "page_titles": cached_data["page_title_refs"],
                "word_frequencies": cached_data["word_frequencies"]
            }
        
        return None
    
    def cache_analysis_results(self, category, results):
        """Store analysis results in cache"""
        analysis_key = f"analysis_{category}"
        
        # Store only the essential data, not the page titles (to save space)
        cache_data = {
            "category": results["category"],
            "page_count": results["page_count"],
            "word_frequencies": results["word_frequencies"],
            "page_title_refs": results["page_titles"]
        }
        
        self.cache['analysis_results'][analysis_key] = cache_data
        self.cache['last_updated'][analysis_key] = time.time()
        self.save_cache()
    
    def remove_analysis(self, category):
        """Remove analysis for a specific category"""
        analysis_key = f"analysis_{category}"
        
        if analysis_key in self.cache.get('analysis_results', {}):
            del self.cache['analysis_results'][analysis_key]
            if analysis_key in self.cache.get('last_updated', {}):
                del self.cache['last_updated'][analysis_key]
            
            self.save_cache()
            print(f"Removed cached analysis for '{category}'")
    
    def get_stats(self):
        """Get cache statistics"""
        return {
            "categories_cached": len(self.cache['categories']),
            "pages_cached": len(self.cache['pages']),
            "analyses_cached": len(self.cache['analysis_results']),
            "from_cached_analysis": False
        } 