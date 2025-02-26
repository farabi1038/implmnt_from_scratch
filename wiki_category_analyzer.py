#!/usr/bin/env python3

import argparse
import webbrowser
import os
from modules.cache_manager import CacheManager
from modules.wikipedia_api import WikipediaAPI
from modules.text_processor import TextProcessor
from modules.report_generator import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description='Analyze word frequencies in a Wikipedia category')
    parser.add_argument('category', help='Wikipedia category name (without "Category:" prefix)')
    parser.add_argument('--reset-cache', action='store_true', help='Ignore and reset the cache')
    parser.add_argument('--reset-analysis', action='store_true', help='Re-analyze even if cached analysis exists')
    parser.add_argument('--cache-dir', default='cache', help='Directory to store cache files')
    parser.add_argument('--output-dir', default='.', help='Directory to store output files')
    args = parser.parse_args()
    
    # Initialize components
    cache_manager = CacheManager(
        cache_dir=args.cache_dir,
        reset_cache=args.reset_cache
    )
    
    # If reset_analysis is requested, remove just the analysis for this category
    if args.reset_analysis:
        cache_manager.remove_analysis(args.category)
    
    # Initialize API and processors
    wiki_api = WikipediaAPI(cache_manager)
    text_processor = TextProcessor()
    report_generator = ReportGenerator(output_dir=args.output_dir)
    
    # Check if we have cached analysis
    cached_analysis = cache_manager.get_cached_analysis(args.category)
    if cached_analysis:
        results = cached_analysis
        results['cache_stats'] = cache_manager.get_stats()
        results['cache_stats']['from_cached_analysis'] = True
    else:
        # Get pages in category
        print(f"Analyzing category: {args.category}")
        pages = wiki_api.get_pages_in_category(args.category)
        print(f"Found {len(pages)} pages in the category")
        
        # Process pages and collect words
        all_words = []
        for i, page_title in enumerate(pages):
            print(f"Processing page {i+1}/{len(pages)}: {page_title}")
            content = wiki_api.get_page_content(page_title)
            words = text_processor.process_text(content)
            all_words.extend(words)
        
        # Analyze word frequencies
        word_frequencies = text_processor.count_word_frequencies(all_words)
        
        # Create results
        results = {
            "category": args.category,
            "page_count": len(pages),
            "page_titles": pages,
            "word_frequencies": word_frequencies,
            "cache_stats": cache_manager.get_stats()
        }
        
        # Cache the analysis results
        cache_manager.cache_analysis_results(args.category, results)
    
    # Generate and open the report
    html_path = report_generator.create_html_report(results)
    print(f"\nAnalysis complete! Opening report in browser...")
    webbrowser.open('file://' + os.path.abspath(html_path))

if __name__ == '__main__':
    main() 