from flask import Flask, render_template, request, jsonify
import os
import json
from modules.cache_manager import CacheManager
from modules.wikipedia_api import WikipediaAPI
from modules.text_processor import TextProcessor
from modules.color_palette import ColorPalette

app = Flask(__name__)

# Initialize components
cache_manager = CacheManager(cache_dir='cache')
wiki_api = WikipediaAPI(cache_manager)
text_processor = TextProcessor()

@app.route('/')
def index():
    """Render the main page"""
    # Get all available color palettes
    palettes = ColorPalette.get_all_color_palettes()
    palette_data = [{"name": p.get_name(), "colors": p.get_colors()} for p in palettes]
    
    return render_template('index.html', palettes=palette_data)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a category and return word frequencies"""
    category = request.form.get('category', 'Large_language_models')
    reset_cache = request.form.get('reset_cache') == 'true'
    
    if reset_cache:
        cache_manager.remove_analysis(category)
    
    # Check if we have cached analysis
    cached_analysis = cache_manager.get_cached_analysis(category)
    if cached_analysis:
        results = cached_analysis
        results['cache_stats'] = cache_manager.get_stats()
        results['cache_stats']['from_cached_analysis'] = True
    else:
        # Get pages in category
        pages = wiki_api.get_pages_in_category(category)
        
        # Process pages and collect words
        all_words = []
        for page_title in pages:
            content = wiki_api.get_page_content(page_title)
            words = text_processor.process_text(content)
            all_words.extend(words)
        
        # Analyze word frequencies
        word_frequencies = text_processor.count_word_frequencies(all_words)
        
        # Create results
        results = {
            "category": category,
            "page_count": len(pages),
            "page_titles": pages,
            "word_frequencies": word_frequencies,
            "cache_stats": cache_manager.get_stats()
        }
        
        # Cache the analysis results
        cache_manager.cache_analysis_results(category, results)
    
    return jsonify(results)

if __name__ == '__main__':
    # Ensure the templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        
    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    
    app.run(debug=True) 