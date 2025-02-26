import os
from datetime import datetime

class ReportGenerator:
    """Generates HTML reports for analysis results"""
    
    def __init__(self, output_dir='.'):
        """Initialize the report generator"""
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def create_html_report(self, results):
        """Create an HTML report of the results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        from_cache = results['cache_stats'].get('from_cached_analysis', False)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wikipedia Category Analysis: {results['category']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .stats {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .cache-info {{ font-size: 0.9em; color: #666; margin-top: 10px; }}
                .cached-label {{ background-color: #dff0d8; color: #3c763d; padding: 3px 6px; border-radius: 3px; font-size: 0.8em; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .pages-list {{ max-height: 200px; overflow-y: auto; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Wikipedia Category Analysis</h1>
                <div class="stats">
                    <h2>
                        Category: {results['category']}
                        {f'<span class="cached-label">CACHED ANALYSIS</span>' if from_cache else ''}
                    </h2>
                    <p>Number of pages analyzed: {results['page_count']}</p>
                    <p class="cache-info">
                        Analysis generated: {timestamp}<br>
                        Categories in cache: {results['cache_stats']['categories_cached']}<br>
                        Pages in cache: {results['cache_stats']['pages_cached']}<br>
                        Analyses in cache: {results['cache_stats']['analyses_cached']}
                    </p>
                </div>
                
                <h2>Pages in this category:</h2>
                <div class="pages-list">
                    <ul>
        """
        
        for page in results['page_titles']:
            html_content += f"<li><a href='https://en.wikipedia.org/wiki/{page.replace(' ', '_')}' target='_blank'>{page}</a></li>\n"
        
        html_content += """
                    </ul>
                </div>
                
                <h2>Word Frequency Analysis</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Word</th>
                        <th>Frequency</th>
                    </tr>
        """
        
        for i, (word, count) in enumerate(results['word_frequencies']):
            html_content += f"<tr><td>{i+1}</td><td>{word}</td><td>{count}</td></tr>\n"
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        output_file = os.path.join(self.output_dir, 'wikipedia_analysis.html')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file 