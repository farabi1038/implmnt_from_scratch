# Wikipedia Category Word Analyzer

A powerful tool for analyzing word frequencies across Wikipedia categories, featuring both a command-line interface and an interactive web visualization.

## Overview

This project provides a suite of tools to analyze the frequency of non-common words across all pages within a specified Wikipedia category. It offers two main interfaces:

1. **Command-line script**: Process categories and generate HTML reports
2. **Interactive web application**: Visualize results with a word cloud and detailed frequency data

The analyzer intelligently caches both raw Wikipedia data and processed results, making repeated analyses fast and efficient.

## Features

- **Category-based analysis**: Process any Wikipedia category to extract word frequencies
- **Word Cloud Visualization**: Interactive visualization where word size represents frequency
- **Raw Frequency View**: Complete tabular data of all word frequencies with percentages
- **Multiple Color Palettes**: 8 different color schemes for customizing visualizations
- **Intelligent Caching**: Multi-level caching system that stores:
  - Lists of pages in categories (expires after 7 days)
  - Page content (expires after 30 days)
  - Analysis results (expires after 1 day)
- **Statistics Dashboard**: View key metrics about your analyzed category
- **Responsive Design**: Works on both desktop and mobile devices

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/farabi1038/https://github.com/farabi1038/implmnt_from_scratchwikipedia-category-analyzer.git
   cd wikipedia-category-analyzer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install packages manually:
   ```bash
   pip install flask requests nltk
   ```

3. Initial NLTK data download (optional, will be done automatically on first run):
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

## Usage

### Command-line Interface

Run the command-line tool to analyze a category and generate an HTML report:

```bash
python wiki_category_analyzer.py Large_language_models
```

#### Options:

- `--reset-cache`: Clear the cache and fetch fresh data
- `--reset-analysis`: Re-analyze using cached Wikipedia data
- `--cache-dir PATH`: Specify a custom cache directory
- `--output-dir PATH`: Specify where to save the output report

Example:
```bash
python wiki_category_analyzer.py Artificial_intelligence --reset-analysis --output-dir reports
```

### Web Application

Start the Flask web server:

```bash
python app.py
```

Then open your browser and navigate to http://127.0.0.1:5000/

The web interface allows you to:
- Enter any Wikipedia category name
- Switch between word cloud and raw frequency views
- Select different color palettes
- Toggle frequency display in the word cloud
- Explore word frequencies with interactive tooltips

## Project Structure

```
├── wiki_category_analyzer.py     # Command-line script
├── app.py                        # Flask web application
├── modules/                      # Modular components
│   ├── __init__.py
│   ├── cache_manager.py          # Caching functionality
│   ├── color_palette.py          # Color scheme definitions
│   ├── text_processor.py         # Text analysis and processing
│   ├── wikipedia_api.py          # MediaWiki API interaction
│   └── report_generator.py       # HTML report creation
├── templates/                    # Flask HTML templates
│   └── index.html                # Web interface 
├── static/                       # Static web assets
│   └── style.css                 # Custom styling
├── cache/                        # Cache storage (created on first run)
│   └── wikipedia_cache.json      # Cache data
└── requirements.txt              # Project dependencies
```

## Technical Details

### Data Processing Pipeline

1. Retrieve all pages in the specified Wikipedia category
2. Download and extract plaintext content from each page
3. Process text to remove common words, punctuation, and digits
4. Count word frequencies across all pages
5. Sort by frequency and display/visualize results

### Caching System

The multi-level caching system optimizes performance:

- **Category caching**: Stores lists of pages in each category
- **Content caching**: Stores the text content of individual pages
- **Analysis caching**: Stores processed frequency results

Each cache level has its own expiration period, balancing freshness with performance.

### Word Cloud Generation

The word cloud visualization is generated using:
- D3.js for SVG manipulation
- d3-cloud layout for word cloud positioning
- Custom scaling to ensure proper word sizing
- Interactive elements like hovering and tooltips

## Customization

### Color Palettes

The application includes 8 built-in color palettes:
- Classic
- Pastel
- Dark
- Earth
- Ocean
- Sunset
- Forest
- Monochrome

To add new palettes, edit the `ColorPalette.get_all_color_palettes()` method in `modules/color_palette.py`.

### Word Filtering

The analyzer filters out common words (stopwords) and short words (< 3 letters). To customize this behavior, modify the `process_text()` method in `modules/text_processor.py`.

## Troubleshooting

### Common Issues

- **Missing NLTK Data**: If you get an error about missing NLTK data, run:
  ```python
  import nltk
  nltk.download('stopwords')
  ```

- **Rate Limiting**: Wikipedia's API has rate limits. The code includes delays to respect these limits, but if you're analyzing many large categories, you might need to increase the delay in `get_page_content()` in `modules/wikipedia_api.py`.

- **Cache Issues**: If you encounter strange results, try running with `--reset-cache` to start fresh.

### Performance Tips

- For large categories (100+ pages), the initial analysis can take several minutes
- Subsequent analyses of the same category will be much faster due to caching
- The web interface can handle displaying thousands of words, but performance is best with 100-500 words

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaWiki API](https://www.mediawiki.org/wiki/API:Main_page) for providing access to Wikipedia data
- [NLTK](https://www.nltk.org/) for text processing capabilities
- [D3.js](https://d3js.org/) and [d3-cloud](https://github.com/jasondavies/d3-cloud) for visualization
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for UI components 