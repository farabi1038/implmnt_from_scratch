import re
import html
import nltk
from nltk.corpus import stopwords
from collections import Counter

class TextProcessor:
    """Processes and analyzes text data"""
    
    def __init__(self):
        """Initialize the text processor and download required resources"""
        self.download_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
    
    def download_nltk_resources(self):
        """Download NLTK resources if not already available"""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
    
    def process_text(self, text):
        """Process text to extract non-common words."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML entities
        text = html.unescape(text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Remove stopwords (common words) and short words
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return words
    
    def count_word_frequencies(self, words):
        """Count word frequencies and return sorted results"""
        # Count word frequencies
        word_counts = Counter(words)
        
        # Sort by frequency (most common first)
        return word_counts.most_common() 