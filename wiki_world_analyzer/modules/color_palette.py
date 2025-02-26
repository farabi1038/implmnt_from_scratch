class ColorPalette:
    """Base class for color palettes containing 6 hex colors"""
    
    def __init__(self, name, colors):
        """Initialize with a name and list of 6 hex colors"""
        self.name = name
        if len(colors) != 6:
            raise ValueError("ColorPalette must contain exactly 6 colors")
        self.colors = colors
    
    def get_colors(self):
        """Return the list of colors"""
        return self.colors
    
    def get_name(self):
        """Return the palette name"""
        return self.name
    
    @staticmethod
    def get_all_color_palettes():
        """Return a list of all available color palettes"""
        return [
            ColorPalette("Classic", ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]),
            ColorPalette("Pastel", ["#a8d8ea", "#aa96da", "#fcbad3", "#ffffd2", "#a1de93", "#f3a683"]),
            ColorPalette("Dark", ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#533483", "#11052c"]),
            ColorPalette("Earth", ["#5d4037", "#795548", "#a1887f", "#8d6e63", "#6d4c41", "#4e342e"]),
            ColorPalette("Ocean", ["#1e3d59", "#f5f0e1", "#ff6e40", "#ffc13b", "#19b5fe", "#006a71"]),
            ColorPalette("Sunset", ["#ff7e5f", "#feb47b", "#ffac81", "#ffd2bc", "#f0997d", "#ffb7b2"]),
            ColorPalette("Forest", ["#2c5f2d", "#97bc62", "#d0e562", "#1e441e", "#5a8c5a", "#a4c2a5"]),
            ColorPalette("Monochrome", ["#000000", "#333333", "#666666", "#999999", "#cccccc", "#ffffff"]),
        ] 