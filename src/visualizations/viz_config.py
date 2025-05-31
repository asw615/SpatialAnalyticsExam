#!/usr/bin/env python3
"""
Visualization Configuration
===========================

Centralized configuration for all visualization modules including colors,
fonts, figure settings, and style definitions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Figure settings
FIGURE_SIZE = (12, 8)
DPI = 300
FONT_SIZE = 12
TITLE_SIZE = 16
LABEL_SIZE = 12

# Font settings
FONT_SIZES = {
    'small': 10,
    'medium': 12,
    'large': 14,
    'x-large': 16,
    'xx-large': 18
}

# DPI settings for different output types
DPI_SETTINGS = {
    'screen': 100,
    'print': 300,
    'publication': 600
}

# Color schemes
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple  
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'info': '#87CEEB',         # Light blue
    'warning': '#FFD700',      # Gold
    'danger': '#DC143C',       # Crimson
    'light': '#F8F9FA',        # Light gray
    'dark': '#343A40'          # Dark gray
}

# Orienteering-specific colors
ORIENTEERING_COLORS = {
    'fast_terrain': '#90EE90',     # Light green - fast running
    'medium_terrain': '#FFD700',   # Gold - medium difficulty  
    'slow_terrain': '#FF6347',     # Tomato - difficult terrain
    'impassible': '#8B0000',       # Dark red - impassible areas
    'water': '#4169E1',            # Royal blue - water bodies
    'paths': '#8B4513',            # Saddle brown - paths/tracks
    'controls': '#FF1493',         # Deep pink - control points
    'routes': '#32CD32'            # Lime green - planned routes
}

# Terrain analysis colors
TERRAIN_COLORS = {
    'elevation_low': '#2E8B57',    # Sea green
    'elevation_high': '#8B4513',   # Saddle brown
    'slope_gentle': '#98FB98',     # Pale green
    'slope_steep': '#CD853F',      # Peru
    'vegetation_dense': '#228B22', # Forest green
    'vegetation_sparse': '#F0E68C' # Khaki
}

# Cost surface colormap
COST_COLORMAP = 'YlOrRd'  # Yellow-Orange-Red for cost surfaces

# Colormaps for different data types
COLORMAPS = {
    'elevation': 'terrain',
    'slope': 'YlOrBr',
    'cost': 'YlOrRd',
    'speed': 'RdYlGn',
    'density': 'Blues',
    'correlation': 'RdBu_r'
}

# Line styles
LINE_STYLES = {
    'actual': '-',          # Solid line
    'predicted': '--',      # Dashed line
    'optimal': '-.',        # Dash-dot line
    'reference': ':'        # Dotted line
}

# Marker styles
MARKERS = {
    'data': 'o',           # Circle
    'control': 's',        # Square
    'start': '^',          # Triangle up
    'finish': 'v',         # Triangle down
    'checkpoint': 'D'      # Diamond
}

# Alpha (transparency) values
ALPHA = {
    'background': 0.3,
    'overlay': 0.7,
    'highlight': 0.9,
    'subtle': 0.5
}

# Grid settings
GRID_SETTINGS = {
    'alpha': 0.3,
    'linestyle': '-',
    'linewidth': 0.5
}

# Legend settings
LEGEND_SETTINGS = {
    'frameon': True,
    'framealpha': 0.9,
    'fancybox': True,
    'shadow': True
}

def setup_matplotlib():
    """Setup matplotlib with consistent styling."""
    plt.rcParams.update({
        'figure.figsize': FIGURE_SIZE,
        'figure.dpi': DPI_SETTINGS['screen'],
        'savefig.dpi': DPI_SETTINGS['print'],
        'font.size': FONT_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': FONT_SIZES['small'],
        'ytick.labelsize': FONT_SIZES['small'],
        'legend.fontsize': FONT_SIZES['small'],
        'figure.titlesize': FONT_SIZES['x-large'],
        'axes.grid': True,
        'grid.alpha': GRID_SETTINGS['alpha'],
        'legend.frameon': LEGEND_SETTINGS['frameon'],
        'legend.framealpha': LEGEND_SETTINGS['framealpha']
    })

def get_color_palette(n_colors, palette_type='husl'):
    """Get a color palette with n colors."""
    if palette_type == 'orienteering':
        base_colors = list(ORIENTEERING_COLORS.values())
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        else:
            # Extend with seaborn palette
            return base_colors + sns.color_palette("husl", n_colors - len(base_colors))
    else:
        return sns.color_palette(palette_type, n_colors)

def save_figure_publication(fig, filepath, dpi=None):
    """Save figure with publication-quality settings."""
    if dpi is None:
        dpi = DPI_SETTINGS['publication']
    
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')

# Initialize matplotlib settings
setup_matplotlib()
