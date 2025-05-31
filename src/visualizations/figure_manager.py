#!/usr/bin/env python3
"""
Figure Manager
==============

Manages figure creation, organization, and file output for the spatial analytics project.
"""

import os
import json
from pathlib import Path
from datetime import datetime

class FigureManager:
    """Manages figure output paths and organization."""
    
    def __init__(self, base_output_dir="output/figures"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize figure index
        self.index_file = self.base_output_dir / "figure_index.json"
        self.figure_index = self._load_figure_index()
    
    def _load_figure_index(self):
        """Load existing figure index or create new one."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "created": datetime.now().isoformat(),
                "figures": {}
            }
    
    def _save_figure_index(self):
        """Save figure index to file."""
        self.figure_index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(self.figure_index, f, indent=2)
    
    def register_figure(self, category, subcategory, filename, description, 
                       figure_type="working", formats=None):
        """
        Register a new figure and return file paths.
        
        Args:
            category: Main category (e.g., "model_performance")
            subcategory: Subcategory (e.g., "validation")
            filename: Base filename without extension
            description: Figure description
            figure_type: "working", "publication", "presentation"
            formats: List of formats to generate ["png", "pdf", "eps"]
        
        Returns:
            Dictionary of format: filepath pairs
        """
        if formats is None:
            formats = ["png", "pdf"]
        
        # Create directory structure
        fig_dir = self.base_output_dir / category / subcategory
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate file paths
        file_paths = {}
        for fmt in formats:
            file_paths[fmt] = fig_dir / f"{filename}.{fmt}"
        
        # Update figure index
        figure_id = f"{category}_{subcategory}_{filename}"
        self.figure_index["figures"][figure_id] = {
            "category": category,
            "subcategory": subcategory,
            "filename": filename,
            "description": description,
            "figure_type": figure_type,
            "formats": formats,
            "paths": {fmt: str(path) for fmt, path in file_paths.items()},
            "created": datetime.now().isoformat()
        }
        
        # Save updated index
        self._save_figure_index()
        
        return file_paths
    
    def list_figures(self, category=None, figure_type=None):
        """List all registered figures, optionally filtered."""
        figures = self.figure_index["figures"]
        
        if category:
            figures = {k: v for k, v in figures.items() 
                      if v["category"] == category}
        
        if figure_type:
            figures = {k: v for k, v in figures.items() 
                      if v["figure_type"] == figure_type}
        
        return figures
    
    def get_figure_info(self, figure_id):
        """Get information about a specific figure."""
        return self.figure_index["figures"].get(figure_id)
    
    def cleanup_missing_files(self):
        """Remove entries for files that no longer exist."""
        to_remove = []
        
        for figure_id, info in self.figure_index["figures"].items():
            for fmt, path in info["paths"].items():
                if not Path(path).exists():
                    to_remove.append(figure_id)
                    break
        
        for figure_id in to_remove:
            del self.figure_index["figures"][figure_id]
        
        if to_remove:
            self._save_figure_index()
            print(f"Removed {len(to_remove)} missing figure entries")
        
        return len(to_remove)
