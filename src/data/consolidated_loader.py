#!/usr/bin/env python3
"""
Consolidated Dataset Loader for Pseudo-Boolean Polynomial Dimensionality Reduction

This module provides a unified interface for loading all datasets by reading from
a configuration file and using the appropriate specialized loaders based on inheritance.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base_loader import BaseDatasetLoader
from .standard_loader import StandardDatasetLoader
from .uci_loader import UCIDatasetLoader
from .large_loader import LargeDatasetLoader
from .advanced_aids_loader import AdvancedAIDSLoader
from .advanced_genetic_loader import AdvancedGeneticLoader
from .advanced_medical_loader import AdvancedMedicalLoader
from .advanced_nci_loader import AdvancedNCILoader
from .business_loader import BusinessDatasetLoader

class ConsolidatedDatasetLoader(BaseDatasetLoader):
    """
    Main consolidated loader that manages all dataset types.
    
    This class reads dataset names from a configuration file and uses the appropriate
    specialized loader for each dataset category. It provides a unified interface
    for loading all datasets in the workflow.
    """
    
    def __init__(self, data_dir='./data', config_file='src/data/dataset_config.txt'):
        super().__init__(data_dir)
        self.config_file = Path(config_file)
        
        # Initialize specialized loaders
        self.loaders = {
            'standard': StandardDatasetLoader(data_dir),
            'uci': UCIDatasetLoader(data_dir),
            'large': LargeDatasetLoader(data_dir),
            'advanced_aids': AdvancedAIDSLoader(data_dir),
            'advanced_genetic': AdvancedGeneticLoader(data_dir),
            'advanced_medical': AdvancedMedicalLoader(data_dir),
            'advanced_nci': AdvancedNCILoader(data_dir),
            'business': BusinessDatasetLoader(data_dir)
        }
        
        # Load dataset configuration
        self.dataset_config = self._load_dataset_config()
    
    def _load_dataset_config(self) -> Dict[str, List[str]]:
        """Load dataset configuration from file."""
        config = {}
        
        if not self.config_file.exists():
            print(f"Warning: Configuration file {self.config_file} not found")
            return config
        
        try:
            with open(self.config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if ':' in line:
                            category, dataset_name = line.split(':', 1)
                            category = category.strip()
                            dataset_name = dataset_name.strip()
                            
                            if category not in config:
                                config[category] = []
                            config[category].append(dataset_name)
            
            print(f"✓ Loaded dataset configuration: {sum(len(datasets) for datasets in config.values())} datasets across {len(config)} categories")
            
        except Exception as e:
            print(f"Error loading dataset configuration: {e}")
        
        return config
    
    def get_available_datasets(self) -> Dict[str, List[str]]:
        """Get all available datasets by category."""
        return self.dataset_config.copy()
    
    def get_loader_for_dataset(self, dataset_name: str) -> Optional[BaseDatasetLoader]:
        """Get the appropriate loader for a specific dataset."""
        for category, datasets in self.dataset_config.items():
            if dataset_name in datasets:
                if category in self.loaders:
                    return self.loaders[category]
                else:
                    print(f"Warning: No loader found for category '{category}'")
                    return None
        
        print(f"Warning: Dataset '{dataset_name}' not found in configuration")
        return None
    
    def load_dataset(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific dataset using the appropriate specialized loader.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            dict: Dictionary containing X, y, metadata, or None if loading failed
        """
        loader = self.get_loader_for_dataset(dataset_name)
        if loader is None:
            return None
        
        try:
            return loader.load_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return None
    
    def load_datasets_by_category(self, category: str) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Load all datasets in a specific category.
        
        Args:
            category: Category name (e.g., 'standard', 'uci', 'advanced_medical')
            
        Returns:
            dict: Dictionary mapping dataset names to loaded datasets
        """
        if category not in self.dataset_config:
            print(f"Warning: Category '{category}' not found in configuration")
            return {}
        
        if category not in self.loaders:
            print(f"Warning: No loader found for category '{category}'")
            return {}
        
        results = {}
        loader = self.loaders[category]
        
        for dataset_name in self.dataset_config[category]:
            try:
                results[dataset_name] = loader.load_dataset(dataset_name)
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                results[dataset_name] = None
        
        return results
    
    def load_all_datasets(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Load all datasets from all categories.
        
        Returns:
            dict: Dictionary mapping dataset names to loaded datasets
        """
        all_results = {}
        
        for category in self.dataset_config.keys():
            print(f"\n=== Loading {category} datasets ===")
            category_results = self.load_datasets_by_category(category)
            all_results.update(category_results)
            
            # Print summary for this category
            successful = sum(1 for result in category_results.values() if result is not None)
            total = len(category_results)
            print(f"✓ {category}: {successful}/{total} datasets loaded successfully")
        
        print(f"\n=== Total Summary ===")
        successful_total = sum(1 for result in all_results.values() if result is not None)
        total_datasets = len(all_results)
        print(f"✓ Overall: {successful_total}/{total_datasets} datasets loaded successfully")
        
        return all_results
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific dataset."""
        # First try to load the dataset
        dataset = self.load_dataset(dataset_name)
        if dataset is not None:
            return {
                'name': dataset_name,
                'shape': dataset['X'].shape,
                'n_classes': len(np.unique(dataset['y'])),
                'data_type': dataset.get('metadata', {}).get('data_type', 'unknown'),
                'description': dataset.get('metadata', {}).get('description', ''),
                'source': dataset.get('metadata', {}).get('source', 'unknown'),
                'reshaping_strategy': dataset.get('metadata', {}).get('reshaping_strategy', 'unknown')
            }
        
        # If loading failed, try to get info from saved dataset
        loader = self.get_loader_for_dataset(dataset_name)
        if loader is not None:
            return loader.get_dataset_info(dataset_name)
        
        return None
    
    def get_all_dataset_info(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get information about all datasets."""
        info = {}
        for category, datasets in self.dataset_config.items():
            for dataset_name in datasets:
                info[dataset_name] = self.get_dataset_info(dataset_name)
        return info
    
    def save_dataset_summary(self, output_file: str = 'dataset_summary.json'):
        """Save a summary of all datasets to a JSON file."""
        summary = {
            'total_datasets': sum(len(datasets) for datasets in self.dataset_config.values()),
            'categories': list(self.dataset_config.keys()),
            'datasets_by_category': self.dataset_config,
            'dataset_info': self.get_all_dataset_info()
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Dataset summary saved to {output_path}")
        return summary
    
    def validate_dataset_config(self) -> Dict[str, List[str]]:
        """Validate the dataset configuration and report issues."""
        issues = {
            'missing_loaders': [],
            'unknown_datasets': [],
            'duplicate_datasets': []
        }
        
        # Check for missing loaders
        for category in self.dataset_config.keys():
            if category not in self.loaders:
                issues['missing_loaders'].append(category)
        
        # Check for duplicate datasets
        all_datasets = []
        for datasets in self.dataset_config.values():
            all_datasets.extend(datasets)
        
        seen = set()
        for dataset in all_datasets:
            if dataset in seen:
                issues['duplicate_datasets'].append(dataset)
            seen.add(dataset)
        
        # Print validation results
        if any(issues.values()):
            print("Dataset configuration validation issues:")
            for issue_type, items in issues.items():
                if items:
                    print(f"  {issue_type}: {items}")
        else:
            print("✓ Dataset configuration validation passed")
        
        return issues
    
    def reload_config(self):
        """Reload the dataset configuration from file."""
        self.dataset_config = self._load_dataset_config()
        print("✓ Dataset configuration reloaded")
    
    def add_dataset(self, category: str, dataset_name: str):
        """Add a dataset to the configuration."""
        if category not in self.dataset_config:
            self.dataset_config[category] = []
        
        if dataset_name not in self.dataset_config[category]:
            self.dataset_config[category].append(dataset_name)
            print(f"✓ Added {dataset_name} to {category} category")
        else:
            print(f"Dataset {dataset_name} already exists in {category} category")
    
    def remove_dataset(self, category: str, dataset_name: str):
        """Remove a dataset from the configuration."""
        if category in self.dataset_config and dataset_name in self.dataset_config[category]:
            self.dataset_config[category].remove(dataset_name)
            print(f"✓ Removed {dataset_name} from {category} category")
        else:
            print(f"Dataset {dataset_name} not found in {category} category")
    
    def save_config(self, output_file: str = None):
        """Save the current configuration to file."""
        if output_file is None:
            output_file = self.config_file
        
        try:
            with open(output_file, 'w') as f:
                f.write("# Dataset Configuration File\n")
                f.write("# This file contains all dataset names organized by category\n")
                f.write("# Format: category:dataset_name\n\n")
                
                for category, datasets in self.dataset_config.items():
                    f.write(f"# {category.title()} datasets\n")
                    for dataset in datasets:
                        f.write(f"{category}:{dataset}\n")
                    f.write("\n")
            
            print(f"✓ Configuration saved to {output_file}")
            
        except Exception as e:
            print(f"Error saving configuration: {e}")


def main():
    """Main function to demonstrate the consolidated loader."""
    print("=== Consolidated Dataset Loader Demo ===\n")
    
    # Initialize the consolidated loader
    loader = ConsolidatedDatasetLoader()
    
    # Validate configuration
    print("Validating dataset configuration...")
    issues = loader.validate_dataset_config()
    
    if not any(issues.values()):
        # Load all datasets
        print("\nLoading all datasets...")
        all_datasets = loader.load_all_datasets()
        
        # Save summary
        print("\nSaving dataset summary...")
        summary = loader.save_dataset_summary()
        
        print(f"\n=== Demo Complete ===")
        print(f"Total datasets configured: {summary['total_datasets']}")
        print(f"Categories: {', '.join(summary['categories'])}")
    else:
        print("Configuration issues found. Please fix before loading datasets.")


if __name__ == "__main__":
    main() 