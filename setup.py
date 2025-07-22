"""
Setup Script for Dataset Testing

This script sets up the environment for testing datasets with the pseudo-Boolean
polynomial dimensionality reduction approach.
"""

import subprocess
import sys
import os

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = ['./data', './data/images']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def run_dataset_loader():
    """Run the dataset loader."""
    print("Loading and transforming datasets...")
    try:
        subprocess.check_call([sys.executable, "dataset_loader.py"])
        print("✓ Datasets loaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error loading datasets: {e}")
        return False

def run_example():
    """Run the example script."""
    print("Running example analysis...")
    try:
        subprocess.check_call([sys.executable, "example_usage.py"])
        print("✓ Example analysis completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running example: {e}")
        return False

def main():
    """Main setup function."""
    print("="*60)
    print("SETUP: Pseudo-Boolean Polynomial Dataset Testing")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists("dataset_loader.py"):
        print("Error: Please run this script from the clustering directory")
        print("Current directory:", os.getcwd())
        return False
    
    # Step 1: Install requirements
    # if not install_requirements():
    #     return False
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Load datasets
    if not run_dataset_loader():
        return False
    
    # Step 4: Run example (optional)
    if len(sys.argv) > 1 and sys.argv[1] == '--run-example':
        run_example()
    else:
        print("\nWould you like to run the example analysis? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes']:
                run_example()
        except KeyboardInterrupt:
            print("\nSetup completed without running example.")
    
    print("\n" + "="*60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Next steps:")
    print("1. Run 'python test_datasets.py' to test all datasets")
    print("2. Run 'python example_usage.py' to see examples")
    print("3. Check ./data/ directory for results")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 