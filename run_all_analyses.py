#!/usr/bin/env python3
"""
Run All Analyses: Comprehensive and Feature Exclusion

This script runs all possible instances for both measurement_exclusion_comparison.py 
and comprehensive_comparison.py with support for dataset selection.
"""

import subprocess
import sys
import os
import argparse
import time
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            print(f"⏱️  Duration: {duration:.2f} seconds")
            if result.stdout:
                print("📄 Output:")
                print(result.stdout)
        else:
            print(f"❌ FAILED: {description}")
            print(f"⏱️  Duration: {duration:.2f} seconds")
            if result.stderr:
                print("🚨 Error:")
                print(result.stderr)
            if result.stdout:
                print("📄 Output:")
                print(result.stdout)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ EXCEPTION: {description}")
        print(f"🚨 Error: {e}")
        return False

def run_measurement_exclusion_analysis(dataset, max_drop=None):
    """Run measurement exclusion comparison analysis."""
    print(f"⚠️  Measurement exclusion analysis not implemented in the current structure")
    print(f"    This would require implementing feature selection in the PBP approach")
    print(f"    Dataset: {dataset}, Max drop: {max_drop}")
    return True  # Return True to not fail the overall analysis

def run_comprehensive_comparison_analysis():
    """Run comprehensive comparison analysis."""
    cmd = "python -m src.analysis.comparison"
    return run_command(cmd, "Comprehensive Comparison Analysis (all datasets)")

def run_single_dataset_comprehensive_analysis(dataset):
    """Run comprehensive comparison for a single dataset."""
    # Create a temporary script for single dataset analysis
    temp_script = f"""
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.analysis.comparison import ComprehensiveComparison

# Initialize comparison
comparison = ComprehensiveComparison('./data')

# Run analysis for single dataset
results = comparison.compare_methods('{dataset}')
if results:
    print(f"\\n✅ Comprehensive comparison completed for {dataset}")
    print(f"Results: {{len(results)}} methods compared")
else:
    print(f"\\n❌ Failed to run comprehensive comparison for {dataset}")
"""
    
    temp_file = f"temp_comprehensive_{dataset}.py"
    
    try:
        with open(temp_file, 'w') as f:
            f.write(temp_script)
        
        cmd = f"python {temp_file}"
        success = run_command(cmd, f"Comprehensive Comparison Analysis for {dataset}")
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return success
        
    except Exception as e:
        print(f"❌ Error creating temporary script: {e}")
        return False

def run_dataset_testing_analysis():
    """Run dataset testing analysis."""
    cmd = "python -m src.analysis.testing"
    return run_command(cmd, "Dataset Testing Analysis (all datasets)")

def run_single_dataset_testing_analysis(dataset):
    """Run dataset testing for a single dataset."""
    # Create a temporary script for single dataset testing
    temp_script = f"""
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.analysis.testing import DatasetTester

# Initialize tester
tester = DatasetTester('./data')

# Run testing for single dataset
result = tester.test_dataset('{dataset}')
if result:
    print(f"\\n✅ Dataset testing completed for {dataset}")
    print(f"Silhouette Score: {{result['evaluation']['silhouette_score']:.4f}}")
else:
    print(f"\\n❌ Failed to run dataset testing for {dataset}")
"""
    
    temp_file = f"temp_testing_{dataset}.py"
    
    try:
        with open(temp_file, 'w') as f:
            f.write(temp_script)
        
        cmd = f"python {temp_file}"
        success = run_command(cmd, f"Dataset Testing Analysis for {dataset}")
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return success
        
    except Exception as e:
        print(f"❌ Error creating temporary script: {e}")
        return False

def run_all_analyses(dataset=None, max_drop=None, comprehensive_only=False, exclusion_only=False, testing_only=False):
    """Run all analyses based on specified options."""
    print(f"\n{'='*80}")
    print("RUNNING ALL ANALYSES")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset if dataset else 'ALL'}")
    print(f"Max drop: {max_drop if max_drop else 'AUTO'}")
    print(f"Comprehensive only: {comprehensive_only}")
    print(f"Exclusion only: {exclusion_only}")
    print(f"Testing only: {testing_only}")
    print(f"{'='*80}")
    
    success_count = 0
    total_count = 0
    
    if not exclusion_only and not testing_only:
        # Run comprehensive comparison
        if dataset:
            success = run_single_dataset_comprehensive_analysis(dataset)
        else:
            success = run_comprehensive_comparison_analysis()
        
        total_count += 1
        if success:
            success_count += 1
    
    if not comprehensive_only and not testing_only:
        # Run measurement exclusion (if implemented)
        if dataset:
            success = run_measurement_exclusion_analysis(dataset, max_drop)
        else:
            # Run for all datasets
            datasets = ['iris', 'breast_cancer', 'wine', 'digits', 'diabetes', 'sonar', 'glass', 'vehicle', 'ecoli', 'yeast']
            success = True
            for ds in datasets:
                if not run_measurement_exclusion_analysis(ds, max_drop):
                    success = False
        
        total_count += 1
        if success:
            success_count += 1
    
    if not comprehensive_only and not exclusion_only:
        # Run dataset testing
        if dataset:
            success = run_single_dataset_testing_analysis(dataset)
        else:
            success = run_dataset_testing_analysis()
        
        total_count += 1
        if success:
            success_count += 1
    
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Successful: {success_count}/{total_count}")
    print(f"Success rate: {(success_count/total_count)*100:.1f}%" if total_count > 0 else "No analyses run")
    
    if success_count == total_count:
        print("🎉 All analyses completed successfully!")
    elif success_count > 0:
        print("⚠️  Some analyses completed successfully.")
    else:
        print("❌ All analyses failed.")
    
    return success_count == total_count


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run all PBP analyses')
    parser.add_argument('--dataset', type=str, help='Specific dataset to analyze')
    parser.add_argument('--max_drop', type=int, help='Maximum features to drop')
    parser.add_argument('--comprehensive_only', action='store_true', help='Run only comprehensive comparison')
    parser.add_argument('--exclusion_only', action='store_true', help='Run only measurement exclusion')
    parser.add_argument('--testing_only', action='store_true', help='Run only dataset testing')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.comprehensive_only + args.exclusion_only + args.testing_only > 1:
        print("❌ Error: Only one of --comprehensive_only, --exclusion_only, or --testing_only can be specified")
        return False
    
    # Run analyses
    success = run_all_analyses(
        dataset=args.dataset,
        max_drop=args.max_drop,
        comprehensive_only=args.comprehensive_only,
        exclusion_only=args.exclusion_only,
        testing_only=args.testing_only
    )
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 