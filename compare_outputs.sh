#!/bin/bash

# Simple script to run original and refactored versions side-by-side
# Usage: ./compare_outputs.sh <runner_name> [agg_function]

RUNNER=$1
AGG=${2:-sum}

if [ -z "$RUNNER" ]; then
    echo "Usage: $0 <runner_name> [agg_function]"
    echo "Example: $0 iris2 sum"
    echo ""
    echo "Available runners:"
    ls backup_original_scripts/run_*.py 2>/dev/null | sed 's/.*run_//g' | sed 's/.py//g' | grep -v har | grep -v iris | sort
    exit 1
fi

CURRENT="run_${RUNNER}.py"
ORIGINAL="backup_original_scripts/run_${RUNNER}.py"

if [ ! -f "$ORIGINAL" ]; then
    echo "❌ Original file not found: $ORIGINAL"
    exit 1
fi

if [ ! -f "$CURRENT" ]; then
    echo "❌ Current file not found: $CURRENT"
    exit 1
fi

echo "=========================================="
echo "Comparing: $RUNNER with agg=$AGG"
echo "=========================================="

# Create temp files for outputs
ORIG_OUT=$(mktemp)
REF_OUT=$(mktemp)

# Run original
echo ""
echo "📊 Running ORIGINAL: $ORIGINAL"
echo "------------------------------------------"
.venv/bin/python "$ORIGINAL" --agg "$AGG" --no-plot 2>&1 | tee "$ORIG_OUT" | grep -E "(Loaded|PBP vectors|Metrics|v_measure|adjusted_rand|silhouette|calinski|davies|linear_sep|cv_score|margin_score|boundary)"

# Run current (refactored)
echo ""
echo "📊 Running CURRENT (refactored): $CURRENT"
echo "------------------------------------------"
.venv/bin/python "$CURRENT" --agg "$AGG" --no-plot 2>&1 | tee "$REF_OUT" | grep -E "(Loaded|PBP vectors|Metrics|v_measure|adjusted_rand|silhouette|calinski|davies|linear_sep|cv_score|margin_score|boundary)"

# Compare key metrics
echo ""
echo "=========================================="
echo "📈 METRICS COMPARISON"
echo "=========================================="

echo ""
echo "Original metrics:"
grep -E "(v_measure|adjusted_rand|silhouette|calinski|davies|linear_sep|cv_score|margin_score|boundary)" "$ORIG_OUT" | tail -4

echo ""
echo "Current (refactored) metrics:"
grep -E "(v_measure|adjusted_rand|silhouette|calinski|davies|linear_sep|cv_score|margin_score|boundary)" "$REF_OUT" | tail -4

# Check if outputs are identical for metrics lines
echo ""
echo "=========================================="
ORIG_METRICS=$(grep -E "^- (v_measure|silhouette|linear_sep)" "$ORIG_OUT" | sed 's/[[:space:]]//g')
REF_METRICS=$(grep -E "^- (v_measure|silhouette|linear_sep)" "$REF_OUT" | sed 's/[[:space:]]//g')

if [ "$ORIG_METRICS" = "$REF_METRICS" ]; then
    echo "✅ SUCCESS: Metrics match!"
else
    echo "❌ WARNING: Metrics differ!"
    echo ""
    echo "Differences:"
    diff -u <(echo "$ORIG_METRICS") <(echo "$REF_METRICS")
fi

# Cleanup
rm -f "$ORIG_OUT" "$REF_OUT"