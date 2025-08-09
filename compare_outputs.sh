#!/bin/bash

# Simple script to run original and refactored versions side-by-side
# Usage: ./compare_outputs.sh <runner_name> [agg_function]

RUNNER=$1
AGG=${2:-sum}

if [ -z "$RUNNER" ]; then
    echo "Usage: $0 <runner_name> [agg_function]"
    echo "Example: $0 iris2 sum"
    echo ""
    echo "Available refactored runners:"
    ls run_*_refactored.py 2>/dev/null | sed 's/run_//g' | sed 's/_refactored.py//g' | sort
    exit 1
fi

ORIGINAL="run_${RUNNER}.py"
REFACTORED="run_${RUNNER}_refactored.py"

if [ ! -f "$ORIGINAL" ]; then
    echo "❌ Original file not found: $ORIGINAL"
    exit 1
fi

if [ ! -f "$REFACTORED" ]; then
    echo "❌ Refactored file not found: $REFACTORED"
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

# Run refactored
echo ""
echo "📊 Running REFACTORED: $REFACTORED"
echo "------------------------------------------"
.venv/bin/python "$REFACTORED" --agg "$AGG" --no-plot 2>&1 | tee "$REF_OUT" | grep -E "(Loaded|PBP vectors|Metrics|v_measure|adjusted_rand|silhouette|calinski|davies|linear_sep|cv_score|margin_score|boundary)"

# Compare key metrics
echo ""
echo "=========================================="
echo "📈 METRICS COMPARISON"
echo "=========================================="

echo ""
echo "Original metrics:"
grep -E "(v_measure|adjusted_rand|silhouette|calinski|davies|linear_sep|cv_score|margin_score|boundary)" "$ORIG_OUT" | tail -4

echo ""
echo "Refactored metrics:"
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