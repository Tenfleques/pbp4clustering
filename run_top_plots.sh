#!/usr/bin/env bash
set -euo pipefail
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=max/rows=na/tagg=na/bodyacc=na --agg max --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=sum/rows=na/tagg=na/bodyacc=na --agg sum --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=mean/rows=na/tagg=na/bodyacc=na --agg mean --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=entropy/rows=na/tagg=na/bodyacc=na --agg entropy --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=rms/rows=na/tagg=na/bodyacc=na --agg rms --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=range/rows=na/tagg=na/bodyacc=na --agg range --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=trimmed_mean/rows=na/tagg=na/bodyacc=na --agg trimmed_mean --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=robust_adaptive/rows=na/tagg=na/bodyacc=na --agg robust_adaptive --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=std/rows=na/tagg=na/bodyacc=na --agg std --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=median/rows=na/tagg=na/bodyacc=na --agg median --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=adaptive/rows=na/tagg=na/bodyacc=na --agg adaptive --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=iqr/rows=na/tagg=na/bodyacc=na --agg iqr --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=six_axis/pbp=gini/rows=na/tagg=na/bodyacc=na --agg gini --include-body-acc
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=axis_feature_columns/pbp=max/rows=na/tagg=max/bodyacc=false --agg max --axis-feature-columns --time-agg max
python run_har.py --data-dir ./data/har --results-dir results/plots/fmt=axis_feature_format/pbp=adaptive/rows=na/tagg=na/bodyacc=true --agg adaptive --axis-feature-format --include-body-acc
