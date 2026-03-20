
set -euo pipefail


mkdir -p logs/efficiency

export TOKENIZERS_PARALLELISM=false

NUM_SHARDS=4

echo "Starting efficiency analysis for all traces (basic metrics)"
echo "Analyzing: token length, sentence count"
echo "Sharding: $NUM_SHARDS shards (CPU-based)"
echo "=========================================="

for dataset in math gpqa connections; do
    echo ""
    echo "Processing dataset: $dataset"
    echo "----------------------------------------"
    
    traces_dir="outputs/traces/$dataset"
    
    if [ ! -d "$traces_dir" ]; then
        echo "  Warning: Directory $traces_dir not found, skipping..."
        continue
    fi
    
    trace_count=$(find "$traces_dir" -maxdepth 1 -name "traces_*.pkl" -type f | wc -l)
    
    if [ "$trace_count" -eq 0 ]; then
        echo "  No trace files found in $traces_dir, skipping..."
        continue
    fi
    
    echo "  Found $trace_count trace files"
    
    for trace_file in "$traces_dir"/traces_*.pkl; do
        trace_basename=$(basename "$trace_file")
        model_name="${trace_basename#traces_}"
        model_name="${model_name%.pkl}"
        
        model_formatted=$(echo "$model_name" | sed 's/_/\//')
        
        echo ""
        echo "  Model: $model_formatted"
        echo "  ----------------------------------------"
        
        declare -a PIDS
        for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
            echo "    Starting shard $shard_id..."
            python3 -u -m src.scripts.analyze_efficiency \
                --model_name "$model_formatted" \
                --dataset_name "$dataset" \
                --output_dir outputs/efficiency_analysis \
                --log_path logs/efficiency \
                --num_shards $NUM_SHARDS \
                --shard_id $shard_id &
            
            PIDS[$shard_id]=$!
        done
        
        echo "    Waiting for all shards to complete..."
        failed=0
        for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
            wait ${PIDS[$shard_id]}
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "      ERROR: Shard $shard_id failed with exit code $exit_code"
                failed=1
            else
                echo "      Shard $shard_id completed successfully"
            fi
        done
        
        if [ $failed -eq 1 ]; then
            echo "    ERROR: One or more shards failed. Skipping merge."
            continue
        fi
        
        echo "    Merging shards..."
        model_underscore=$(echo "$model_formatted" | tr '/' '_')
        
        python3 -m src.scripts.merge_efficiency_shards \
            --input_pattern "outputs/efficiency_analysis/${dataset}/${model_underscore}/efficiency_analysis_shard*of${NUM_SHARDS}.pkl" \
            --output_file "outputs/efficiency_analysis/${dataset}/${model_underscore}/efficiency_analysis.pkl"
        
        echo "    ✓ Completed analysis for $model_formatted"
    done
    
    echo ""
    echo "Completed dataset: $dataset"
    echo "=========================================="
done

echo ""
echo "All efficiency analyses complete!"
echo "Results in: outputs/efficiency_analysis/"
