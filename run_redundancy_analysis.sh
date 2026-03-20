
mkdir -p outputs/logs/redundancy

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false

NUM_SHARDS=4
GPUS_PER_SHARD=1

REDUNDANCY_THRESHOLD=0.85
EMBEDDING_MODEL="all-MiniLM-L6-v2"

echo "Starting redundancy analysis for all traces"
echo "Analyzing: semantic redundancy using sentence embeddings"
echo "Embedding model: $EMBEDDING_MODEL"
echo "Similarity threshold: $REDUNDANCY_THRESHOLD"
echo "Sharding: $NUM_SHARDS shards with $GPUS_PER_SHARD GPU(s) each"
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
        
        model_underscore=$(echo "$model_formatted" | tr '/' '_')
        output_file="outputs/redundancy_analysis/${dataset}/${model_underscore}/redundancy_analysis.pkl"
        
        if [ -f "$output_file" ]; then
            echo "  Analysis for $model_formatted already exists, skipping..."
            continue
        fi
        
        echo ""
        echo "  Model: $model_formatted"
        echo "  ----------------------------------------"
        
        declare -a PIDS
        for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
            gpu_id=$shard_id
            
            echo "    Starting shard $shard_id on GPU $gpu_id..."
            CUDA_VISIBLE_DEVICES=$gpu_id python3 -u -m src.scripts.analyze_redundancy \
                --model_name "$model_formatted" \
                --dataset_name "$dataset" \
                --output_dir outputs/redundancy_analysis \
                --efficiency_dir outputs/efficiency_analysis \
                --log_path outputs/logs/redundancy \
                --redundancy_threshold $REDUNDANCY_THRESHOLD \
                --embedding_model "$EMBEDDING_MODEL" \
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
        
        python3 -m src.scripts.merge_redundancy_shards \
            --input_pattern "outputs/redundancy_analysis/${dataset}/${model_underscore}/redundancy_analysis_shard*of${NUM_SHARDS}.pkl" \
            --output_file "outputs/redundancy_analysis/${dataset}/${model_underscore}/redundancy_analysis.pkl"
        
        echo "    ✓ Completed analysis for $model_formatted"
    done
    
    echo ""
    echo "Completed dataset: $dataset"
    echo "=========================================="
done

echo ""
echo "All redundancy analyses complete!"
echo "Results in: outputs/redundancy_analysis/"
