
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export NCCL_DEBUG=TRACE
export NCCL_P2P_DISABLE=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1


df -h /tmp

NUM_SHARDS=6
GPUS_PER_SHARD=1

STUDENTS=(
    "meta-llama/Llama-3.2-1B-Instruct"
)

echo "Student models to evaluate: ${STUDENTS[@]}"
echo ""

echo "Discovering datasets in outputs/traces/..."



DATASETS=("gpqa" "math" "connections")

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "ERROR: No datasets found in outputs/traces/"
    exit 1
fi

echo ""
echo "Processing ${#DATASETS[@]} datasets: ${DATASETS[@]}"
echo ""

for dataset_name in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Processing dataset: $dataset_name"
    echo "=========================================="
    
    for teacher_file in outputs/traces/${dataset_name}/traces_*.pkl; do
        if [[ "$teacher_file" == *"/archived/"* ]]; then
            continue
        fi
        
        teacher_basename=$(basename "$teacher_file")
        echo "Teacher trace file: $teacher_basename"
        
        teacher_name="${teacher_basename#traces_}"
        teacher_name="${teacher_name%.pkl}"
        
        teacher_formatted=$(echo "$teacher_name" | sed 's/_/\//')
        
        echo "  Formatted teacher model: $teacher_formatted"
        
        trace_timestamp=$(stat -c %Y "$teacher_file" 2>/dev/null || stat -f %m "$teacher_file" 2>/dev/null)
        trace_date=$(date -d @${trace_timestamp} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r ${trace_timestamp} '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
        echo "  Trace file timestamp: $trace_date"
        
        for student in "${STUDENTS[@]}"; do
            echo ""
            echo "  ------------------------------------------"
            echo "  Student model: $student"
            echo "  ------------------------------------------"
            
            student_formatted_underscore=$(echo "$student" | tr '/' '_')
            output_file="outputs/pu/${dataset_name}/${teacher_name}/pu_${student_formatted_underscore}.pkl"
            
            skip_processing=false
            if [ -f "$output_file" ]; then
                pu_timestamp=$(stat -c %Y "$output_file" 2>/dev/null || stat -f %m "$output_file" 2>/dev/null)
                pu_date=$(date -d @${pu_timestamp} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r ${pu_timestamp} '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
                echo "  PU output exists: $pu_date"
                
                if [ "$pu_timestamp" -gt "$trace_timestamp" ]; then
                    echo "  ✓ PU output is newer than trace file, skipping..."
                    skip_processing=true
                else
                    echo "  ⚠ Trace file is newer! PU output is stale."
                    echo "  Moving stale PU output to archived..."
                    
                    timestamp=$(date '+%Y%m%d_%H%M%S')
                    archive_dir="outputs/pu/archived/${dataset_name}/${teacher_name}"
                    mkdir -p "$archive_dir"
                    archive_file="${archive_dir}/pu_${student_formatted_underscore}.${timestamp}.stale.bak"
                    mv "$output_file" "$archive_file"
                    echo "  Archived to: $archive_file"
                fi
            else
                echo "  No existing PU output found"
            fi
            
            if [ "$skip_processing" = true ]; then
                continue
            fi
            if [ "$skip_processing" = true ]; then
                continue
            fi
            
            echo "  Running $NUM_SHARDS shards in parallel with $GPUS_PER_SHARD GPUs each..."
            
            declare -a PIDS
            for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
                gpu_start=$((shard_id * GPUS_PER_SHARD))
                gpu_end=$((gpu_start + GPUS_PER_SHARD - 1))
                gpu_list=$(seq -s, $gpu_start $gpu_end)
                
                echo "    Starting shard $shard_id on GPUs $gpu_list..."
                CUDA_VISIBLE_DEVICES=$gpu_list python3 -u -m src.scripts.pedagogical_utility \
                    --student_model "$student" \
                    --teacher_model "$teacher_formatted" \
                    --dataset_name "$dataset_name" \
                    --output_dir outputs/pu \
                    --log_path outputs/pu \
                    --num_shards $NUM_SHARDS \
                    --shard_id $shard_id &
                
                PIDS[$shard_id]=$!
            done
            
            echo "  Waiting for all shards to complete..."
            failed=0
            for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
                wait ${PIDS[$shard_id]}
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo "    ERROR: Shard $shard_id failed with exit code $exit_code"
                    failed=1
                else
                    echo "    Shard $shard_id completed successfully"
                fi
            done
            
            if [ $failed -eq 1 ]; then
                echo "  ERROR: One or more shards failed. Skipping merge."
                echo "  Shard files preserved for debugging in outputs/pu/${dataset_name}/${teacher_name}/shards/"
                continue
            fi
            
            echo "  Merging shards..."
            teacher_formatted_underscore=$(echo "$teacher_formatted" | tr '/' '_')
            python3 -m src.scripts.merge_pu_shards \
                --input_pattern "outputs/pu/${dataset_name}/${teacher_formatted_underscore}/shards/pu_${student_formatted_underscore}_shard*of${NUM_SHARDS}.pkl" \
                --output_file "outputs/pu/${dataset_name}/${teacher_formatted_underscore}/pu_${student_formatted_underscore}.pkl"
            
            merge_exit=$?
            if [ $merge_exit -ne 0 ]; then
                echo "  ERROR: Failed to merge shards (exit code $merge_exit)"
                echo "  Shard files preserved for debugging"
                continue
            fi
            
            echo "  ✓ Successfully completed PU analysis for student: $student"
            
        done
        
        echo "  Completed processing for teacher: $teacher_formatted"
        echo ""
        rm -rf ~/.cache/vllm
        rm -rf /tmp/torchinductor_$(whoami)

    done
    
    echo "Finished processing dataset: $dataset_name"
    echo ""
done

echo "All datasets and teachers processed!"