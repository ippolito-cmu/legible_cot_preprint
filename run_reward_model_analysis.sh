NUM_SHARDS=3
GPUS_PER_SHARD=2

REWARD_MODELS=(
    "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    "allenai/Llama-3.1-8B-Instruct-RM-RB2"
)

DATASETS=("math" "gpqa" "connections")

echo "Reward model analysis"
echo "  shards: $NUM_SHARDS"
echo "  gpus per shard: $GPUS_PER_SHARD"
echo "  reward models: ${REWARD_MODELS[@]}"
echo ""

rm_safe() {
    echo "$1" | tr '/' '_'
}

for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Processing dataset: $dataset"
    echo "=========================================="

    traces_dir="outputs/traces/$dataset"
    if [ ! -d "$traces_dir" ]; then
        echo "  Warning: $traces_dir not found, skipping"
        continue
    fi

    shopt -s nullglob
    trace_files=("$traces_dir"/traces_*.pkl)
    shopt -u nullglob

    if [ ${#trace_files[@]} -eq 0 ]; then
        echo "  No trace files found in $traces_dir, skipping"
        continue
    fi

    for trace_file in "${trace_files[@]}"; do
        if [[ "$trace_file" == *"/archived/"* ]]; then
            continue
        fi

        trace_basename=$(basename "$trace_file")
        teacher_name="${trace_basename#traces_}"
        teacher_name="${teacher_name%.pkl}"
        teacher_formatted=$(echo "$teacher_name" | sed 's/_/\//')

        echo ""
        echo "  Teacher: $teacher_formatted"
        echo "  Trace:   $trace_file"

        teacher_us=$(echo "$teacher_formatted" | tr '/' '_')

        for rm in "${REWARD_MODELS[@]}"; do
            rm_us=$(rm_safe "$rm")
            echo ""
            echo "  Reward model: $rm"

            already_done="outputs/reward_model_analysis/${dataset}/${teacher_us}/reward_model_${rm_us}.pkl"
            if [[ -f "$already_done" ]]; then
                echo "    Skipping, already exists: $already_done"
                continue
            fi
            declare -a PIDS
            for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
                gpu_start=$((shard_id * GPUS_PER_SHARD))
                gpu_end=$((gpu_start + GPUS_PER_SHARD - 1))
                gpu_list=$(seq -s, $gpu_start $gpu_end)

                echo "    Starting shard $shard_id on GPUs $gpu_list..."
                CUDA_VISIBLE_DEVICES=$gpu_list python3 -u -m src.scripts.analyze_reward_models \
                    --teacher_model "$teacher_formatted" \
                    --dataset_name "$dataset" \
                    --trace_file "$trace_file" \
                    --output_dir outputs/reward_model_analysis \
                    --num_shards $NUM_SHARDS \
                    --shard_id $shard_id \
                    --batch_size 8 \
                    --max_length 8192 \
                    --input_mode question_trace \
                    --include_system_prompt \
                    --reward_model "$rm" &

                PIDS[$shard_id]=$!
            done

            echo "  Waiting for shards..."
            failed=0
            for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
                wait ${PIDS[$shard_id]}
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo "    ERROR: Shard $shard_id failed with exit code $exit_code"
                    failed=1
                else
                    echo "    Shard $shard_id completed"
                fi
            done

            if [ $failed -eq 1 ]; then
                echo "  ERROR: One or more shards failed for RM=$rm. Skipping merge."
                continue
            fi

            echo "  Merging shards..."
            python3 -m src.scripts.merge_reward_model_shards \
                --input_pattern "outputs/reward_model_analysis/${dataset}/${teacher_us}/shards/reward_model_${rm_us}_shard*of${NUM_SHARDS}.pkl" \
                --output_file "outputs/reward_model_analysis/${dataset}/${teacher_us}/reward_model_${rm_us}.pkl"

            echo "  ✓ Completed reward scoring for $teacher_formatted (RM=$rm)"
        done
    done

done

echo "All reward model analyses complete!"
