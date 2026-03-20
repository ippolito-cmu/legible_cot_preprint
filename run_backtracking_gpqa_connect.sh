

run_analysis() {
    dataset=$1
    model=$2
    smoke_test=$3

    echo "Running backtracking analysis for $dataset/$model (smoke_test: $smoke_test)"

    args=(--dataset "$dataset" --model "$model")
    if [[ "$smoke_test" == "true" || "$smoke_test" == "1" ]]; then
        args+=(--smoke-test)
    fi
    args+=(--max-concurrent 5 --delay 5.0 --batch-size 5 --output-dir outputs/backtracking_analysis)
    export PYTHONUNBUFFERED=1
    python3 -m -u src.scripts.analyze_backtracking "${args[@]}"

    echo "Completed $dataset/$model"
}

echo "Starting backtracking analysis on all traces..."

datasets=("gpqa" "connections")

MAX_JOBS=${MAX_JOBS:-5}
echo "Using MAX_JOBS=$MAX_JOBS"

running=0
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    for file in outputs/traces/$dataset/traces_*.pkl; do
        if [[ $file == *"archived"* ]]; then
            continue
        fi
        
        model=$(basename "$file" .pkl | sed 's/traces_//')
        
        echo "Found model: $model for $dataset"

        srun --exclusive -n1 bash -c "PYTHONUNBUFFERED=1 python3 -u -m src.scripts.analyze_backtracking --dataset '$dataset' --model '$model' --output-dir outputs/backtracking_analysis" &
        running=$((running+1))

        if (( running >= MAX_JOBS )); then
            wait -n
            running=$((running-1))
        fi
    done
done

wait

echo "All backtracking analyses completed."

echo "API Usage Summary:"
echo "Total API calls: [parse from logs]"
echo "Estimated cost: [calculate based on usage]"