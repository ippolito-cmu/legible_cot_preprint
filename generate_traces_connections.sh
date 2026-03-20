df -h /tmp

models=(
  nvidia/OpenReasoning-Nemotron-32B
  nvidia/Llama-3.1-Nemotron-Nano-8B-v1
)

datasets=(
  connections
)


for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "Running model: $model on dataset: $dataset"
    python3 -u -m src.scripts.generate_traces \
      --model_name "$model" \
      --dataset_name "$dataset" \
      --verbose
  done
done