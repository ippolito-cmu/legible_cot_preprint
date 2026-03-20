models=(
  openai/gpt-oss-120b
  openai/gpt-oss-20b
)

datasets=(
  math
  gpqa
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