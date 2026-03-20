
df -h /tmp

models=(
  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
  mistralai/Magistral-Small-2509
)

datasets=(
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