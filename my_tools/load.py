from huggingface_hub import snapshot_download

# 指定模型名称
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# 将模型下载到挂载点目录
model_local_dir = "/root/model/qwen"

# # 指定数据集名称
# dataset_name = "OpenRLHF/preference_dataset_mixture2_and_safe_pku" 
# # 下载目录
# dataset_local_dir = "/root/dataset"

snapshot_download(repo_id=model_name, repo_type="model", local_dir=model_local_dir, ignore_patterns="*.safetensors")

# snapshot_download(repo_id=dataset_name, repo_type="dataset", local_dir=dataset_local_dir)

print(f"Model downloaded to {model_local_dir}")
# print(f"Dataset downloaded to {dataset_local_dir}")