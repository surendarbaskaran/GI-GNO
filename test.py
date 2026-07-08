from huggingface_hub import hf_hub_download
# repo_id = "srndrbas/blendednet_harvard_dataverse"
filename = "Dataset.zip"
token = "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" # Your private token

folder_path="data"
repo_id="srndrbas/blendednet_harvard_dataverse"#"srndrbas/preprocessed"
repo_type="dataset"

# # Download the file

local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    token=token,
    local_dir="." # Optional: downloads to current folder instead of cache
)

print(f"File downloaded to: {local_path}")
# !unzip Dataset.zip "train/*"
# !rm train/vtk/case_5331.vtk  train/vtk/case_7474.vtk train/vtk/case_7709.vtk Dataset.zip
