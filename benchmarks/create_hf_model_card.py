import sys

from huggingface_hub import RepoCard, create_repo

model_id = sys.argv[1].strip()
with open(f"./model_card_{model_id.split('/')[-1]}.md", "r") as f:
    META_STRING = f.read()

url = create_repo(model_id, exist_ok=True)

card = RepoCard(META_STRING, ignore_metadata_errors=True)
card.push_to_hub(model_id)
