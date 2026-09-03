import sys
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

def test_model(repo_id):
    print(f"Testing {repo_id}...")
    try:
        model = AutoModelForImageClassification.from_pretrained(repo_id)
        print("  Labels:", model.config.id2label)
    except Exception as e:
        print(f"  Failed: {e}")

models = [
    "umm-maybe/AI-image-detector",
    "buildborderless/CommunityForensics-DeepfakeDet-ViT",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2"
]

for m in models:
    test_model(m)
