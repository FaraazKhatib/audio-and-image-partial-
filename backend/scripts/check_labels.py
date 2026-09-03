import requests

models = [
    'umm-maybe/AI-image-detector',
    'buildborderless/CommunityForensics-DeepfakeDet-ViT',
    'prithivMLmods/Deep-Fake-Detector-v2-Model',
    'prithivMLmods/deepfake-detector-model-v1',
    'prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2'
]

for m in models:
    url = f'https://huggingface.co/{m}/resolve/main/config.json'
    r = requests.get(url)
    if r.status_code == 200:
        config = r.json()
        print(f"{m}: {config.get('id2label')}")
    else:
        print(f"{m}: Failed ({r.status_code})")
