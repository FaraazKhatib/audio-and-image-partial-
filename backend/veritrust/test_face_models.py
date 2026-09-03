import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

def test_model(repo, image_paths):
    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForImageClassification.from_pretrained(repo)
    model.eval()
    
    print(f'\n--- {repo} ---')
    print('Labels:', model.config.id2label)
    
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        inputs = processor(images=img, return_tensors='pt')
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        print(f'{path.split("\\\\")[-1]}:')
        for i, p in enumerate(probs):
            print(f'  {model.config.id2label[i]}: {p.item():.4f}')

test_images = [
    r'C:\Users\Faraaz Khatib\.gemini\antigravity-ide\brain\f15500e4-35b8-4f7a-8923-b2e7cb6e48d7\.user_uploaded\media_1787589065567.png',
    r'C:\Users\Faraaz Khatib\.gemini\antigravity-ide\brain\f15500e4-35b8-4f7a-8923-b2e7cb6e48d7\.user_uploaded\media_1787590446642.png'
]

repos = [
    'prithivMLmods/Deep-Fake-Detector-v2-Model',
    'dima806/deepfake_vs_real_image_detection',
    'prithivMLmods/deepfake-detector-model-v1'
]

for repo in repos:
    test_model(repo, test_images)
