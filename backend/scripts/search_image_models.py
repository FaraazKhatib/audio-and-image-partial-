import requests
import json

def search(query):
    url = f'https://huggingface.co/api/models?search={query}&filter=image-classification&sort=downloads&direction=-1&limit=15'
    response = requests.get(url)
    for model in response.json():
        print(f"{model['id']} (Downloads: {model.get('downloads', 0)}, Likes: {model.get('likes', 0)})")

print('--- AI Image Detection ---')
search('ai-image')
print('\n--- Deepfake Image Detection ---')
search('deepfake')
print('\n--- Real vs Fake Image ---')
search('real fake')
