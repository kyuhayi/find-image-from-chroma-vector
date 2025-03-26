import kagglehub
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel, ViTImageProcessor


path = kagglehub.dataset_download("trolukovich/food11-image-dataset")

print("Path to dataset files:", path)

image_open = Image.open(
    "C:\\Users\\kyuha\\.cache\\kagglehub\\datasets\\trolukovich\\food11-image-dataset\\versions\\1\\training\\Rice\\19.jpg")



feature_extractor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
pretrained = ViTModel.from_pretrained('facebook/dino-vits16')
model = pretrained.to("cuda")

img_tensor = feature_extractor(images=image_open, return_tensors="pt").to("cuda")
outputs = model(**img_tensor)

embedding = outputs.pooler_output.detach().cpu().numpy().squeeze()

print(embedding)
print(embedding.shape)

import chromadb
from glob import glob

client = chromadb.Client()

collection = client.create_collection("foods")

img_list = sorted(glob(
    "C:\\Users\\kyuha\\.cache\\kagglehub\\datasets\\trolukovich\\food11-image-dataset\\versions\\1\\training\\*\\*.jpg"))

print(len(img_list))



from tqdm import tqdm

embeddings = []
metadatas = []
ids = []

for i, img_path in enumerate(tqdm(img_list)):
    img = Image.open(img_path)
    cls = img_path.split("\\")[1]

    img_tensor = feature_extractor(images=img, return_tensors="pt").to("cuda")
    outputs = model(**img_tensor)
    embedding = outputs.pooler_output.detach().cpu().numpy().squeeze()
    embeddings.append(embedding)
    metadatas.append({
        "uri": img_path,
        "name": cls
    })
    ids.append(str(i))

print("All images have been loaded into Chroma")


import requests

test_img = Image.open(requests.get("https://i.ibb.co/7Ksr5mw/yNp6qTS.png", stream=True).raw).convert("RGB")


test_img_tensor = feature_extractor(images=test_img, return_tensors="pt").to("cuda")
test_outputs = model(**test_img_tensor)

test_embedding = test_outputs.pooler_output.detach().cpu().numpy().squeeze()

query_result = collection.query(query_embeddings=[test_embedding], n_results=3)

print(query_result)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 10))
for i, metadata in enumerate(query_result["metadatas"][0]):
    distance = query_result["distances"][0][i]

    axes[i].imshow(Image.open(metadata["uri"]))
    axes[i].set_title(f"{metadata['name']}: {distance:.2f}")
    axes[i].axis("off")



