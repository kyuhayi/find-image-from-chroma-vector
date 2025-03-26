import requests
from PIL import Image
from io import BytesIO




test_img = Image.open(requests.get("https://i.ibb.co/7Ksr5mw/yNp6qTS.png", stream=True).raw).convert("RGB")


# url = "https://i.ibb.co/7Ksr5mw/yNp6qTS.png"
#
# # Fetch the image stream
# response = requests.get(url)
# img_stream = BytesIO(response.content)  # Load into memory as bytes
#
# # Open and convert to RGB
# img = Image.open(img_stream)
# test_img = img.convert("RGB")


test_img.show()