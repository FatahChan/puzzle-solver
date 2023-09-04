import requests
import os
def getImage(url, path):
    
    print("Downloading images from " + url)
    r = requests.get(url)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            f.write(r.content)
        print("Download complete!")


url = "https://picsum.photos/v2/list?page=1&limit=100"
imagesList = requests.get(url)
os.makedirs('images', exist_ok=True)
for image in imagesList.json():
    getImage(image['download_url'], 'images/' + image['id'] + '.jpg')


