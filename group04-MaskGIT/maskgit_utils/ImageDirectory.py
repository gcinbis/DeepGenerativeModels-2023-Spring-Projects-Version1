from PIL import Image
# Import whatever extra thing you like!
import os

class ImageDirectory:

    def __init__(self, directory_path, transform=None):
        self._directory_path = directory_path
        self._transform = transform
        self._image_names = [os.path.join(os.path.join(self._directory_path, folder), image) for folder in os.listdir(self._directory_path) for image in os.listdir(os.path.join(self._directory_path, folder))]
        self._length = len(self._image_names)
    
    def __getitem__(self, i):

        img_file = os.path.join(self._directory_path, self._image_names[i])
        image = Image.open(img_file)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self._transform:
            image = self._transform(image)

        return image

    def __len__(self):

        return self._length