from PIL import Image
# Import whatever extra thing you like!
import os

class ImageDirectory:
    """ A dataset defined on a directory full of images, like the provided cifar10/train. """

    def __init__(self, directory_path, transform=None):
        """ 
        Initialize the dataset over a given directory, with (optionally) a transformation
        to apply to the images.

        Usually, images are loaded as PIL images by default, and the transformation is
        composed from stuff in torchvision.transforms. Many of the transformations
        in torchvision.transforms work both on PIL images and torch tensors.
        """

        self._directory_path = directory_path
        self._transform = transform
        self._image_names = sorted(os.listdir(directory_path))
        self._length = len(self._image_names)
    
    def __getitem__(self, i):
        """ Return the i'th image from the dataset. Remember to apply the transformation! """

        img_file = os.path.join(self._directory_path, self._image_names[i])
        image = Image.open(img_file)

        if self._transform:
            image = self._transform(image)

        return image

    def __len__(self):
        """ Return the size of the dataset (number of images) """

        return self._length