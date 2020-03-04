import numpy as np
from PIL import Image

"""
sissgeo

def load_train_test( dataset_path, type ):
   # Lists all image names in dataset_path/type, which type being either `train` or `test`
   list_of_images = ...

   x = np.ndarray(shape = (n,32,32,3), dtype = np.uint8)  # n -> actual number of images in the directory
   y = np.ndarray(shape = (n), dtype = np.uint8)  # n -> actual number of images in the directory
   for each image_name in list_of_images:
      x[i] = imageio.imread(image_name)  # read each image in the dir
      y[i] = <extract class number from image_name>

   return x, y
"""


def load_sissgeo(dataset_path, shape):
	"""
        Load the tiny-imagenet dataset

        Parameters
        ----------
        dataset_path : str
        	path to the dataset files

        shape : tuple
        	target shape of the loaded instances

        Returns
        -------
        x_train : np.array
            training instances
        y_train : np.array
            training labels 
        x_test : np.array
            testing instances
        x_test : np.array
            testing labels
    """

    x_train, y_train = load_train_test( dataset_path, "train" )
    x_test, y_test = load_train_test( dataset_path, "test" )

	return x_train, y_train, x_test, y_test
