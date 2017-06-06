import skimage
from skimage import data  # most functions are in subpackages

camera = data.camera()
camera.dtype

camera.shape

from skimage import restoration
filtered_camera = restoration.denoise_bilateral(camera)
type(filtered_camera)