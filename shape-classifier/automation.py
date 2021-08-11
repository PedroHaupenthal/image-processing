# Constants
MIN_IMAGE_NUMBER = 1
MAX_IMAGE_NUMBER = 300
BASE_URL = 'https://raw.githubusercontent.com/PedroHaupenthal/Image-Processing/master/shape-classifier/frutas_dataset/'
FILE_FORMAT = '.bmp'

for i in range(MIN_IMAGE_NUMBER, MAX_IMAGE_NUMBER + 1):
  URL = f'{BASE_URL}{i}{FILE_FORMAT}'
  !wget "$URL"
