import os
from lib.utility import load_image, images_data
from lib import kNN

import matplotlib.pyplot as plt
import matplotlib

train_f,train_l = images_data('data/digits/trainingDigits')
