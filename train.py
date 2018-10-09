import keras
from keras.datasets import cifar10

# from gan_libs.DCGAN import build_generator, build_discriminator, get_training_function
# from gan_libs.LSGAN import build_generator, build_discriminator, get_training_function
from gan_libs.SNGAN import build_generator, build_discriminator, get_training_function
#from gan_libs.WGAN_GP import build_generator, build_discriminator, get_training_function
import glob
from PIL import Image
from utils.common import set_gpu_config, predict_images
import numpy as np

set_gpu_config("0",0.8)

img_dir = 'downloaded/**/'
img_side_len = 256 
epoch = 50
num_of_data = len(glob.glob('downloaded/**/*.jpg'))-18
image_size = (img_side_len,img_side_len,3)
noise_size = (2,2,img_side_len)
batch_size = 16

fails = []
print(num_of_data)
img_data = np.zeros((num_of_data, img_side_len, img_side_len, 3), dtype='float32')
for i, im in enumerate(glob.glob(img_dir + '*.jpg')):
    try:
        img_data[i-len(fails)] = np.array(Image.open(im).resize((img_side_len, img_side_len)))
    except:
        # A few images are greyscale, which will mess up the resize. This is needed for those
        fails.append((i, im)) 

img_data = img_data.astype('float32')
img_data = (img_data/255)

#take 85% of the dataset for training, 15% for testing
split = int(num_of_data*.85)
#split = 100
x_train, x_test = img_data[:split], img_data[split:]

#y_train = keras.utils.to_categorical(y_train,10)
#y_test = keras.utils.to_categorical(y_test,10)

generator = build_generator(noise_size)
discriminator = build_discriminator(image_size)
d_train, g_train = get_training_function(batch_size,noise_size,image_size,generator,discriminator)
import pdb;pdb.set_trace()
for e in range(epoch):
    for s in range(split):
        real_images = x_train[np.random.permutation(split)[:batch_size]]
        d_loss, = d_train([real_images, 1])
        g_loss, = g_train([1])
        if s % 1000 == 0:
            print ("[{0}/{1}] [{2}/{3}] d_loss: {4:.4}, g_loss: {5:.4}".format(e, epoch, s, split, d_loss, g_loss))

    generator.save_weights("e{0}_generator.h5".format(e))
    predict_images("e{0}_img.png".format(e), generator,noise_size,10,image_side_len)





