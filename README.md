# Image Colorization Project

# Useful resources

Relevant Paper: https://arxiv.org/pdf/1603.08511.pdf

Training Data (256 x 256): http://places2.csail.mit.edu/download.html

# How to run convertToBW.m (NOT NECESSARY for running CNN code)

1. Change the image_folder to the folder in the data directory that contains the images
you want to convert to black and white.

2. Run the matlab script. A new folder image_folder + 'bw' will be created that contains
all of the black and white images.

Use convertToBW.m to change a folder of RGB images into Black and White images.

# How to run CNN code

1. Open getdata.py and set MAX_IMAGES to the number of training images that you want
and set IMAGE_DIR to the directory which contains the images you want to use. 
Images must be 256x256 RGB. We used the 256 x 256 training data linked above for our
images.

2. Open cnn.py and change num_epochs in train_cnn() to the number of iterations you want
to train the cnn.

3. Also in cnn.py, change the value of image_num to change which image is the test
image. By default, the image_num is 0, so the test image will be the first image
which is read by the getdata function.

4. Now, run cnn.py by calling 'python cnn.py'. The code will load the image data, train
the cnn, and compute and save the test images. The actual image after the distortion caused
by going from rgb -> Lab -> rgb will be saved as 'actual.jpg' and the predicted image will
be saved as 'prediction.jpg' in the same directory as the code.

