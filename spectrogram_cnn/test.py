from cnn import classification_cnn
from glob import glob
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from skimage import io
import h5py
import keras

# load data
fnames = [f for f in sorted(glob('images/*.jpg'))]
X = np.zeros((len(fnames), 128, 128, 1))
for i in range(len(fnames)):
    im = io.imread('images/'+fnames)
    im -= np.mean(im)
    X[i,:,:,:] = im
Y = np.zeros((len(fnames), 10))

# construct model and load weights
model = classification_cnn((128, 128, 1), 10, 32, 5)
model.load_weights('model.hdf5')

# test model
checkpoint = ModelCheckpoint('model.hdf5', monitor='val_loss', save_best_only=True)
Y_predict = model.predict(x=X, y=Y, batch_size=num_training, epochs=50, verbose=1, validation_split=0.2, shuffle=True)

