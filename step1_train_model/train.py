from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import History
from keras.models import Model

import numpy as np
import os
import time

from preprocess import *
from helper import *
import settings


def train_and_predict(data_path, img_rows, img_cols, n_epoch, input_no  = 3, output_no = 3,
	fn= "model", mode = 1):

	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, msks_train = load_data(data_path,"_train")
	imgs_train, msks_train = update_channels(imgs_train, msks_train, input_no, output_no,
		mode)

	print('-'*30)
	print('Loading and preprocessing test data...')
	print('-'*30)
	imgs_test, msks_test = load_data(data_path,"_test")
	imgs_test, msks_test = update_channels(imgs_test, msks_test, input_no, output_no, mode)


	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	if "-mobile" in fn:
		model		= model5_mobile_unet(False, False, img_rows, img_cols, input_no, output_no, batch_normalization="-bn=False" not in fn, batch_normalization_after="-bnafter=False" not in fn, balanced="-balanced=False" not in fn)
	else:
		model		= model5_MultiLayer(False, False, img_rows, img_cols, input_no,	output_no)
	model_fn	= os.path.join(data_path, fn+'__epoch={epoch:03d}.hdf5')
	print ("Writing model to ", model_fn)

	model_checkpoint = ModelCheckpoint(model_fn, monitor='loss', save_best_only=False)
	# saves all models when set to False

	board = TensorBoard(log_dir="logs/{}/{}__{}".format(fn, fn, time.time()))

	# initial_epoch=1
	# if initial_epoch > 1:
	# 	model_fn	= os.path.join(data_path, '{}__epoch={:03d}.hdf5'.format(fn, initial_epoch))
	# 	model.load_weights(model_fn)

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	history = History()
	history = model.fit(imgs_train, msks_train,
		batch_size=128,
		epochs=n_epoch,
		validation_data = (imgs_test, msks_test),
		verbose=1,
		callbacks=[model_checkpoint, board])

	json_fn = os.path.join(data_path, fn+'.json')
	with open(json_fn,'w') as f:
		f.write(model.to_json())


	print('-'*30)
	print('Loading saved weights...')
	print('-'*30)
	epochNo = len(history.history['loss'])
	model_fn	= os.path.join(data_path, '{}__epoch={:03d}.hdf5'.format(fn, epochNo))
	model.load_weights(model_fn)

	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	msks_pred = model.predict(imgs_test, verbose=1)
	print("Done ", epochNo, np.min(msks_pred), np.max(msks_pred))
	np.save(os.path.join(data_path, 'msks_pred.npy'), msks_pred)

	scores = model.evaluate(imgs_test, msks_test, batch_size=128,verbose = 2)
	print ("Evaluation Scores", scores)


if __name__ =="__main__":
	'''
	MODEL_FNS = (
	 	"brainWholeTumor-mobile-balanced=True-bn=False",
		"brainWholeTumor-mobile-balanced=False-bn=False",
		"brainWholeTumor-mobile-balanced=True-bn=True",
		"brainWholeTumor-mobile-balanced=False-bn=True"
		)
	MODEL_FNS = (
		"brainWholeTumor-mobile-sample_dice-balanced=True-bn=False-next",
		)
	'''
	MODEL_FNS = (
		"brainWholeTumor-mobile-out={}-balanced=True-bn=False".format(settings.OUT_CHANNEL_NO),
		)
	for model_fn in MODEL_FNS:
		train_and_predict(settings.OUT_PATH, settings.IMG_ROWS/settings.RESCALE_FACTOR,
			settings.IMG_COLS/settings.RESCALE_FACTOR,
			settings.EPOCHS, settings.IN_CHANNEL_NO, \
			settings.OUT_CHANNEL_NO, model_fn, settings.MODE)
'''
	train_and_predict(settings.OUT_PATH, settings.IMG_ROWS/settings.RESCALE_FACTOR,
		settings.IMG_COLS/settings.RESCALE_FACTOR,
		settings.EPOCHS, settings.IN_CHANNEL_NO, \
		settings.OUT_CHANNEL_NO, settings.MODEL_FN, settings.MODE)
'''
