from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.models import Model
from keras.layers import Lambda
from keras import losses

import numpy as np
import os
import time
import math

from preprocess import *
from helper import *
import settings

import matplotlib
import matplotlib.pyplot as plt

def interactive_overfitting(data_path, img_rows, img_cols, input_no  = 3, output_no = 3,
	fn= "model",  mode=1):

	print('Epsilon {}'.format(K.epsilon()))
	K.set_epsilon(1.0e-7)
	print('Epsilon {}'.format(K.epsilon()))


	print('-'*30)
	print('Loading and preprocessing test data...')
	print('-'*30)
	imgs_test, msks_test = load_data(data_path,"_test")
	imgs_test, msks_test = update_channels(imgs_test, msks_test, input_no, output_no, mode)
	print('Number of test images is ', imgs_test.shape[0])

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	if "-mobile" in fn:
		model		= model5_mobile_unet(False, False, img_rows, img_cols, input_no, output_no, batch_normalization="-bn=False" not in fn, batch_normalization_after="-bnafter=False" not in fn, balanced="-balanced=False" not in fn)
	else:
		model		= model5_MultiLayer(False, False, img_rows, img_cols, input_no,	output_no)

	print('-'*30)
	print('Loading saved weights...')
	print('-'*30)
	model_fn	= os.path.join(data_path, fn)
	model.load_weights(model_fn)
	#model.summary()


	print('-'*30)
	test_no = 84
	print('Predict on test image: (not necessary to call predict)', test_no)
	#print('-'*30)
	test = imgs_test[[test_no]]
	msks_pred = model.predict(test, batch_size=128, verbose=0)

	print('-'*30)
	print('Scoring model on test image: ', test_no)
	#print('-'*30)
	scores = model.evaluate(test, msks_test[[test_no]], batch_size=128, verbose = 0)
	print ("Scores on original wrt gt {0}: {1}".format(model.metrics_names, scores))

	# Generate a new input with guided perturbation (overfitting) based on the previous output
	#
	#
	print('-'*30)
	print('Creating guided perturbation pipeline ...')
	print('-'*30)
	input_img = model.inputs[0]
	output_img = model.outputs[0]

	#
	# define a mask based on the most confident pixels
	confident = 0.5 #0.9999
	threshold_msks_pred = K.greater_equal(output_img, confident)
	modified_msks_pred = K.cast(threshold_msks_pred, dtype='float32')
	modified_msks_pred = output_img
	#
	# Draw
	foreground_strokes = np.full((1, int(img_rows), int(img_cols), int(output_no)), fill_value=0.0, dtype='float32')
	foreground_strokes[0, 88:90, 111:113, 0] = 1 #1000000.0
	#foreground_strokes[0, 75:102, 100:125, 0] = 0.25
	foreground_strokes[:, :, :, 1] = 1.0 - foreground_strokes[:, :, :, 0]
	foreground_strokes_var = K.variable(value=foreground_strokes, dtype='float32', name='foreground_strokes')
	def apply_foreground_strokes(inputs):
		(x, y) = inputs
		return K.concatenate([K.expand_dims(K.maximum(x[:,:,:,0], y[:,:,:,0]), axis=-1), K.expand_dims(K.minimum(x[:,:,:,1], y[:,:,:,1]), axis=-1)])
	def apply_foreground_strokes_shape(input_shape):
		return input_shape[0]
	modified_msks_pred = Lambda(apply_foreground_strokes, output_shape=apply_foreground_strokes_shape)([modified_msks_pred, Input(tensor=foreground_strokes_var)])
	print('Output of lambda? {}'.format(modified_msks_pred))
	#modified_msks_pred = foreground_strokes_var
	#
	# Erase
	background_strokes = np.full((1, int(img_rows), int(img_cols), int(output_no)), fill_value=0.0, dtype='float32')
	background_strokes[0, 51:54, 97:99, 1] = 1.0
	background_strokes[0, 53:56, 98:100, 1] = 1.0
	background_strokes[0, 55:57, 99:101, 1] = 1.0
	background_strokes[:, :, :, 0] = 1.0 - background_strokes[:, :, :, 1]
	background_strokes_var = K.variable(value=background_strokes, dtype='float32', name='background_strokes')
	def apply_background_strokes(inputs):
		(x, y) = inputs
		return K.concatenate([K.expand_dims(K.minimum(x[:,:,:,0], y[:,:,:,0]), axis=-1), K.expand_dims(K.maximum(x[:,:,:,1], y[:,:,:,1]), axis=-1)])
	def apply_background_strokes_shape(input_shape):
		return input_shape[0]
	modified_msks_pred = Lambda(apply_background_strokes, output_shape=apply_background_strokes_shape)([modified_msks_pred, Input(tensor=background_strokes_var)])
	#modified_msks_pred = background_strokes_var
	#
	#
	difference_modified_msks_pred = modified_msks_pred - output_img
	#print('Confident portion of outputs: ', modified_msks_pred)

	# define a function to perform creating a mask from the confident pixels
	modified_inference = K.function([output_img, K.learning_phase()], [modified_msks_pred])
	belief = K.function([output_img, K.learning_phase()], [difference_modified_msks_pred])

	# use the loss function from the model, applied to our confident pixel mask and the previous output, loss(true, predicted)
	#loss = model.loss(modified_msks_pred, output_img)
	#loss = K.categorical_crossentropy(modified_msks_pred, output_img)
	loss = losses.mean_squared_error(modified_msks_pred, output_img)

	# gradients of the loss function wrt the input
	grads = K.gradients(loss, input_img)[0]
	#large_grads = K.cast(K.greater(grads, math.sqrt(K.epsilon())), 'float32')
	#squash = K.ones_like(grads) * large_grads
	#grads = K.sign(grads) * K.maximum(K.abs(grads), K.epsilon())
	#print('Gradients wrt inputs: ', grads)
	#print('Layer {}'.format(model.layers))

	# define a function show we can show the gradients
	normalized_gradients = K.function([input_img, K.learning_phase()], [K.l2_normalize(grads)])
	#normalized_gradients = K.function([input_img, K.learning_phase()], [grads])

	# modify the input placeholder with guided perturbations, modify input to descend gradient
	#step = 0.10 * (K.max(grads) - K.min(grads)) #2000000 #0.2 # 20000
	#delta = step * K.l2_normalize(grads) # K.sign(grads)
	step = 2
	delta = step * K.l2_normalize(grads)
	modified_input_img = input_img - delta
	#modified_input_img = (modified_input_img - K.mean(modified_input_img)) / K.std(modified_input_img)    # re-normalize the input image
	#print('Perturbed input: ', modified_input_img)

	# define a function to perform guided perturbation
	perturb = K.function([input_img, K.learning_phase()], [modified_input_img])


	clim = None
	clim_grad = None
	for iter in range(15):
		# apply guided perturbations to the input
		print('-'*30)
		print('Perturb the input (test image)')
		print('-'*30)
		perturbed_input_img = perturb([test, 0]) # pass in the value for K.learning_phase()

		# score the perturbed input
		print('-'*30)
		print('Scoring model on perturbed input (overfit) ...')
		#print('-'*30)
		scores = model.evaluate(perturbed_input_img, msks_test[[test_no]], batch_size=128, verbose = 0)
		print ("Scores on guided perturbation wrt gt {0}: {1}".format(model.metrics_names, scores))
		scores = model.evaluate(perturbed_input_img, modified_inference([msks_pred, 0]), batch_size=128, verbose = 0)
		print ("Scores on guided perturbation wrt markup {0}: {1}".format(model.metrics_names, scores))

		fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,6))
		axes[0,0].set_title('Original/Perturbed @{}'.format(iter))
		im = axes[0, 0].imshow(test[0].squeeze(), clim=clim)
		clim=im.properties()['clim']
		axes[0, 0].axis('off')
		axes[0, 1].set_title('Inference')
		axes[0, 1].imshow(msks_pred[0].squeeze()[:,:,0])  # [:,:,0]
		axes[0, 1].axis('off')
		axes[0, 2].set_title('Markup')
		axes[0, 2].imshow(modified_inference([msks_pred, 0])[0].squeeze()[:,:,0])
		axes[0, 2].axis('off')
		axes[0, 3].set_title('GT')
		axes[0, 3].imshow(msks_test[test_no].squeeze()[:,:,0])
		axes[0, 3].axis('off')
		axes[1, 0].set_title('Diff-inference')
		axes[1, 0].imshow(belief([msks_pred, 0])[0].squeeze()[:,:,0])
		axes[1, 0].axis('off')
		axes[1, 1].set_title('Normalized gradients')
		print('Normalized gradients size {}'.format(normalized_gradients([test, 0])[0].shape))
		im_grad = axes[1, 1].imshow(normalized_gradients([test, 0])[0].squeeze(), clim=clim_grad)
		#clim_grad=im_grad.properties()['clim']
		axes[1, 1].axis('off')
		axes[1, 2].set_title('Perturbed input @{}X'.format(step))
		axes[1, 2].imshow(perturbed_input_img[0].squeeze(), clim=clim)
		axes[1, 2].axis('off')
		axes[1, 3].set_title('GP Inference')
		perturbed_msks_pred = model.predict(perturbed_input_img, batch_size=128, verbose=0)
		axes[1, 3].imshow(perturbed_msks_pred[0].squeeze()[:,:,0])
		axes[1, 3].axis('off')
		fig.canvas.draw()

		test = perturbed_input_img[0]
		msks_pred = perturbed_msks_pred

if __name__ =="__main__":
	model = "brainWholeTumor-mobile-balanced=True-bn=True-bnafter"
	model = "brainWholeTumor-mobile-balanced=True-bn=False"
	model = "brainWholeTumor-mobile-out=2-balanced=True-bn=False"
	model_number = 9
	model_fn = '{}__epoch={:03d}.hdf5'.format(model, model_number)
	interactive_overfitting(settings.OUT_PATH, settings.IMG_ROWS/settings.RESCALE_FACTOR,
		settings.IMG_COLS/settings.RESCALE_FACTOR,
		settings.IN_CHANNEL_NO, \
		settings.OUT_CHANNEL_NO, model_fn, mode=settings.MODE)
