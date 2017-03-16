import net
from PIL import Image
import os
import torch
from torch.autograd import Variable
from torch.utils.serialization import load_lua
import numpy  as np
# Load VGG16 for torch and save
def init_vgg16(model_folder ='model'):
	"""load the vgg16 model feature"""
	if not os.path.exists(model_folder+'/vgg16.weight'):
		if not os.path.exists(model_folder+'/vgg16.t7'):
			os.system('wget http://bengxy.com/dataset/vgg16.t7 '+model_folder+'/vgg16.t7')
		vgglua = load_lua(model_folder + '/vgg16.t7')
		vgg= net.Vgg16Part()
		for ( src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
			dst[:] = src[:]
		torch.save(vgg.state_dict(), model_folder+'/vgg16.weight')

# Loss
def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w*h)
	features_t = features.transpose(1,2)
	gram = features.bmm(features_t) /(ch*h*w)
	return gram

# Image Processing
def shift_mean(X):
	"""Image Color normalization"""
	tensortype = type(X.data)
	mean = tensortype(X.data.size())
	mean[:, 0, :, :] = 103.939
	mean[:, 1, :, :] = 116.779
	mean[:, 2, :, :] = 123.680
	#mean[:,0:3,:,:] = tensor([103.939, 116.779, 123.680])
	X -= Variable(mean)

# save tensor in gpu to disk
def X2img(X, image_name, mod='rgb'):
	if mod=='bgr':
		(b,g,r) = torch.chunk(X, 3)
		X = torch.cat((r,g,b))
	img = X.clone().cpu().clamp(0,255).numpy()
	img = img.transpose(1,2,0).astype('uint8')
	img = Image.fromarray(img)
	img.save(image_name)

# load image
def img2X(image_name, size=None):
	img = Image.open(image_name)
	if size is not None:
		img = img.resize((size, size), Image.ANTIALIAS)
	img = np.array(img).transpose(2,0,1)
	img =torch.from_numpy(img).float()
	return img

# change rgb2bgr or bgr2rgb
def excg_rgb_bgr(batch):
	batch=batch.transpose(0,1)
	(r,g,b) = torch.chunk(batch, 3)
	batch = torch.cat((b,g,r))
	batch = batch.transpose(0,1)
	return batch

# Save model
def save_model(model, name):
	state = model.state_dict()
	for key in state:
		state[key] = state[key].clone().cpu()
	torch.save(state, name)
