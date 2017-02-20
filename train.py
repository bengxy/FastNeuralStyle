import argparse
from datetime import datetime
parser = argparse.ArgumentParser(description='Fast Neural Style')
parser.add_argument('--epoches', default=2, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--style_image', default='images/wave.jpg', type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--checkpoint', default=0, type=int)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--style_size', default=256, type=int)
# weight of vgg and style | According to paper   1:5 works well
parser.add_argument('--rate_content_loss', default=1.0, type=float)
parser.add_argument('--rate_style_loss', default=5.0, type=float)
parser.add_argument('--prefix', default='pre_trained', type=str)
parser.add_argument('--dataset', default='./dataset', type=str)
parser.add_argument('--debug', default=False, type=bool)

args = parser.parse_args()

if args.debug:
	args.dataset = './debug/data'


import torch

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim

from torchvision import transforms
from torchvision import datasets

import utils
import net

use_cuda = torch.cuda.is_available()

transform = transforms.Compose([
	transforms.Scale( args.image_size),  #handle non-square img
	transforms.CenterCrop(args.image_size), 
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x.mul(255))
	])

train_img = datasets.ImageFolder( args.dataset, transform)
train_loader = DataLoader(train_img, batch_size=args.batch_size, num_workers=4)
n_iter = len(train_loader)
print('=> %d Iter Step of 1 Epoch'%n_iter)

#init model
print('=> Init Model')
style_model = net.StylePart() #empyt model
vgg_model = net.Vgg16Part() # fill pretrained vgg
utils.init_vgg16()
vgg_model.load_state_dict( torch.load('model/vgg16.weight'))


# Load style_image
print('=> Init Style Image')
style = utils.img2X(args.style_image, args.style_size)
style =style.repeat(args.batch_size, 1, 1, 1)
style =utils.excg_rgb_bgr(style)

# put on GPU
if use_cuda:
	print('=> Use CUDA')
	style_model.cuda()
	vgg_model.cuda()
	style = style.cuda()

# calc ground truth of style img
style_X = Variable(style, volatile=True)
utils.shift_mean(style_X)

#feature_s = vgg_model(style_X)
print('=> Calculate Style Image Feature')
feature_style = vgg_model(style_X)
gram_style = [utils.gram_matrix(y) for y in feature_style]

# set Loss and Opt
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(style_model.parameters(), lr=args.lr)
print('\n=> Start Training\n')
style_model.train()
start_time = datetime.now().strftime('%H:%M:%S')
print('=> Start Time %s'%start_time)
for epoch in range(args.epoches):
	iter_i = 0
	for batch in train_loader:
		optimizer.zero_grad()
		current_bs = len(batch[1])  # current_bs length as a mask for gram_style
		data = batch[0].clone()
		#data = utils.excg_rgb_bgr(data)
		if use_cuda:
			data = data.cuda()
		# style diff
		X = Variable(data.clone())
		y = style_model(X)

		X_content = Variable(data.clone(), volatile=True)

		# y -> RGB2BGR -> mean -> vgg 
		# batch(xc) -> RGB2BGR -> mean -> vgg
		utils.excg_rgb_bgr(y)
		utils.excg_rgb_bgr(X_content)

		utils.shift_mean(y)
		utils.shift_mean(X_content)

		#feature_hat = vgg_model(y)
		feature_generated = vgg_model(y)
		feature_content = vgg_model(X_content)
		# content diff
		
		# here we use relu2_2: ref to paper
		feature_relu2_2 = Variable(feature_content[1].data, requires_grad=False)

		# content_loss
		L = args.rate_content_loss * loss_fn(feature_generated[1], feature_relu2_2)

		# style_loss :  relu:  1_2 | 2_2 | 3_3 | 4_3
		for m in range(0, len(feature_generated)):
			gram_level = Variable(gram_style[m].data, requires_grad=False)
			L += args.rate_style_loss*loss_fn(utils.gram_matrix(feature_generated[m]), gram_level[0:current_bs, :,:])
		L.backward()
		optimizer.step()
		if iter_i%10 == 0:
			#dt = datetime.now().strftime('%H:%M:%S')
			print('epoch %d \t batch %6d/%6d \t loss %8.4f'%(epoch, iter_i, n_iter, L.data[0]))
			#print('epoch {} batch {}/{}    loss is {}'.format(epoch, iter_i, n_iter, L.data[0]))
		
		if args.checkpoint > 0 and 1 == iter_i % args.checkpoint:
			utils.save_model(style_model, './model/{}_{}_{}.model'.format(args.prefix, epoch, iter_i))
		iter_i = iter_i + 1
	utils.save_model(style_model, './model/{}_{}.model'.format(args.prefix, epoch))
	end_time = datetime.now().strftime('%H:%M:%S')
	print('=> End 1 Epoch Time:%s'%end_time)
utils.save_model(style_model, './model/{}.model'.format(args.prefix))

end_time = datetime.now().strftime('%H:%M:%S')
print('=> Model Finish \n=>Start Time: %s \n=>End Time: %s'%(start_time, end_time))
#

