import argparse
from datetime import datetime
parser = argparse.ArgumentParser(description='Neural Style')
parser.add_argument('--epoches', default=2, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--style_image', default='images/wave.jpg', type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--checkpoint', default=10000, type=int)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--style_size', default=256, type=int)
# weight of vgg and style
parser.add_argument('--lambda_feature', default=1.0, type=float)
parser.add_argument('--lambda_style', default=5.0, type=float)
args = parser.parse_args()
args.prefix = 'style_prefix'
#
args.dataset = './dataset'

#
import torch
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision import datasets
#import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import optim
args.cuda = torch.cuda.is_available()



transform = transforms.Compose([
	transforms.Scale( args.image_size), 
	transforms.CenterCrop(args.image_size), 
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x.mul(255))
	])

train = datasets.ImageFolder( args.dataset, transform)
train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=4)
n_iter = len(train_loader)

import utils
import net

#init model
utils.init_vgg16()
style_model = net.StylePart()
vgg_model = net.Vgg16Part()
vgg_model.load_state_dict( torch.load('model/vgg16feature'))

# set Loss and Opt
optimizer = optim.Adam(style_model.parameters(), lr=args.lr)


# Load style_image
style = utils.img2X(args.style_image, args.style_size)
style =style.repeat(args.batch_size, 1, 1, 1)
style =utils.excg_rgb_bgr(style)

# put on GPU
if args.cuda:
	print('use cuda')
	style_model.cuda()
	#vgg_model = torch.nn.DataParallel(style_model, device_ids=[0,1,2,3])

	vgg_model.cuda()
	#vgg_model = torch.nn.DataParallel(vgg_model, device_ids=[0,1,2,3])
	style = style.cuda()

# calc ground truth style label
style_var = Variable(style, volatile=True)
utils.shift_mean(style_var)

#print( style_var)

feature_s = vgg_model(style_var)
gram_s = [utils.gram_matrix(y) for y in feature_s]


loss_fn = torch.nn.MSELoss()

style_model.train()

for epoch in range(args.epoches):
	iter_i = 0
	for batch in train_loader:
		optimizer.zero_grad()

		data = batch[0].clone()
		data = utils.excg_rgb_bgr(data)
		if args.cuda:
			data = data.cuda()
		# style diff
		x = Variable(data.clone())
		y = style_model(x)
		utils.shift_mean(y)
		feature_hat = vgg_model(y)

		# content diff
		xc = Variable(data.clone(), volatile=True)
		utils.shift_mean(xc)
		feature = vgg_model(xc)

		feature_v = Variable(feature[1].data, requires_grad=False)
		L = args.lambda_feature * loss_fn(feature_hat[1], feature_v)
		for m in range(0, len(feature_hat)):
			gram_v = Variable(gram_s[m].data, requires_grad=False)
			L += args.lambda_style*loss_fn(utils.gram_matrix(feature_hat[m]), gram_v)
		L.backward()
		optimizer.step()
		if iter_i%10 == 0:
			dt = datetime.now().strftime('%H:%M:%S')
			print('{} epoch {} batch {}/{}    loss is {}'.format(dt, epoch, iter_i, n_iter, L.data[0]))
		
		if args.checkpoint > 0 and 1 == iter_i % args.checkpoint:
			utils.save_model(style_model, '{}_{}_{}.pth'.format(args.prefix, epoch, iter_i))
		iter_i = iter_i + 1
	utils.save_model(style_model, '{}_{}.pth'.format(args.prefix, epoch))
utils.save_model(style_model, '{}.pth'.format(args.prefix))


#

