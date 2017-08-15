from __future__  import print_function
import argparse
parser = argparse.ArgumentParser(description='Fast Neural Style - generator')
parser.add_argument('--model', default='./model/pre_trained.model', type=str)
parser.add_argument('--input', type=str, help='input image')
parser.add_argument('--output', type=str, help='output image')
args = parser.parse_args()

import numpy as np 
from PIL import Image
from datetime import datetime

import torch
from torch.autograd import Variable

import net

use_cuda = torch.cuda.is_available()

style_model = net.StylePart()
style_model.load_state_dict(torch.load(args.model))

# Origin Image
if use_cuda:
	style_model.cuda()

print('=> Load Origin Image')
start_time = datetime.now().strftime('%H:%M:%S')
img = Image.open(args.input)
img = np.array(img)  # PIL->numpy
img = np.array(img[..., ::-1])  # RGB->BGR
img = img.transpose(2, 0, 1)  # HWC->CHW
img = img.reshape((1, ) + img.shape)  # CHW->BCHW
img = torch.from_numpy(img).float()
img = Variable(img, volatile=True)

if use_cuda:
	img = img.cuda()

print('=> Generating')
style_model.eval()
output = style_model(img)
end_time = datetime.now().strftime('%H:%M:%S')
print('=> Save ')
output = output.data.cpu().clamp(0, 255).byte().numpy()
output = output[0].transpose((1, 2, 0))
output = output[..., ::-1]
output = Image.fromarray(output)
output.save(args.output)
print('Start Time: %s\nEnd Time: %s'%(start_time,end_time))
