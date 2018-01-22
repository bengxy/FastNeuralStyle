# FastNeuralStyle by Pytorch

Fast Neural Style for Image Style Transform by Pytorch

This is famous Fast Neural Style of Paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) of Feifei Li.

## Result

Style Image | Origin Image | Generated Image 
|:---:|:---:|:---:|
![style](https://github.com/bengxy/FastNeuralStyle/raw/master/images/wave.jpg)|![style](https://github.com/bengxy/FastNeuralStyle/raw/master/images/nymph.jpg)|![style](https://github.com/bengxy/FastNeuralStyle/raw/master/images/output_nymph.jpg)
![style](https://github.com/bengxy/FastNeuralStyle/raw/master/images/wave.jpg)|![style](https://github.com/bengxy/FastNeuralStyle/raw/master/images/chicago.jpg)|![style](https://github.com/bengxy/FastNeuralStyle/raw/master/images/output_chicago.jpg)


# How to Run

## Train and Test
### Training DataSet

I **Strongly Recommend** you to download coco80k 2014 dataset from 
[coco80k 2014](http://mscoco.org/dataset/#download)
 (This is also used by the original paper)

You can use your own huge image dataset as well

### Configure

put dataset under ./dataset, the folder will looks like

    dataset 
     --train2014
     --put_coco_train_here

### Run

	train 
	python train.py -h

	test
	python go.py -h
	

# Pretrained Model

models are saved in **./model** folder by default.

I also release two pretrained model .

[wave.model](https://cloud.bengxy.com/index.php/s/wjy9KecTIieMILH)

You can download and put it under ./model folder


