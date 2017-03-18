# FastNeuralStyle by Pytorch
Fast Neural Style for Image Style Transform by Pytorch

## Reference Paper and DataSet: 
#### Paper:
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

#### Dataset:
[coco80k 2014](http://mscoco.org/dataset/#download)

## Configure

put dataset under ./dataset

    dataset 
     --train2014
     --put_coco_train_here

## Usage

default : 
`python train.py `

show help
`python train.py -h`

generate
`python go.py`

show generate help
`python go.py -h`

## Result

**Style**
![style](https://github.com/bengxy/FastNeuralStyle/tree/master/images/wave.jpg)
**Origin**
![origin](https://github.com/bengxy/FastNeuralStyle/tree/master/images/westlake.jpg)
**Generated**
![gen](https://github.com/bengxy/FastNeuralStyle/tree/master/images/output_westlake.jpg)

