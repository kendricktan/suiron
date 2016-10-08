# Suiron
### Machine Learning for RC Cars 

## Prediction visualization (green = actual, blue = prediction)
![](https://thumbs.gfycat.com/DarlingForkedAcaciarat-size_restricted.gif)

## Click the video below to see it in action!
[![IMAGE ALT TEXT](http://img.youtube.com/vi/tFwCyHdAWf0/0.jpg)](https://youtu.be/tFwCyHdAWf0 "Machine Learning Car")

## Dependencies
#### __Python 2.7__ was chosen as it was supported by all the libraries used at the time
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python-opencv python-dev

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL

sudo pip install -r requirements.txt
```

## Collecting data
```
python collect.py
```

## Visualizing collected data
```
python visualize_collect.py
```

## Training data
```
python train.py
```

## Visualizing predicted data
```
python visualize_predict.py
```

# References

Blog Post detailing how the hardware and software communicate - [Communicating between RC Car and the On-board Computer - Jabelone](http://jabelone.com.au/blog/make-autonomous-car-code-included/)


Communication between hardware and software repo - [car-controller](https://github.com/jabelone/car-controller)

Neural Network architecture was based on NVIDIA's Self-driving car paper - [End-To-End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)
