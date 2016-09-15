# Suiron
#### Machine Learning for RC Cars


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

## Visualizing predicted data (off existing training set)
```
python visualize_predict.py
```