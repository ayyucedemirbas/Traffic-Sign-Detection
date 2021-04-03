# Traffic-Sign-Detection
Custom tensorflow object detection for traffic signs

sudo pip3 install opencv-contrib-python

sudo pip3 install numpy

sudo pip3 install scikit-learn

sudo pip3 install scikit-image

sudo pip3 install imutils

sudo pip3 install tensorflow==2.0.0


python3 train.py --dataset gtsrb-german-traffic-sign \
	--model output/trafficsignnet.model
  
  
python3 predict.py --model output/trafficsignnet.model \
	--images gtsrb-german-traffic-sign/Test \
	--examples examples
