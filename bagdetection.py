import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import datetime
from pygame import mixer

options = {
	'model': 'cfg/yolo.cfg',
	'load': 'bin/yolov2.weights',
	'threshold': 0.2,
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
	stime = time.time()
	ret, frame = capture.read()
	if ret:
		results = tfnet.return_predict(frame)
		count=0
		for color, result in zip(colors, results):
			tl = (result['topleft']['x'], result['topleft']['y'])
			br = (result['bottomright']['x'], result['bottomright']['y'])
			label = result['label']
			if label=='person':
				count=1
			currentT = datetime.datetime.now()
			if label == 'backpack'  or label=='suitcase' or label == 'handbag' and count==1:
				mixer.init()
				mixer.music.load("siren.mp3")
				mixer.music.play()
			confidence = result['confidence']
			text = '{}: {:.0f}%'.format(label, confidence * 100)
			frame = cv2.rectangle(frame, tl, br, color, 5)
			frame = cv2.putText(
				frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
		cv2.imshow('frame', frame)
		print('FPS {:.1f}'.format(1 / (time.time() - stime)))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	

capture.release()
cv2.destroyAllWindows()
