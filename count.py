import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import datetime
from pygame import mixer

options = {
	'model': 'cfg/tiny-yolo-voc.cfg',
	'load': 'bin/tiny-yolo-voc.weights',
	'threshold': 0.2,
	'labels': 'person',
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture('ch01_20190330132014.mp4')
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
tr = (0,50)
W = None
H = None
count=0
while True:
	stime = time.time()
	ret, frame = capture.read()
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	if ret:
		results = tfnet.return_predict(frame)
		for color, result in zip(colors, results):
			tl = (result['topleft']['x'], result['topleft']['y'])
			br = (result['bottomright']['x'], result['bottomright']['y'])
			label = result['label']
			print(tl[0],tl[1], W // 2)
			if label=='person' and (tl[0] > W //2 and tl[0] < W // 2 + 10):
				count = count + 1
			currentT = datetime.datetime.now()
			confidence = result['confidence']
			text = '{}: {:.0f}%'.format(label, confidence * 100)
			frame = cv2.rectangle(frame, tl, br, color, 5)
			frame = cv2.putText(
				frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
		text2 = '{}: {:.0f}'.format('count',count)
		frame = cv2.putText(
			frame, text2, tr, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
		cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 255, 255), 2)
		cv2.imshow('frame', frame)
		
		'''if count > 10:
			print("HIGH ALERT !!!!crowd detected!!!!")
			mixer.init()
			mixer.music.load("siren.mp3")
			mixer.music.play()'''
		print('FPS {:.1f}'.format(1 / (time.time() - stime)))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()
