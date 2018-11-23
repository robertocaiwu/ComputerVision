# Roberto Cai / Ramesh Kumar
from __future__ import print_function
import numpy as np
import cv2
try:
	import tkinter as tk
except:
	import Tkinter as tk
import sys
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from keras.models import load_model

class Application:
	def __init__(self, s='0'):
		self.s = s
		self.b_threshold = 170
		self.edge_threshold1 = 100
		self.edge_threshold2 = 255
		self.edge_aperture = 3
		self.frame_counter = 0
		self.load_video_capture()
		self.master = tk.Tk()
		self.gui()
		self.digit_classifier = load_model('./model/TrainedModelMnist.hdf5')
		self.crop_size = 28
		self.text_color = (255, 0, 0)
		self.recognize = 0

	# Loads the video capture object to get video feed from camera or file:
	# 0: laptop camera
	# 1: external usb camera
	# 2: read from video file
	def load_video_capture(self):
		if self.s == '0':
			self.video_capture = cv2.VideoCapture(0)
		elif self.s == '1':
			self.video_capture = cv2.VideoCapture(1)
		else:
			self.video_source = 'data/video/7.mp4'
			self.video_capture = cv2.VideoCapture(self.video_source)
		if self.video_capture.isOpened():
			print("Successfully opened a camera.")
			self.video_capture.read()
		else:
			print("Not opened.")

	# Contains main code for pre-prosessing from video feed
	def show_frame(self):
		maxValue = 255
		contour_color = (100, 0, 0)
		contour_width = 2

		_, self.frame = self.video_capture.read()

		# Only applies for video read from file.
		# Resets the frame counter at the end to restart video.
		# Resize video to 1080,720 resolution
		if self.s == '2':
			self.frame_counter += 1
			if self.frame_counter == self.video_capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)-1:
				self.frame_counter = 0
				self.video_capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
			self.frame = cv2.resize(self.frame,(1080,720), interpolation = cv2.INTER_CUBIC)

		rgb=cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
		orig = rgb.copy()
		imgray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)

		# Pre-processing operations:
		# GaussianBlur
		# threshold
		# Canny
		# medianBlur
		blurred = cv2.GaussianBlur(imgray, (5, 5), 0)
		if not self.inv.get():
			_, self.thresh = cv2.threshold(blurred,self.b_threshold,maxValue,cv2.THRESH_BINARY)
		else:
			_, self.thresh = cv2.threshold(blurred,self.b_threshold,maxValue,cv2.THRESH_BINARY_INV)

		edged = cv2.Canny(self.thresh, self.edge_threshold1, self.edge_threshold2, self.edge_aperture)
		median = cv2.medianBlur(self.thresh,5)

		im2 = median.copy()

		# Extracting contours
		if self.canny.get():
			contours, hierarchy = cv2.findContours(edged,cv2.cv.CV_RETR_EXTERNAL,cv2.cv.CV_CHAIN_APPROX_SIMPLE)
		else:
			contours, hierarchy = cv2.findContours(im2,cv2.cv.CV_RETR_EXTERNAL,cv2.cv.CV_CHAIN_APPROX_SIMPLE)

		# Drawing contours and getting bounding box coordinates
		rectangles = list()
		for c in contours:
			area = cv2.contourArea(c)

			if area > 50:
				cv2.drawContours(im2, c, -1, contour_color , contour_width)
				rectangles.append(cv2.boundingRect(c))

		# Drawing bounding box
		self.bbox = orig
		self.bbox_list = []
		for rect in rectangles:
			side = max(int(rect[2])* 1.2, int(rect[3])* 1.2)
			if side > 10 and side < 400:
				center_x, center_y = ((rect[0] + rect[2]/ 2.0) ,
									  (rect[1] + rect[3]/ 2.0) )
				x1,y1,x2,y2 = int(center_x - side/2), \
							  int(center_y - side/2), \
							  int(center_x + side/2), \
							  int(center_y + side/2)
				cv2.rectangle(self.bbox, (x1,y1), (x2,y2), 100, 3)
				self.bbox_list.append([x1,y1,x2,y2])

		# Crop each bounding box and send to classifier.
		# Draws a label for each predictions
		if self.recognize:
			for im in self.bbox_list:
				# y: y+h, x: x+w
				crop = self.thresh[im[1]:im[3], im[0]:im[2]].copy()
				try:
					resized_image = cv2.resize(crop, (self.crop_size, self.crop_size))
					predictions = self.digit_classifier.predict(resized_image.reshape(1,self.crop_size,self.crop_size,1))

					digit_label_arg = np.argmax(predictions)

					self.draw_text(im, self.bbox, str(digit_label_arg), self.text_color,
								   10, -10, 1, 1)
				except:
					pass

		# Show image type in tkinter GUI
		if self.dropVar.get() == 'Original':
			img = Image.fromarray(rgb)
		elif self.dropVar.get() == 'Gray':
			img = Image.fromarray(imgray)
		elif self.dropVar.get() == 'Threshold':
			img = Image.fromarray(self.thresh)
		elif self.dropVar.get() == 'Edge':
			img = Image.fromarray(edged)
		elif self.dropVar.get() == 'Median':
			img = Image.fromarray(median)
		elif self.dropVar.get() == 'Contours':
			img = Image.fromarray(im2)
		elif self.dropVar.get() == 'Bbox':
			img = Image.fromarray(self.bbox)
		else:
			img = 0

		self.save_img = np.asarray(img)
		imgtk = ImageTk.PhotoImage(image=img)
		self.lmain.imgtk = imgtk
		self.lmain.configure(image=imgtk)
		self.lmain.after(100, self.show_frame)

	def gui(self):
		self.master.bind('<Escape>', lambda e: self.master.quit())
		self.lmain = tk.Label(self.master)
		self.lmain.grid(row=0, rowspan=20, column=0)
		# self.lmain.pack()
		row=0
		l_b = tk.Label(self.master, text="Select img type").grid(row=row, column=1)
		row+=1
		optionList = ["Original", "Gray", "Threshold", "Edge", "Median", "Contours", "Bbox"]
		self.dropVar = tk.StringVar(self.master)
		self.drop_set = "Bbox"
		self.dropVar.set(self.drop_set)
		self.combo = tk.OptionMenu(self.master, self.dropVar, *optionList,
								   command=self.set_threshold)
		self.combo.grid(row=row, column=1)
		row+=1
		l_b = tk.Label(self.master, text="Binary threshold").grid(row=row, column=1)
		row+=1
		self.w1 = tk.Scale(self.master, from_=0, to=255, tickinterval=50,
						   orient='horizontal', length=200,
						   command=self.set_threshold)
		self.w1.set(170)
		self.w1.grid(row=row, column=1)
		row+=1
		self.inv = tk.IntVar()
		self.chk1 = tk.Checkbutton(self.master, text="Invert Binary",
								   variable=self.inv)
		self.chk1.grid(row=row, column=1)
		row+=1
		# l_edge = tk.Label(self.master, text="Canny edge").grid(row=row, column=1)
		self.canny = tk.IntVar()
		self.chk2 = tk.Checkbutton(self.master, text="Canny edge",
								   variable=self.canny)
		self.chk2.grid(row=row, column=1)
		row+=1
		self.w2 = tk.Scale(self.master, from_=0, to=100,tickinterval=50,
						   orient='horizontal', length=200,
						   command=self.set_threshold)
		self.w2.set(100)
		self.w2.grid(row=row, column=1)
		row+=1
		self.w3 = tk.Scale(self.master, from_=100, to=255,tickinterval=50,
						   orient='horizontal', length=200,
						   command=self.set_threshold)
		self.w3.set(255)
		self.w3.grid(row=row, column=1)
		row+=1
		self.w4 = tk.Scale(self.master, from_=0, to=255,tickinterval=50,
						   orient='horizontal', length=200,
						   command=self.set_threshold)
		self.w4.set(3)
		self.w4.grid(row=row, column=1)
		row+=1
		self.b1 = tk.Button(self.master, text='Plot', command=self.plt_bbox)
		self.b1.grid(row=row, column=1)
		row+=1
		self.b2 = tk.Button(self.master, text='Recognize', command=self.crop_bbox)
		self.b2.grid(row=row, column=1)
		row+=1
		self.b3 = tk.Button(self.master, text='Save Img', command=self.save_img)
		self.b3.grid(row=row, column=1)
		row+=1

	def digit_prediction(self, img):
		img_list = np.asarray(img).reshape(len(img), self.crop_size, self.crop_size, 1)
		prediction = self.digit_classifier.predict(img_list)
		return np.argmax(prediction)

	def set_threshold(self, val):
		self.b_threshold = self.w1.get()
		self.edge_threshold1 = self.w2.get()
		self.edge_threshold2 = self.w3.get()
		self.edge_aperture = self.w4.get()

	def plt_bbox(self):
		self.cropped_numbers = []
		current_bbox = self.bbox_list[:]
		for im in current_bbox:
			# y: y+h, x: x+w
			crop = self.thresh[im[1]:im[3], im[0]:im[2]].copy()
			resized_image = cv2.resize(crop, (self.crop_size, self.crop_size))
			self.cropped_numbers.append(resized_image)
			print(resized_image.shape)
			plt.imshow(resized_image)
			plt.show()

	def crop_bbox(self):
		if self.recognize == 1:
			self.recognize = 0
		else:
			self.recognize = 1

	def save_img(self):
		cv2.imwrite('img.png', self.save_img)

	def draw_bounding_box(self, face_coordinates, image_array, color):
		x, y, w, h = face_coordinates
		cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

	def draw_text(self, coordinates, image_array, text, color, x_offset=0, y_offset=0,
												font_scale=2, thickness=2):
		x, y = coordinates[:2]
		cv2.putText(image_array, text, (x + x_offset, y + y_offset),
					cv2.FONT_HERSHEY_SIMPLEX,
					font_scale, color, thickness, cv2.CV_AA)



if __name__ == '__main__':
	if len(sys.argv) < 2:
		s = '0'
	else:
		s = sys.argv[1]
	print(sys.argv, len(sys.argv))
	app = Application(s=s)
	app.master.after(1,app.show_frame())
	app.master.mainloop()