from __future__ import print_function
import numpy as np
import cv2
import Tkinter as tk
import sys
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class Application:
	def __init__(self, s='y'):
		self.s = s
		self.b_threshold = 170
		self.edge_threshold1 = 100
		self.edge_threshold2 = 255
		self.edge_aperture = 3
		self.frame_counter = 0
		self.master = tk.Tk()
		self.gui()


	def show_frame(self):
		if self.s == 'y':
			video_capture = cv2.VideoCapture(0)
		else:
			video_source = 'data/video/87.webm'
			video_capture = cv2.VideoCapture(video_source)
		if video_capture.isOpened():
			video_capture.read()


		maxValue = 255
		contour_color = (100, 0, 0)
		contour_width = 2

		_, frame = video_capture.read()
		if self.s == 'n':
			self.frame_counter += 1
			if self.frame_counter == video_capture.get(cv2.CAP_PROP_FRAME_COUNT)-1:
				self.frame_counter = 0
				video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

		orig = frame.copy()

		imgray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)

		blurred = cv2.GaussianBlur(imgray, (7, 7), 0)

		_, self.thresh = cv2.threshold(blurred,self.b_threshold,maxValue,cv2.THRESH_BINARY)

		edged = cv2.Canny(self.thresh, self.edge_threshold1, self.edge_threshold2, self.edge_aperture)

		median = cv2.medianBlur(self.thresh,7)

		im2, contours, hierarchy = cv2.findContours(median.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		cv2.drawContours(im2, contours, -1, contour_color , contour_width)

		rectangles = [cv2.boundingRect(rect) for rect in contours]

		self.bbox_list = []
		for rect in rectangles:
			x1,y1,x2,y2 = int(rect[0] - 0.15 * rect[2]), \
					  	  int(rect[1] - 0.15 * rect[3]), \
						  int(rect[0] + rect[2] * 1.15), \
					      int(rect[1] + rect[3] * 1.15)
			bbox = cv2.rectangle(orig, (x1,y1), (x2,y2), 100, 3)
			self.bbox_list.append([x1,y1,x2,y2])



		if self.dropVar.get() == 'Original':
			img = Image.fromarray(frame)
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
			img = Image.fromarray(bbox)
		else:
			img = 0


		imgtk = ImageTk.PhotoImage(image=img)
		self.lmain.imgtk = imgtk
		self.lmain.configure(image=imgtk)
		self.lmain.after(1000, self.show_frame)

	def gui(self):

		self.master.bind('<Escape>', lambda e: self.master.quit())
		self.lmain = tk.Label(self.master)
		self.lmain.grid(row=0, rowspan=20, column=0)

		row=0
		l_b = tk.Label(self.master, text="Select img type").grid(row=row, column=1)
		row+=1
		optionList = ["Original", "Gray", "Threshold", "Edge", "Median", "Contours", "Bbox"]
		self.dropVar = tk.StringVar(self.master)
		self.drop_set = "Threshold"
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
		l_edge = tk.Label(self.master, text="Canny edge").grid(row=row, column=1)
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
		self.b1 = tk.Button(self.master, text='Crop', command=self.crop_bbox)
		self.b1.grid(row=row, column=1)
		row+=1

	def set_threshold(self, val):

		self.b_threshold = self.w1.get()
		self.edge_threshold1 = self.w2.get()
		self.edge_threshold2 = self.w3.get()
		self.edge_aperture = self.w4.get()

	def crop_bbox(self):
		self.cropped_numbers = []
		for im in self.bbox_list:
			# y: y+h, x: x+w
			crop = self.thresh[im[1]:im[3], im[0]:im[2]].copy()
			resized_image = cv2.resize(crop, (32, 32))
			self.cropped_numbers.append(resized_image)
			plt.imshow(resized_image)
			plt.show()



if __name__ == '__main__':
	if len(sys.argv) < 2:
		s = 'n'
	else:
		s = sys.argv[1]

	app = Application(s=s)
	app.master.after(1,app.show_frame())
	app.master.mainloop()
