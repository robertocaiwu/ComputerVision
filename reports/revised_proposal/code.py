frame = cv2.resize(frame,(1080,720), interpolation = cv2.INTER_CUBIC)
imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(imgray, (5, 5), 0)
_, thresh = cv2.threshold(blurred,self.b_threshold,maxValue,cv2.THRESH_BINARY_INV)
edged = cv2.Canny(thresh, self.edge_threshold1, self.edge_threshold2, self.edge_aperture)
median = cv2.medianBlur(self.thresh,5)
