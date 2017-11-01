import h5py
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.io import loadmat
from matplotlib.patches import Rectangle

class DigitStructure:
    
    def __init__(self, src):
        self.src = src 
        self.struct_src = self.src + '/digitStruct.mat'
        self.data = h5py.File(self.struct_src , 'r')
        self.digitStructName = self.data['digitStruct']['name']
        self.digitStructBbox = self.data['digitStruct']['bbox']
    
    def get_name(self,n):
        name = ''.join([chr(c[0]) for c in self.data[self.digitStructName[n][0]].value])
        return name
        
    def _get_attr(self,attr):
        if (len(attr) > 1):
            attr = [self.data[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr
    
    def get_attr(self, n):
        attr = {}
        bb = self.digitStructBbox[n].item()
        attr['height'] = self._get_attr(self.data[bb]["height"])
        attr['left'] = self._get_attr(self.data[bb]["left"])
        attr['top'] = self._get_attr(self.data[bb]["top"])
        attr['width'] = self._get_attr(self.data[bb]["width"])
        attr['label'] = self._get_attr(self.data[bb]["label"])
        return attr
    
    def get_crop(self, n):
        bbox = {}
        cropped = {}
        attr = self.get_attr(n)
        min_left, min_top, max_right, max_bottom = (min(attr['left']),
                                                    min(attr['top'] ),
                                                    max(map(lambda x, y: x + y, attr['left'], attr['width'])),
                                                    max(map(lambda x, y: x + y, attr['top'] , attr['height'])))
        center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                        (min_top + max_bottom) / 2.0,
                                        max(max_right - min_left, max_bottom - min_top))
        bbox['left'], bbox['top'], bbox['width'], bbox['height'] = (center_x - max_side / 2.0, 
                                                                     center_y - max_side / 2.0, 
                                                                     max_side,
                                                                     max_side)
        cropped['left'], cropped['top'], cropped['width'], cropped['height'] = (int(round(bbox['left'] - 0.15 * bbox['width'])),
                                                                                int(round(bbox['top'] - 0.15 * bbox['height'])),
                                                                                int(round(bbox['width'] * 1.3)),
                                                                                int(round(bbox['height'] * 1.3)))
        return cropped, bbox
    

    def show_img_bbox(self, n):
        image = Image.open(self.src + '/' + self.get_name(n))
        attr = self.get_attr(n)
        cropped, bbox = self.get_crop(n)
        plt.figure()
        currentAxis = plt.gca()
        currentAxis.imshow(image)
        currentAxis.add_patch(Rectangle((cropped['left'], cropped['top']), cropped['width'], cropped['height'], fill=False, edgecolor='red'))
        currentAxis.add_patch(Rectangle((bbox['left'], bbox['top']), bbox['width'], bbox['height'], fill=False, edgecolor='green'))
        for attr_left, attr_top, attr_width, attr_height in zip(attr['left'], attr['top'], attr['width'], attr['height']):
            currentAxis.add_patch(Rectangle((attr_left, attr_top), attr_width, attr_height, fill=False, edgecolor='white', linestyle='dotted'))
        plt.show()
    
    def get_digit_structure(self, n):
        d = {}
        d['crop'], d['bbox'] = self.get_crop(n)
        d['name'] = self.get_name(n)
        d['attr'] = self.get_attr(n)
        return d
    
    def get_all_digit_structure(self):
        digits = [self.get_digit_structure(n) for n in range(len(self.digitStructName))]
        return digits
    
    def print_img_data(self,n):
        d = self.get_digit_structure(n)
        print('cropped: left=%d, top=%d, width=%d, height=%d' % (d['crop']['left'], d['crop']['top'], d['crop']['width'], d['crop']['height']))
        print('bbox: left=%d, top=%d, width=%d, height=%d' % (d['bbox']['left'], d['bbox']['top'], d['bbox']['width'], d['bbox']['height']))
        for i in range(len(d['attr']['label'])):
            print('label: %d' % (d['attr']['label'][i]))
            print('attr: left=%d, top=%d, width=%d, height=%d' % (d['attr']['left'][i], d['attr']['top'][i], d['attr']['width'][i], d['attr']['height'][i]))

            