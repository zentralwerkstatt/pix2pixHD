from anth_midi import Controller
with open('anth_vert.glsl', 'r') as f: vertex=f.read()
with open('anth_frag.glsl', 'r') as f: fragment=f.read()
with open('anth_frag_c.glsl', 'r') as f: fragment_c=f.read()
import cv2
import numpy as np
from vispy import app
from vispy import gloo
from vispy.gloo.util import _screenshot
import time
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import util.util as util
from threading import Thread
from queue import Queue
from queue import Empty
import os
import math
import av

# Global options
W = 1280
H = 720
W_HD = 1920
H_HD = 1080
NET = 864
FPS = 60
RATE_MAX = 20

# Offset measures

W2 = int(W/2)
# Top and bottom offset for WH centered in NET (WH>H)		
TOP_WH_IN_NET = int((NET-H) / 2)
BOTTOM_WH_IN_NET = int(((NET-H) / 2) + H)

# Left and right offset for WH/2 centered in WH (WH/2<W)
LEFT_WH2_IN_WH = int((W-(W/2)) / 2)
RIGHT_WH2_IN_WH = int(((W-(W/2)) / 2) + (W/2))

# Left and right offset for NET centered in WH (NET<WH)
LEFT_NET_IN_WH = int((W-NET) / 2)
RIGHT_NET_IN_WH = int(((W-NET) / 2) + NET)

# Left and right offset for WH/2 centered in NET (WH/2<NET)
LEFT_WH2_IN_NET = int((NET-W2) / 2)
RIGHT_WH2_IN_NET = int(((NET-W2) / 2) + W2)

# Communicate between projector and control windows
black = np.zeros((H, W, 3), dtype=np.uint8)
control = [black.copy(), black.copy(), black.copy(), black.copy()]

class VideoSource():

	def __init__(self, source, is_video=True, is_pyav=False, w=W_HD, h=H_HD, fps=FPS):
		self.source = source
		self.is_video = is_video # Is this a webcam?
		self.is_pyav = is_pyav # Do we want this to be played by pyav?

		if is_video:
			if not is_pyav:
				self._len = source.get(cv2.CAP_PROP_FRAME_COUNT)
				self._frame = 0
			else:
				self.stream = self.source.decode(video=0)

		if not is_video:
			_ = self.source.set(cv2.CAP_PROP_FRAME_WIDTH, w)
			_ = self.source.set(cv2.CAP_PROP_FRAME_HEIGHT, h) 
			_ = self.source.set(cv2.CAP_PROP_FPS, fps)

	def read(self, rate):
		# Give some space at the beginning and end so we never reach it
		if (self.is_video):
			if not self.is_pyav:
				self._frame = self._frame + rate
				if self._frame <= RATE_MAX: self._frame = RATE_MAX
				if self._frame >= self._len - RATE_MAX: self._frame = self._len - RATE_MAX
				self.source.set(cv2.CAP_PROP_POS_FRAMES, self._frame)
				return self.source.read()[1] # Returns a tuple
			else:
				try:
					return next(self.stream).to_nd_array(format='bgr24')
				except StopIteration:
					return black
		else:
			return self.source.read()[1] # Returns a tuple
	
class CanvasProjector(app.Canvas):

	# Controller
	mid = Controller('nanoKONTROL2:nanoKONTROL2 MIDI 1 20:0')

	# pix2pix
	from models.pix2pixHD_model_custom import Pix2PixHDModel
	model = Pix2PixHDModel() 
	model.initialize()
	transform_list = []
	transform_list.append(transforms.ToTensor())
	# transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))) # Seems to be not necessary
	transform = transforms.Compose(transform_list)

	# Initialize arrays
	im_cap = black.copy()
	im_map = black.copy()
	im_map_right = np.zeros((H, W2, 3), dtype=np.uint8)
	im_lut = black.copy()
	im_lut_for_map = np.zeros((H, W2, 3), dtype=np.uint8)
	im_last = black.copy()

	# Draw function "callbacks"
	last_C1 = 0
	last_C2 = 0
	last_zoom = 0
	last_x = 0
	last_y = 0

	def __init__(self):
		app.Canvas.__init__(self, size=(W, H), keys='interactive')

		# Load sources
		self.reload_sources()

		# Options
		# (MIDI channel, min, max, random?, unform, callback, int?, sin?)

		self.OPT_CHANNEL = 0
		self.OPT_MIN = 1
		self.OPT_MAX = 2
		self.OPT_RANDOM = 3
		self.OPT_UNIFORM = 4
		self.OPT_CALLBACK = 5
		self.OPT_INT = 6
		self.OPT_SIN = 7

		self.opt = { \
		'CLOUDS':(7, 0.0, 2.0, 0, 'u_clouds', None, False, 0),\
		'GLITCH':(5, 0.0, 0.1, 69, 'u_glitch', None, False, 37),\
		'FEEDBACK':(4, 0.0, 0.99, 0, 'u_feedback', None, False, 0),\
		'MAPMIX':(2, 0.0, 1.0, 0, 'u_mapmix', None, False, 34),\
		'MOSAIC':(6, 0.0, 10.0, 70, 'u_mosaic', None, True, 38),\
		'LUT_C1':(18, 0.0, 255.0, 0, '', None, True, 0),\
		'LUT_C2':(19, 0.0, 255.0, 0, '', None, True, 0),\
		'SAT':(32, 1.0, 0.0, 0, '', None, True, 0),\
		'MUTE':(48, 0.0, 1.0, 0, '', None, True, 0),\
		'RATE':(17, -RATE_MAX, RATE_MAX, 0, '', None, True, 0),\
		'SOURCE':(1, 0.0, len(self.sources)-1, 65, '', None, True, 0),\
		'MASTER':(0, 0.0, 1.0, 0, 'u_master', None, False, 0),\
		'BORDER':(16, 0.0, 0.2, 0, 'u_border', None, False, 0),\
		'GENERATE':(41, 0.0, 1.0, 0, '', None, True, 0),\
		'REAL':(50, 0.0, 1.0, 0, '', None, True, 0),\
		'DIFF':(52, 0.0, 1.0, 0, '', None, True, 0),\
		'ZOOM':(22, 0.0, 1.4, 71, '', None, False, 0),\
		'X':(20, -1.0, 1.0, 71, '', None, False, 0),\
		'Y':(21, -1.0, 1.0, 71, '', None, False, 0),\
		'SPEED':(23, 0.5, 5.0, 0, '', None, False, 0),\
		'EXPOSURE':(3, 1.0, 2.0, 0, 'u_exposure', None, False, 0),\
		'RELOAD':(46, 0.0, 1.0, 0, '', self.reload_anfang, True, 0)}

		# Load shaders
		self.program = gloo.Program(vertex, fragment, count=4)
		
		# Static options
		self.program['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
		self.program['texcoord'] = [(1, 1), (1, 0), (0, 1), (0, 0)]
		self.program['u_resolution'] = [(W, H)]
		
		# Initialize textures
		self.program['t_video'] = black
		self.program['t_feedback'] = black
		self.program['t_map'] = black

		# Write option minimums
		for k, o in self.opt.items():
			if o[self.OPT_UNIFORM]: self.program[o[self.OPT_UNIFORM]] = o[self.OPT_MIN]
			# For CPU heavy options, just change when controller changes
			if o[self.OPT_CALLBACK]: self.mid.attach_callback(o[self.OPT_CHANNEL], o[self.OPT_CALLBACK])

		# Window/viewport
		width, height = self.physical_size
		gloo.set_viewport(0, 0, width, height)

		# Timing
		self.timer_app = app.Timer(1.0 / FPS, connect=self.on_timer_app, start=True)
		self.start_time = time.time()

		# Generate thread
		self.q_map = Queue(1)
		self.thread_gen = Thread(target=self.generate_map)
		self.thread_gen.start()

		self.q_lut = Queue(1)
		self.thread_lut = Thread(target=self.generate_lut)
		self.thread_lut.start()

		# Show
		self.show()

	def reload_sources(self):
		self.sources = []

		# Load webcams and videos
		self.sources.append(VideoSource(cv2.VideoCapture(0), is_video=False))
		self.sources.append(VideoSource(cv2.VideoCapture(1), is_video=False))
		self.sources.append(VideoSource(cv2.VideoCapture('debate.mp4')))
		self.sources.append(VideoSource(av.open('anfang.mp4'), is_pyav=True))
		self.sources.append(VideoSource(av.open('ende.mp4'), is_pyav=True))

	def reload_anfang(self):
		self.sources[3] = VideoSource(av.open('anfang.mp4'), is_pyav=True)
		self.sources[4] = VideoSource(av.open('ende.mp4'), is_pyav=True)

	def on_resize(self, event):
		width, height = event.physical_size
		gloo.set_viewport(0, 0, width, height)
		self.program['u_resolution'] = (event.physical_size[0], event.physical_size[1])

	def regenerate_lut(self):
		self.q_lut.put(self.im_cap_gray)
		

	def generate_lut(self):
		while(True):
			try:
				gray = self.q_lut.get()[:,LEFT_WH2_IN_WH:RIGHT_WH2_IN_WH]
			except Empty:
				pass
			else:
				# Lookup table to convert grayscale image to three-color image
				c1 = self.get_lerp('LUT_C1')
				c2 = self.get_lerp('LUT_C2')
				lutmap = np.zeros((256, 3), dtype=np.uint8)
				lutmap[:c1,:] = 0
				lutmap[c1:c2,:] = 255
				lutmap[c2:,0] = 255
				lutted = lutmap[gray]
				self.im_lut[:,W2:,:] = lutted
				self.im_lut_for_map = lutted

	def generate_map(self):
		while(True):
			map = np.zeros((H, W2, 3), dtype=np.uint8)
			try:
				map = self.q_map.get()
			except Empty:
				pass
			else:
				s0 = map
				s1 = np.zeros((NET, NET, 3), dtype=np.uint8) 
				s1[TOP_WH_IN_NET:BOTTOM_WH_IN_NET,LEFT_WH2_IN_NET:RIGHT_WH2_IN_NET,:] = s0
				s2 = self.transform(s1).unsqueeze(0).cuda() # -> GPU, FAST
				s3 = self.model.inference(s2) # SLOW
				s4 = util.tensor2im(s3.data[0]) # -> CPU, FAST
				map = s4[TOP_WH_IN_NET:BOTTOM_WH_IN_NET,LEFT_WH2_IN_NET:RIGHT_WH2_IN_NET,:]
				self.im_map_right = np.flipud(np.fliplr(map)) # Why does this need to be flipped?
			
	def get_lerp(self, id):
		o = self.opt[id]
		
		# If this is a random-enabled channel and the assigned R button is pressed
		if o[self.OPT_RANDOM] > 0 and self.mid.read(o[self.OPT_RANDOM]) > 0: 
			retval = np.random.uniform(o[self.OPT_MIN], self.mid.lerp(o[self.OPT_CHANNEL], o[self.OPT_MIN], o[self.OPT_MAX]))
		else: 
			retval = self.mid.lerp(o[self.OPT_CHANNEL], o[self.OPT_MIN], o[self.OPT_MAX])

		# If this is a sin-enabled channel and the assigned S button is pressed
		if o[self.OPT_SIN] > 0 and self.mid.read(o[self.OPT_SIN]) > 0: 
			factor = (math.sin(self.time) + 1) / 2
			retval *= factor
		
		# If this is an int-enabled channel
		if o[self.OPT_INT]:
			return int(round(retval, 0))
		else:
			return retval

	def on_draw(self, event):
		# Clear screen
		gloo.clear('black')
		
		frame = black

		if not self.get_lerp('MUTE'):
			# Read frame depending on controller status
			source = self.sources[self.get_lerp('SOURCE')]
			frame = source.read(self.get_lerp('RATE'))
			frame = np.flipud(frame) # Why does this need to be flipped?

		# Zoom
		z = self.get_lerp('ZOOM') # Temp. variable to use same random value for both axes
		x = self.get_lerp('X')
		y = self.get_lerp('Y')

		ox = (W_HD - W) * z
		oy = (H_HD - H) * z

		a = int(oy + (oy * y))
		b = int((H_HD-oy) + (oy * y))
		c = int(ox + (ox * x))
		d = int((W_HD-ox) + (ox * x))

		frame = frame[a:b,c:d,:]
		frame = cv2.resize(frame, (W, H))
		
		self.im_cap_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_rgbgray = cv2.cvtColor(self.im_cap_gray, cv2.COLOR_GRAY2RGB)
		
		# Difference image
		if self.get_lerp('DIFF'):
			# Monochrome
			if not self.get_lerp('SAT'):
				frame_diff = cv2.absdiff(self.im_last, frame_rgbgray)
				self.im_last = frame_rgbgray
				self.im_cap = frame_diff
			else:
				frame_diff = cv2.absdiff(self.im_last, frame_rgb)
				self.im_last = frame_rgb
				self.im_cap = frame_diff
		else:
			# Monochrome
			if not self.get_lerp('SAT'):
				self.im_cap = frame_rgbgray
			else:
				self.im_cap = frame_rgb

		# Generate when button is pressed and LUT, zoom, or pan is changed
		# BUG: This can NOT be done by callback functions in the MIDI handler, as the 
		# callbacks will be queued by mido and then successively released instead of
		# going bad
		
		self.new_C1 = self.get_lerp('LUT_C1')
		self.new_C2 = self.get_lerp('LUT_C2')
		self.new_zoom = self.get_lerp('ZOOM')
		self.new_x = self.get_lerp('X')
		self.new_y = self.get_lerp('Y')
		
		if self.new_C1 != self.last_C1: 
			self.regenerate_lut()
			self.last_C1 = self.new_C1
			if self.get_lerp('GENERATE'):
				self.q_map.put(self.im_lut_for_map)

		if self.new_C2 != self.last_C2: 
			self.regenerate_lut()
			self.last_C2 = self.new_C2
			if self.get_lerp('GENERATE'):
				self.q_map.put(self.im_lut_for_map)

		if self.new_zoom != self.last_zoom: 
			self.regenerate_lut()
			self.last_zoom = self.new_zoom
			if self.get_lerp('GENERATE'):
				self.q_map.put(self.im_lut_for_map)

		if self.new_x != self.last_x: 
			self.regenerate_lut()
			self.last_x = self.new_x
			if self.get_lerp('GENERATE'):
				self.q_map.put(self.im_lut_for_map)

		if self.new_y != self.last_y: 
			self.regenerate_lut()
			self.last_y = self.new_y
			if self.get_lerp('GENERATE'):
				self.q_map.put(self.im_lut_for_map)

		# Put map comparison map parts together
		self.im_map[:,:W2,:] = self.im_map_right
		self.im_map[:,W2:,:] = np.flipud(np.fliplr(self.im_cap[:,LEFT_WH2_IN_WH:RIGHT_WH2_IN_WH])) # Why does this need to be flipped?
			
		# Write options
		for k,o in self.opt.items():
			if o[4]: self.program[o[4]] = self.get_lerp(k)

		# Write textures
		self.program['u_time'] = (time.time() - self.start_time)
		self.program['t_video'][...] = self.im_cap
		self.program['t_map'][...] = self.im_map
		
		# Draw
		self.program.draw('triangle_strip')

		# Feedback
		# BUG: Needs to be "treated" once by openCV (-> CPU?) to not produce weird memory leaks
		# BUG: Weird scan lines in full screen mode when screen resolution > canvas resolution
		# BUG: "Flows" to bottom when activated before full screened/moved to second screen
		# WORKAROUND: Start feedback once before moving window, move/full screen while feedback runs
		screenshot = cv2.cvtColor(_screenshot(), cv2.COLOR_RGBA2RGB)
		self.program['t_feedback'][...] = np.fliplr(screenshot)

		# Communicate with control window
		control[0] = self.im_cap
		control[1] = self.im_lut
		control[2] = np.flipud(screenshot) # Why does this need to be flipped?
		control[3] = np.flipud(np.fliplr(self.im_map)) # Why does this need to be flipped?

		self.time = (time.time() * self.get_lerp('SPEED')) - self.start_time

	def on_timer_app(self, event):
		self.update()

class CanvasControl(app.Canvas):

	def __init__(self):
		app.Canvas.__init__(self, size=(W, H), keys='interactive')
		self.program = gloo.Program(vertex, fragment_c, count=4)
		width, height = self.physical_size
		gloo.set_viewport(0, 0, width, height)
		self.program['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
		self.program['texcoord'] = [(1, 1), (1, 0), (0, 1), (0, 0)]
		self.program['u_resolution'] = [(W, H)]
		self.program['t_1'] = control[0]
		self.program['t_2'] = control[1]
		self.program['t_3'] = control[2]
		self.program['t_4'] = control[3]
		self.timer_app = app.Timer(1.0 / FPS, connect=self.on_timer_app, start=True)
		self.show()

	def on_resize(self, event):
		width, height = event.physical_size
		gloo.set_viewport(0, 0, width, height)
		self.program['u_resolution'] = (event.physical_size[0], event.physical_size[1])

	def on_draw(self, event):
		gloo.clear('black')
		self.program['t_1'][...] = control[0]
		self.program['t_2'][...] = control[1]
		self.program['t_3'][...] = control[2]
		self.program['t_4'][...] = control[3]
		self.program.draw('triangle_strip')

	def on_timer_app(self, event):
		self.update()

canvas_projector = CanvasProjector()
canvas_control = CanvasControl()
app.run()