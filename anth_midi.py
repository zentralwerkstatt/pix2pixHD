# pip install python-rtmidi
# pip install mido
import mido
import numpy as np
from collections import defaultdict
from threading import Thread
from queue import Queue
from queue import Empty

class Controller:
	
	state = defaultdict(int)
	callbacks = dict()
	
	def __init__(self, src):
		self.q = Queue(1)
		self.thread = Thread(target=self.do_callback)
		self.thread.start()
		try:
			self.midi = mido.open_ioport(src, callback=self.callback)
		except:
			print('Can not open MIDI device, try one of these:')
			print(mido. get_output_names())
	
	def read(self, channel):
		return self.state[channel]
	
	def lerp(self, channel, low, high):
		return np.interp(self.state[channel], [0,127], [low,high])

	def callback(self, msg):
		if msg.control in self.callbacks:
			self.q.put(self.callbacks[msg.control])
		self.state[msg.control] = msg.value

	def attach_callback(self, channel, func):
		# Dictionary of functions - <3 Python!
		self.callbacks[channel] = func

	def do_callback(self):
		while(True):
			try:
				func = self.q.get()
			except Empty:
				pass
			else:
				func()