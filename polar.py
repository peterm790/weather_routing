from typing import Tuple
from io import FileIO
import math
import numpy as np



class Polar:
	def __init__ (self, polarPath: str, f: FileIO = None):
		"""
		Parameters
		----------
		polarPath : string
			Path of the polar file
		f : File
			File object for passing an opened file
		"""
		self.tws = []
		self.twa = []
		self.vmgdict = {}
		self.speedTable = []

		if f is None:
			f = open (polarPath, "r")

		tws = f.readline ().split ()
		for i in range (1,len(tws)):
			self.tws.append (float (tws[i].replace ('\x02', '')))

		line = f.readline ()
		while line != "":
			data = line.split ()
			twa = float (data[0])
			self.twa.append(twa)
			speedline = []
			for i in range (1,len (data)):
				speed = float(data[i])
				speedline.append(speed)
			self.speedTable.append(speedline)
			line = f.readline ()
		f.close ()

	def myround(self, x, base=5):
		x = base * round(x/base)
		return x

	def getSpeed (self, tws, twa):
		# return speed in knots give boat speed (knots) and twa (0-180) deg
		twa_idx = self.twa.index(self.myround(twa))
		tws_idx = self.tws.index(self.myround(tws))
		return self.speedTable[twa_idx][tws_idx]