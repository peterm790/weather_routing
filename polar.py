from typing import Tuple
from io import FileIO
import math


from typing import Tuple
from io import FileIO
import math

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
			self.twa.append (math.radians (twa))
			speedline = []
			for i in range (1,len (data)):
				speed = float (data[i])
				speedline.append (speed)
			self.speedTable.append (speedline)
			line = f.readline ()
		f.close ()

	def getSpeed (self, tws: float, twa: float) -> float:
		""" Returns the speed (in knots) given tws (in knots) and twa (in radians) """

		tws1 = 0
		tws2 = 0

		for k in range(0, len(self.tws)):
			if tws >= self.tws[k]:
				tws1 = k
		for k in range(len(self.tws) - 1, 0, -1):
			if tws <= self.tws[k]:
				tws2 = k
		if tws1 > tws2: # TWS over table limits
			tws2 = len(self.tws) - 1
		twa1 = 0
		twa2 = 0
		for k in range(0, len(self.twa)):
			if twa >= self.twa[k]:
				twa1 = k
		for k in range(len(self.twa) - 1, 0, -1):
			if twa <= self.twa[k]:
				twa2 = k

		speed1 = self.speedTable[twa1][tws1]
		speed2 = self.speedTable[twa2][tws1]
		speed3 = self.speedTable[twa1][tws2]
		speed4 = self.speedTable[twa2][tws2]

		if twa1 != twa2:
			speed12 = speed1 + (speed2 - speed1) * (twa-self.twa[twa1]) / (self.twa[twa2] - self.twa[twa1])#interpolate twa
			speed34 = speed3 + (speed4 - speed3) * (twa-self.twa[twa1]) / (self.twa[twa2] - self.twa[twa1])#interpo;ate twa
		else:
			speed12 = speed1
			speed34 = speed3
		if tws1 != tws2:
			speed = speed12 + (speed34 - speed12) * (tws - self.tws[tws1]) / (self.tws[tws2] - self.tws[tws1])
		else:
			speed = speed12
		return speed
