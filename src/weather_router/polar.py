from io import FileIO
import numpy as np
import pandas as pd



class Polar:
	def __init__ (self, polarPath = None, f = None, df = None):
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
		self.speedTable = []

		if isinstance(df, pd.DataFrame):
			pass
		else:
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
			df = pd.DataFrame(np.array(self.speedTable))
			df.columns = self.tws
			df.index = self.twa

		if isinstance(df, pd.DataFrame):
			new_tws = list(range(0,df.columns[-1]))
		else:
			new_tws = list(range(0,int(self.tws[-1])))
		new_twa = list(range(0,185,5))

		df_new = pd.DataFrame(np.full((len(new_twa), len(new_tws)), np.nan))
		df_new.columns = new_tws
		df_new.columns = df_new.columns.astype(float)
		df_new.index = new_twa
		df_new.index = df_new.index.astype(float)

		df = df.combine_first(df_new)
		df = df.interpolate(axis = 0).interpolate(axis = 1).fillna(0)

		self.tws = list(df.columns)
		self.twa = list(df.index)
		self.speedTable = df.to_numpy()

	def myround_twa(self, x, base=5):
		x = base * round(x/base)
		return x

	def myround_tws(self, x, base=1):
		x = base * round(x/base)
		return x

	def getSpeed (self, tws, twa):
		# return speed in knots give boat speed (knots) and twa (0-180) deg
		try:
			twa_idx = self.twa.index(self.myround_twa(twa))
			tws_idx = self.tws.index(self.myround_tws(tws))
			speed = self.speedTable[twa_idx][tws_idx]
		except:
			if self.myround_tws(tws) > self.tws[-1]:
				tws = self.tws[-1]
				twa_idx = self.twa.index(self.myround_twa(twa))
				tws_idx = self.tws.index(self.myround_tws(tws))
				speed = self.speedTable[twa_idx][tws_idx]
        return speed
