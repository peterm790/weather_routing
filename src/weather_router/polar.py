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
			df.columns = df.columns.astype(float)
			df.index = df.index.astype(float)
			max_tws = int(max(df.columns)) if len(df.columns) else 0
		else:
			max_tws = int(max(self.tws)) if len(self.tws) else 0
		new_tws = list(range(0, max_tws + 1))
		new_twa = list(range(0,185,5))

		df_new = pd.DataFrame(np.full((len(new_twa), len(new_tws)), np.nan))
		df_new.columns = new_tws
		df_new.columns = df_new.columns.astype(float)
		df_new.index = new_twa
		df_new.index = df_new.index.astype(float)

		df = df.combine_first(df_new).sort_index(axis=0).sort_index(axis=1)
		df = df.interpolate(axis = 0).interpolate(axis = 1).fillna(0)

		self.tws = list(df.columns)
		self.twa = list(df.index)
		self.speedTable = df.to_numpy()
		# Precompute fast index maps
		self._twa_step = 5.0
		self._twa_min = float(self.twa[0]) if len(self.twa) else 0.0
		self._twa_max = float(self.twa[-1]) if len(self.twa) else 180.0
		self._tws_min = float(self.tws[0]) if len(self.tws) else 0.0
		self._tws_max = float(self.tws[-1]) if len(self.tws) else 0.0
		self._twa_to_index = {float(v): i for i, v in enumerate(self.twa)}
		self._tws_to_index = {float(v): i for i, v in enumerate(self.tws)}

	def myround_twa(self, x, base=5):
		x = base * round(x/base)
		return x

	def myround_tws(self, x, base=1):
		x = base * round(x/base)
		return x

	def getSpeed (self, tws, twa):
		# return speed in knots given boat wind speed (knots) and twa (0-180) deg
		# Snap to grid and clamp to known range (upper clamp for tws mirrors original fallback)
		rtwa = int(round(float(twa) / self._twa_step)) * int(self._twa_step)
		if rtwa < self._twa_min:
			rtwa = self._twa_min
		if rtwa > self._twa_max:
			rtwa = self._twa_max
		rtwa = float(rtwa)
		twa_idx = self._twa_to_index.get(rtwa)
		if twa_idx is None:
			raise ValueError(f"TWA {twa} (rounded {rtwa}) not in polar index")

		rtws = int(round(float(tws)))
		if rtws > self._tws_max:
			rtws = int(self._tws_max)
		rtws = float(rtws)
		tws_idx = self._tws_to_index.get(rtws)
		if tws_idx is None:
			# Match previous behavior: only defined fallback was for > max
			# Other invalid inputs should raise
			raise ValueError(f"TWS {tws} (rounded {rtws}) not in polar index")

		return self.speedTable[twa_idx][tws_idx]
