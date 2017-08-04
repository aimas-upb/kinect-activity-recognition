from frame import *

class Sample:
	def __init__(self, filename):
		"""
		Reads an .skeleton file from "NTU RGB+D 3D Action Recognition Dataset"
		Parameters
			filename - full adress and filename of the .skeleton file 
		"""
		for word in filename.split("/"):
			if word.endswith(".skeleton"):
				self.name = word.split(".")[0]
				self.classLabel = int(self.name.split("A")[1])
				self.cameraID = int(self.name.split("C")[1][:3])
		self.readFile(filename)

	def readFile(self, filename):
		"""
		Parameters
			filename - full adress and filename of the .skeleton file
		"""
		#To store multiple frame
		self.frames = []
		with open(filename) as f:
			line = f.readline()
			#No. of the recorded frames
			self.framecount = int(line)
			for idx in range(self.framecount):
				self.frames.append(Frame(f))