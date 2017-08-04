from utils import *

class Frame:
	def __init__(self, f):
		"""
		Class used to store information about a frame
		Parameters
			f - input file
		"""
		#No. of observed skeletons in current frame
		self.bodycount = int(f.readline())
		#To store multiple skeletons per frame
		self.bodies = []
		for idx in range(self.bodycount):
			items = f.readline().split(" ")
			#Read 7 integers (bodyID, clippedEdges, handLeftConfidence, handLeftState, 
			#				  handRightConfidence, handRightState, isRestricted)
			args = list(map(int, items[:7]))
			#Read 2 floats (leanX, leanY)
			args.append(float(items[7]))
			args.append(float(items[8]))
			#Read trackingState
			args.append(int(items[9]))
			joints = []
			#No. of joints (25)
			jointCount = int(f.readline())
			for i in range(jointCount):
				jointInfo = f.readline().split(" ")
				#Read 11 floats
				#1-3 -> (x, y, z) - 3D location of the joint i
				#4-5 -> (depthX, depthY) - 2D location of the joint i in corresponding depth IR/frame
				#6-7 -> (colorX, colorY) - 2D location of the joint i in corresponding RGB frame
				#8-11 -> (orientationW, orientationX, orientationY, orientationZ) - The orientation of the joint i
				jointArgs = list(map(float, jointInfo[:11]))
				#Read the tracking state of the joint i
				jointArgs.append(int(jointInfo[11]))
				joint = Joint(*jointArgs)
				joints.append(joint)
			args.append(joints)
			self.bodies.append(BodyInfo(*args))
