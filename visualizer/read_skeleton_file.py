from paint import *
from time import sleep
import numpy as np
import re
import sys, getopt
from frame import *
from sample import *
import argparse

def getCoordinates(bodyInfo):
	"""
	Extracts the coordinates of the joints
	Parameters
		bodyInfo - Body namedtuple with full information
	"""
	vertices = []
	for joint in bodyInfo.joints:
		vertices.append(np.array([joint.x, joint.y, joint.z]))
	return np.array(vertices)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', action='store', dest='sample_filename', help='Store a sample filename')
	parser.add_argument('-a', action='store', dest='actions_filename', help='Store name for actions file')
	results = parser.parse_args()
	if not (results.actions_filename and results.sample_filename):
		print("usage: python3 read_skeleton_file.py [-h] [-s SAMPLE_FILENAME] [-a ACTIONS_FILENAME]")
		exit(1)
	actions = []
	with open(results.actions_filename) as f:
		actions = [line.rstrip() for line in f.readlines()]
	sample = Sample(results.sample_filename)
	pygame.init()
	pygame.display.set_caption(sample.name + " - Visualization")
	rotation = [0, 0, 0]
	count = 0
	while True:
		if count < 0:
			count = 0
		elif count >= len(sample.frames):
			count = len(sample.frames) - 1
		frame = sample.frames[count]
		skeletons = []
		for body in frame.bodies:
			vertices = getCoordinates(body)
			skeletons.append(vertices)
		paint = Paint(np.array(skeletons), rotation, count, actions[sample.classLabel-1], sample.cameraID, len(sample.frames))
		rotation = paint.rotation
		count = paint.count
	sleep(2)
