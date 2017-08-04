import sys, math
from operator import itemgetter
from math import cos, sin
from collections import namedtuple

"""
Point indices that define lines that build the skeleton
"""
EDGES = {"center" : [{1, 2}, {2, 21}, {3, 21}, {3, 4}],
		 "left" : [{1, 17}, {17, 18}, {18, 19}, {19, 20}, {21, 9}, {9, 10}, {10, 11}, {11, 12}, {12, 25}, {24, 25}],
		 "right" : [{21, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 23}, {23, 22}, {1, 13}, {13, 14}, {14, 15}, {15, 16}]}

"""
NamedTuples used to store the information read from .skeleton file
"""
BodyInfo = namedtuple('BodyInfo', 'bodyID clippedEdges handLeftConfidence handLeftState \
									handRightConfidence handRightState isRestricted \
									leanX leanY trackingState joints')
Joint = namedtuple('Joint', 'x y z depthX depthY colorX colorY orientationW orientationX orientationY orientationZ \
							 trackingState')

"""
RGB values for colors
"""
WHITE, RED = (255, 255, 255), (178, 34, 34)
YELLOW = (255, 215, 0)
GREEN = (0, 100, 0)
BLUE = (30, 144, 255)

"""
The incremental step used for the rotation angle (in radians)
"""
counter_clockwise = 0.1
clockwise = -counter_clockwise

"""
Constants used to make the projection
"""
viewer_distance = 4
win_width = 950
win_height = 850
fov = 2048

def rotation_matrix(α, β, γ):
	"""
	Rotation matrix of α, β, γ radians around x, y, z axes (respectively)
	"""
	sα, cα = sin(α), cos(α)
	sβ, cβ = sin(β), cos(β)
	sγ, cγ = sin(γ), cos(γ)
	return (
		(cβ*cγ, -cβ*sγ, sβ),
		(cα*sγ + sα*sβ*cγ, cα*cγ - sγ*sα*sβ, -cβ*sα),
		(sγ*sα - cα*sβ*cγ, cα*sγ*sβ + sα*cγ, cα*cβ)
	)

def project(point):
	"""
	Transforms this 3D point to 2D using a perspective projection.
	"""
	(x, y, z) = point
	factor = fov / (viewer_distance + z)
	x = x * factor + win_width / 2
	y = -y * factor + win_height / 2
	return (x, y, z)