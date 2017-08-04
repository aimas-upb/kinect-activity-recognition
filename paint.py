from utils import *
import pygame
import numpy as np

class Paint:
	def __init__(self, skeletons, rotation, count, action, camera, noOfFrames):
		"""
		Class used to draw graphic components
		Parameters
			skeletons - The coordinates of the joints
			rotation - Angle of rotation values for each axis
			count - Current frame number
			action - The title of the action class
			camera - The camera ID
			noOfFrames - Total number of frames
		"""
		self.skeletons = skeletons
		self.rotation = rotation
		self.points = {}
		self.bodyCount = len(skeletons)
		self.screen = pygame.display.set_mode([950, 850])
		self.clock = pygame.time.Clock()
		self.initial_skeletons = np.array(self.skeletons)
		#The center of gravity is calculated
		centers = []
		for i in range(self.bodyCount):
			centers.append(np.sum(self.skeletons[i], axis=0) / len(self.skeletons[i]))
		self.center = np.sum(centers, axis=0) / len(centers)
		self.project_center = project(self.center)
		#Rotation and projection are performed
		for i in range(self.bodyCount):
			self.skeletons[i] -= self.center
			self.skeletons[i] = self.skeletons[i].dot(rotation_matrix(*rotation))
			self.skeletons[i] += self.center
			for j in range(len(self.skeletons[i])):
				self.skeletons[i][j] = project(self.skeletons[i][j])
		self.centers = centers
		self.count = count
		self.myfont = pygame.font.SysFont("monospace", 15)
		self.actionLabel = self.myfont.render("Action: " + action, 2, (0, 0, 205))
		self.cameraLabel = self.myfont.render("Camera: " + str(camera), 2, (0, 0, 205))
		self.frameLabel = self.myfont.render("Frame Number: " + str(self.count + 1) + "/" + str(noOfFrames), 2, (0, 0, 205))
		self.leftLabel = self.myfont.render("Left", 2, GREEN)
		self.rightLabel = self.myfont.render("Right", 2, RED)
		self.mainloop()

	def mainloop(self):
		"""
		The main loop executed for each frame
		"""
		done = False
		while not done:
			done = self.handleEvents()
			self.screen.fill(WHITE)
			self.screen.blit(self.actionLabel, (50, 20))
			self.screen.blit(self.cameraLabel, (50, 40))
			self.screen.blit(self.frameLabel, (50, 60))
			self.screen.blit(self.leftLabel, (50, 80))
			self.screen.blit(self.rightLabel, (50, 100))
			self.drawShape()
			pygame.display.flip()
			self.clock.tick(60)
	
	def drawShape(self, thickness = 3):
		"""
		Drawing of graphic components
		Parameters
			thickness - value of thicness for line
		"""
		for i in range(self.bodyCount):
			for (start, end) in EDGES["center"]:
				pygame.draw.line(self.screen, YELLOW, self.skeletons[i][start-1][:2], self.skeletons[i][end-1][:2], thickness)
			for (start, end) in EDGES["left"]:
				pygame.draw.line(self.screen, GREEN, self.skeletons[i][start-1][:2], self.skeletons[i][end-1][:2], thickness)
			for (start, end) in EDGES["right"]:
				pygame.draw.line(self.screen, RED, self.skeletons[i][start-1][:2], self.skeletons[i][end-1][:2], thickness)
			for point in self.skeletons[i]:
				(x, y) = point[:2]
				pygame.draw.circle(self.screen, BLUE, (int(x), int(y)), 5, 0)
			pygame.draw.circle(self.screen, RED, (int(self.project_center[0]), int(self.project_center[1])), 7, 0)

	def handleEvents(self):
		"""
		Handle events that may occur
		"""
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
			if event.type == pygame.KEYDOWN:
				key = event.key
				if key == pygame.K_q:
					self.rotation[0] += counter_clockwise
				elif key == pygame.K_w:
					self.rotation[0] += clockwise
				elif key == pygame.K_a:
					self.rotation[1] += counter_clockwise
				elif key == pygame.K_s:
					self.rotation[1] += clockwise
				elif key == pygame.K_z:
					self.rotation[2] += counter_clockwise
				elif key == pygame.K_x:
					self.rotation[2] += clockwise
				elif key == pygame.K_SPACE:
					self.count += 1
					return True
				elif key == pygame.K_r:
					self.rotation = [0, 0, 0]
				elif key == pygame.K_BACKSPACE:
					self.count -= 1
					return True
				elif key == pygame.K_ESCAPE:
					pygame.quit()
				self.skeletons = np.array(self.initial_skeletons)
				for i in range(self.bodyCount):
					self.skeletons[i] -= self.center
					self.skeletons[i] = self.skeletons[i].dot(rotation_matrix(*self.rotation))
					self.skeletons[i] += self.center
					for j in range(len(self.skeletons[i])):
						self.skeletons[i][j] = project(self.skeletons[i][j])
		return False
