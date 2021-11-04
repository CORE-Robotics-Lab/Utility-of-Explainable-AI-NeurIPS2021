# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

from __future__ import absolute_import
import pickle
import json
import xml.etree.ElementTree
from MalmoPython import AgentHost, ClientPool, ClientInfo, MissionSpec, MissionRecordSpec
from collections import Sequence
import os
from time import sleep
import numpy as np
import six
from PIL import Image
from numpy import zeros, log
from time import time

from malmopy.environment.malmo.malmo import VideoCapableEnvironment, StateBuilder
from math import acos
from math import sqrt
from math import pi


def save_pickle(file_location, file, special_string, want_to_print=False):
	"""
	stores information into a pickle file
	:param file_location: location to store
	:param file: data
	:param special_string: string to save the data with
	:param want_to_print: want to print a little debug statement
	:return:
	"""
	pickle.dump(file, open(os.path.join(file_location, special_string), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	if want_to_print:
		print("Dumped", file, "into ", file_location, "safely!")


def length(v):
	"""
	Calculates the length of a vector.

	Params:

	v(vector): the vector to calculate the length of

	Returns:

	float: the length of the vector
	"""
	return sqrt(v[0] ** 2 + v[1] ** 2)


def dot_product(v, w):
	"""
	Calculates the dot product of two vectors.

	Params:

	v(vector)
	w(vector)

	Returns:

	float: the dot product of v and W
	"""
	return v[0] * w[0] + v[1] * w[1]


def determinant(v, w):
	"""
	Calculates the determinant of two vectors.

	Params:

	v(vector)
	w(vector)

	Returns:

	float: the determinant of the matrix created from v and w
	"""
	return v[0] * w[1] - v[1] * w[0]


def inner_angle(v, w):
	"""
	Calculates the inner angle between two vectors.

	Params:

	v(vector)
	w(vector)

	Returns:

	float: the inner angle in degrees between two vectors
	"""
	cosx = dot_product(v, w) / (length(v) * length(w))
	rad = acos(cosx)  # in radians
	return rad * 180 / pi  # returns degrees


def angle_clockwise(A, B):
	"""
	Calculates the clockwise angle between two vectors.

	Params:

	A(vector)
	B(vector)

	Returns:
	float: the clockwise angle between A and B
	"""

	inner = inner_angle(A, B)
	det = determinant(A, B)
	if det < 0:  # this is a property of the det. If the det < 0 then B is clockwise of A
		return inner
	else:  # if the det > 0 then A is immediately clockwise of B
		return 360 - inner


def allocate_remotes(remotes):
	"""
	Utility method for building a Malmo ClientPool.
	Using this method allows sharing the same ClientPool across
	mutiple experiment
	:param remotes: tuple or array of tuples. Each tuple can be (), (ip,), (ip, port)
	:return: Malmo ClientPool with all registered clients
	"""
	if not isinstance(remotes, list):
		remotes = [remotes]

	pool = ClientPool()
	for remote in remotes:
		if isinstance(remote, ClientInfo):
			pool.add(remote)
		elif isinstance(remote, Sequence):
			if len(remote) == 0:
				pool.add(ClientInfo('localhost', 10000))
			elif len(remote) == 1:
				pool.add(ClientInfo(remote[0], 10000))
			else:
				pool.add(ClientInfo(remote[0], int(remote[1])))
	return pool


class TurnState(object):
	def __init__(self):
		self._turn_key = None
		self._has_played = False

	def update(self, key):
		self._has_played = False
		self._turn_key = key

	@property
	def can_play(self):
		return self._turn_key is not None and not self._has_played

	@property
	def key(self):
		return self._turn_key

	@property
	def has_played(self):
		return self._has_played

	@has_played.setter
	def has_played(self, value):
		self._has_played = bool(value)


class MalmoStateBuilder(StateBuilder):
	"""
	Base class for specific state builder inside the Malmo platform.
	The #build method has access to the currently running environment and all
	the properties exposed to this one.
	"""

	def __call__(self, *args, **kwargs):
		assert isinstance(args[0], MalmoEnvironment), 'provided argument should inherit from MalmoEnvironment'
		return self.build(*args)


class MalmoRGBStateBuilder(MalmoStateBuilder):
	"""
	Generate RGB frame state resizing to the specified width/height and depth
	"""

	def __init__(self, width, height, grayscale):
		assert width > 0, 'width should be > 0'
		assert width > 0, 'height should be > 0'

		self._width = width
		self._height = height
		self._gray = bool(grayscale)

	def build(self, environment):
		import numpy as np

		img = environment.frame

		if img is not None:
			img = img.resize((self._width, self._height))

			if self._gray:
				img = img.convert('L')
			return np.array(img)
		else:
			return zeros((self._width, self._height, 1 if self._gray else 3)).squeeze()


class MalmoALEStateBuilder(MalmoRGBStateBuilder):
	"""
	Commodity class for generating Atari Learning Environment compatible states.

	Properties:
		- depth: Grayscale image
		- width: 84
		- height: 84

	return (84, 84) numpy array
	"""

	def __init__(self):
		super(MalmoALEStateBuilder, self).__init__(84, 84, True)


class MalmoEnvironment(VideoCapableEnvironment):
	"""
	Interaction with Minecraft through the Malmo Mod.
	"""

	MAX_START_MISSION_RETRY = 50

	def __init__(self, mission, actions, remotes,
				 role=0, exp_name="", turn_based=False,
				 recording_path=None, force_world_reset=False):

		assert isinstance(mission, six.string_types), "mission should be a string"
		super(MalmoEnvironment, self).__init__()

		self._agent = AgentHost()
		self._mission = MissionSpec(mission, True)

		# validate actions
		self._actions = actions
		assert actions is not None, "actions cannot be None"
		assert isinstance(actions, Sequence), "actions should be an iterable object"
		assert len(actions) > 0, "len(actions) should be > 0"

		# set up recording if requested
		if recording_path:
			self._recorder = MissionRecordSpec(recording_path)
			self._recorder.recordCommands()
			self._recorder.recordMP4(12, 400000)
			self._recorder.recordRewards()
			self._recorder.recordObservations()
		else:
			self._recorder = MissionRecordSpec()

		self._clients = allocate_remotes(remotes)

		self._force_world_reset = force_world_reset
		self._role = role
		self._exp_name = exp_name
		self._turn_based = bool(turn_based)
		self._turn = TurnState()

		self._world = None
		self._world_obs = None
		self._previous_action = None
		self._last_frame = None
		self._action_count = None
		self._end_result = None

		self.chest_state = {'log': 0, 'wooden_axe': 0, 'planks': 0, 'wooden_pickaxe': 0, 'cobblestone': 0, 'stick': 0, 'stone': 0}
		self.crafting_table_location = [35.5, 41.5]
		self.chest_location = [36.5, 41.5]
		self.inventory_axes = []
		self.inventory_pickaxes = []

		self.no_axe_cutting_time = 5
		self.axe_cutting_time = 2.8
		self.craft_delay = 6.5
		self.store_resources_delay = 10.5

		self.participant_number = None
		# if you cannot see chest, when you are transferring use these numbers
		self.chest_unknown_birch_wood_quant = 0
		self.chest_unknown_jungle_wood_quant = 0
		self.chest_unknown_birch_plank_quant = 0
		self.chest_unknown_jungle_plank_quant = 0
		self.chest_unknown_stick_quant = 0
		self.chest_unknown_cobblestone_quant = 0
		self.chest_unknown_stone_quant = 0
		self.birch = [[17, 16], [13, 20], [10, 10], [20, 7], [42, 41],
					  [15, 9], [13, 5], [16, 22], [9, 37], [32, 18], [35, 12], [26, 22], [26, 6],
					  [23, 25], [21, 25], [30, 10], [25, 14]]
		self.jungle = [[16, 24], [36, 15], [12, 21], [6, 14], [4, 12], [35, 48], [16, 36], [4, 30], [17, 46], [11, 52],
					   [6, 50], [9, 44], [21, 32], [17, 40], [15, 44], [13, 30], [10, 34],
					   [4, 46], [19, 38], [5, 28], [11, 25], [36, 45], [40, 19]]
		self.stone = np.array([[54, 32],
       [54, 34],
       [51, 35],
       [58, 37],
       [50, 38],
       [53, 38],
       [54, 40],
       [56, 42],
       [58, 45],
       [50, 47],
       [58, 48],
       [54, 49],
       [51, 53]])
		self.cobblestone = np.array([[49,  6],
       [55,  7],
       [51,  8],
       [54, 12],
       [54, 14],
       [58, 16],
       [58, 18],
       [57, 20],
       [54, 22],
       [56, 23],
       [53, 26],
       [51, 28],
       [54, 28],
       [58, 31]])

	@property
	def available_actions(self):
		return len(self._actions)

	@property
	def state(self):
		raise NotImplementedError()

	@property
	def end_result(self):
		return self._end_result

	@property
	def reward(self):
		return 0.

	@property
	def done(self):
		latest_ws = self._agent.peekWorldState()
		return latest_ws.has_mission_begun and not latest_ws.is_mission_running

	@property
	def action_count(self):
		return self._action_count

	@property
	def previous_action(self):
		return self._previous_action

	@property
	def frame(self):
		latest_ws = self._agent.peekWorldState()

		if hasattr(latest_ws, 'video_frames') and len(latest_ws.video_frames) > 0:
			self._last_frame = latest_ws.video_frames[-1]

		return Image.frombytes('RGB',
							   (self._last_frame.width, self._last_frame.height),
							   bytes(self._last_frame.pixels))

	@property
	def recording(self):
		return super(MalmoEnvironment, self).recording

	@recording.setter
	def recording(self, val):
		self._recording = bool(val)

		if self.recording:
			if not self._mission.isVideoRequested(0):
				self._mission.requestVideo(212, 160)

	@property
	def is_turn_based(self):
		return self._turn_based

	@property
	def world_observations(self):
		latest_ws = self._agent.peekWorldState()
		if latest_ws.number_of_observations_since_last_state > 0:
			self._world_obs = json.loads(latest_ws.observations[-1].text)

		return self._world_obs

	def _ready_to_act(self, world_state):
		if not self._turn_based:
			return True
		else:
			if not world_state.is_mission_running:
				return False

			if world_state.number_of_observations_since_last_state > 0:
				data = json.loads(world_state.observations[-1].text)
				turn_key = data.get(u'turn_key', None)

				if turn_key is not None and turn_key != self._turn.key:
					self._turn.update(turn_key)
			return self._turn.can_play

	def get_robot_info(self):
		for e, i in enumerate(self.world_observations['entities']):

			if i['name'] == 'Robot':
				robot_info = i
				robot_index = e
				return robot_info, robot_index

	def get_euclidean_distance_to_all_trees(self, single_point):
		# from https://stackoverflow.com/questions/4370975/python-numpy-euclidean-distance-calculation-between-matrices-of-row-vectors/38976555

		trees = np.array(self.per_item_dict['log'])

		dist = (trees - single_point) ** 2
		dist = np.sum(dist, axis=1)
		dist = np.sqrt(dist)
		return dist

	def get_robot_inventory(self):
		"""
		Updates the robot's inventory

		Params:

		Returns:
		"""
		robot_resources = {}
		inventory_of_robot = {'InventorySlot_0_size': 'InventorySlot_0_item',
							  'InventorySlot_1_size': 'InventorySlot_1_item',
							  'InventorySlot_2_size': 'InventorySlot_2_item',
							  'InventorySlot_3_size': 'InventorySlot_3_item',
							  'InventorySlot_4_size': 'InventorySlot_4_item',
							  'InventorySlot_5_size': 'InventorySlot_5_item',
							  'InventorySlot_6_size': 'InventorySlot_6_item',
							  'InventorySlot_7_size': 'InventorySlot_7_item',
							  'InventorySlot_8_size': 'InventorySlot_8_item',
							  'InventorySlot_9_size': 'InventorySlot_9_item',
							  'InventorySlot_10_size': 'InventorySlot_10_item',
							  'InventorySlot_11_size': 'InventorySlot_11_item',
							  'InventorySlot_12_size': 'InventorySlot_12_item',
							  'InventorySlot_13_size': 'InventorySlot_13_item',
							  'InventorySlot_14_size': 'InventorySlot_14_item',
							  'InventorySlot_15_size': 'InventorySlot_15_item',
							  'InventorySlot_16_size': 'InventorySlot_16_item',
							  'InventorySlot_17_size': 'InventorySlot_17_item',
							  'InventorySlot_18_size': 'InventorySlot_18_item',
							  'InventorySlot_19_size': 'InventorySlot_19_item',
							  'InventorySlot_20_size': 'InventorySlot_20_item',
							  'InventorySlot_21_size': 'InventorySlot_21_item',
							  'InventorySlot_22_size': 'InventorySlot_22_item',
							  'InventorySlot_23_size': 'InventorySlot_23_item',
							  'InventorySlot_24_size': 'InventorySlot_24_item',
							  'InventorySlot_25_size': 'InventorySlot_25_item',
							  'InventorySlot_26_size': 'InventorySlot_26_item',
							  'InventorySlot_27_size': 'InventorySlot_27_item',
							  'InventorySlot_28_size': 'InventorySlot_28_item',
							  'InventorySlot_29_size': 'InventorySlot_29_item',
							  'InventorySlot_30_size': 'InventorySlot_30_item',
							  'InventorySlot_31_size': 'InventorySlot_31_item',
							  'InventorySlot_32_size': 'InventorySlot_32_item',
							  'InventorySlot_33_size': 'InventorySlot_33_item',
							  'InventorySlot_34_size': 'InventorySlot_34_item',
							  'InventorySlot_35_size': 'InventorySlot_35_item',
							  'InventorySlot_36_size': 'InventorySlot_36_item',
							  'InventorySlot_37_size': 'InventorySlot_37_item',
							  'InventorySlot_38_size': 'InventorySlot_38_item',
							  'InventorySlot_39_size': 'InventorySlot_39_item',
							  'InventorySlot_40_size': 'InventorySlot_40_item'}
		for i in inventory_of_robot.keys():
			if self.world_observations[i] != 0:
				if self.world_observations[inventory_of_robot[i]] not in list(robot_resources.keys()):
					robot_resources[self.world_observations[inventory_of_robot[i]]] = self.world_observations[i]
				else:
					robot_resources[self.world_observations[inventory_of_robot[i]]] += self.world_observations[i]
		self.robot_inventory = robot_resources

	def does_robot_have_axe(self):
		"""
		Returns whether the robot currently has an axe.

		Params:

		Returns:
		bool: whether or not the robot has an axe.
		"""
		if 'wooden_axe' in list(self.robot_inventory.keys()):
			return True
		else:
			return False

	def does_robot_have_an_axe_already(self):
		"""
		Returns whether the robot currently has an axe.

		Params:

		Returns:
		bool: whether or not the robot has an axe.
		"""
		i = 0
		self.get_robot_inventory()
		if self.robot_inventory['wooden_axe'] > 1:
			return True
		else:
			return False

	def does_robot_have_an_pickaxe_already(self):
		"""
		Returns whether the robots currently has a pickaxe.

		Params:

		Returns:
		bool: whether or not the robot has a pickaxe.
		"""
		i = 0
		self.get_robot_inventory()
		if self.robot_inventory['wooden_pickaxe'] > 1:
			return True
		else:
			return False

	def does_robot_have_one_pickaxe(self):
		i = 0
		self.get_robot_inventory()
		if 'wooden_pickaxe' not in self.robot_inventory.keys():
			return False
		else:
			return True

	def does_robot_have_one_axe(self):
		i = 0
		self.get_robot_inventory()
		if 'wooden_axe' not in self.robot_inventory.keys():
			return False
		else:
			return True

	def wield_axe(self):
		"""
		Makes the robot wield an axe.

		Params:

		Returns:
		"""
		if self.does_robot_have_axe():
			inventory_num = self.get_inventory_number_of_axe()
			for i in range(8):
				self._agent.sendCommand("hotbar." + str(i) + " 0")
			self._agent.sendCommand("hotbar." + str(np.random.randint(8)) + " 0")
			self._agent.sendCommand("hotbar." + str(inventory_num + 1) + " 1")
		else:
			for i in range(8):
				self._agent.sendCommand("hotbar." + str(i) + " 0")
			self._agent.sendCommand("hotbar." + str(np.random.randint(8)) + " 1")

	# self._agent.sendCommand('use 0')

	def does_robot_have_pickaxe(self):
		"""
		Returns whether the robots currently has a pickaxe.

		Params:

		Returns:
		bool: whether or not the robot has a pickaxe.
		"""
		if 'wooden_pickaxe' in list(self.robot_inventory.keys()):
			return True
		else:
			return False

	def wield_pickaxe(self):
		"""
		Makes the robot wield a pickaxe.

		Params:

		Returns:
		"""
		self.get_robot_inventory()
		if self.does_robot_have_pickaxe():
			for i in range(8):
				self._agent.sendCommand("hotbar." + str(i) + " 0")
			inventory_num = self.get_inventory_number_of_pickaxe() + 1
			self._agent.sendCommand("hotbar." + str(inventory_num) + " 1")
		else:
			pass

	def get_distance_of_LoS(self):
		if 'LineOfSight' in self.world_observations.keys():
			return self.world_observations['LineOfSight']['distance']
		else:
			return 100

	def get_object_in_LoS(self):
		"""
		Returns the object the agent is currently looking at. 

		Params:

		Returns:
		The type, variant, and distance of the block the agent is currently looking at
		and whether it is in attack range. Returns [-1,-1,-1,-1] if the block is not a log, stone or cobblestone.
		"""
		if 'LineOfSight' in self.world_observations:
			# print('We see something')
			if self.world_observations['LineOfSight']['type'] == 'log':
				return self.world_observations['LineOfSight']['type'], \
					   self.world_observations['LineOfSight']['variant'], \
					   self.world_observations['LineOfSight']['distance'], \
					   self.world_observations['LineOfSight']['inRange']

			elif self.world_observations['LineOfSight']['type'] == 'cobblestone':
				return self.world_observations['LineOfSight']['type'], \
					   0, \
					   self.world_observations['LineOfSight']['distance'], \
					   self.world_observations['LineOfSight']['inRange']

			elif self.world_observations['LineOfSight']['type'] == 'stone':
				return self.world_observations['LineOfSight']['type'], \
					   self.world_observations['LineOfSight']['variant'], \
					   self.world_observations['LineOfSight']['distance'], \
					   self.world_observations['LineOfSight']['inRange']

			return [-1, -1, -1, -1]

		return [-1, -1, -1, -1]

	def shave_array_to_include_stone_group(self, stone_array, type):
		if type == 'stone':
			group = self.stone
		else:
			group = self.cobblestone
		x = []
		for i in stone_array:
			if i in group:
				x.append(i)
		return np.array(x)


	def turn_to_object(self, type, trees, dist, line, current_point, robot_info):
		"""
		Turns to the closest instance of the specified object type and returns the distance to that object.

		Params:

		type: the type of object to turn to.
		trees(tuple array): (x,y) locations of all trees
		dist(float array): euclidean distance to all trees
		line(float array): sorted euclidean distance to all trees
		current_point(tuple): current (x,y) location of robot
		robot_info: current pose of the robot (position and orientation)

		Returns:

		dist_to_object(float): distance to closest instance of specified object type
		counter(int): index of the instance chosen in the 'line' array.
		"""
		counter = 0

		# while self.get_object_in_LoS()[1] != type and counter < (len(trees) - 1):
		if type == 'birch':
			group = self.birch
		else:
			group = self.jungle
		while counter < (len(trees) - 1):
			if trees[dist[line[counter]]].tolist() not in group:
				counter += 1
				continue
			dist_to_tree = np.linalg.norm(current_point - trees[dist[line[counter]]])
			angle_to_tree = angle_clockwise((trees[dist[line[counter]]] + np.array([.5, .5])) - np.array(current_point), np.array(
				[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90

			self._agent.sendCommand("setYaw " + str(angle_to_tree + (robot_info['yaw'] % 360)))
			sleep(.3)
			# if trees[dist[line[counter]]].tolist() in self.jungle:
			# 	print('jungle tree')
			# elif trees[dist[line[counter]]].tolist() in self.birch:
			# 	print('birch')
			# else:
			# 	print('not in anything')
			if self.get_object_in_LoS()[1] == type:
				break
			counter += 1
		dist_to_object = np.linalg.norm(current_point - trees[dist[line[counter]]])

		return dist_to_object, counter

	def pitch_till_facing_stone(self, type, robot_index, single_point):
		""" for stone """

		dat = self.get_object_in_LoS()
		counter = 0
		self._agent.sendCommand("setPitch " + str(30))
		while (not dat[0] == type) or dat[3] == False or dat[2] > 3.0:
			try:

				self._agent.sendCommand("setPitch " + str(-5 + (self.world_observations['entities'][robot_index]['pitch'])))
				# print(np.linalg.norm(
				# 	single_point - np.array([self.world_observations['LineOfSight']['x'], self.world_observations['LineOfSight']['z']])))
				sleep(.2)
				dat = self.get_object_in_LoS()
				counter += 1
				if (counter > 60):
					break
			except KeyError:
				self.mine_stone(type)
				return

	def rotate_till_facing_object(self, type, robot_index, single_point):
		"""
		Rotate in place until facing a column of the input object type.

		Params:

		type: the type of object to turn to
		robot_index:
		single_point: current location of robot

		Returns:
		"""
		dat = self.get_object_in_LoS()
		counter = 0
		if type == "birch" or type == "jungle":
			while (not dat[0] == 'log') or dat[3] == False:
				try:
					self._agent.sendCommand("setYaw " + str(5 + (self.world_observations['entities'][robot_index]['yaw'] % 360)))
					sleep(.2)
					dat = self.get_object_in_LoS()
					counter += 1
					if (counter > 60):
						break
				except KeyError:
					self.chop_wood(type)
					return
		else:
			while (not dat[1] == type) or dat[3] == False or dat[2] > 3.0:
				try:

					self._agent.sendCommand("setYaw " + str(5 + (self.world_observations['entities'][robot_index]['yaw'] % 360)))
					# print(np.linalg.norm(
					# 	single_point - np.array([self.world_observations['LineOfSight']['x'], self.world_observations['LineOfSight']['z']])))
					sleep(.5)
					dat = self.get_object_in_LoS()
					counter += 1
					if (counter > 60):
						break
				except KeyError:
					self.mine_stone(type)
					return

	def chop_wood(self, type='birch'):
		""" 
		Chops a column of 4 blocks of wood, of the specified type.	
	
		Params:

		type: the type of wood to chop. Default = birch

		Returns:
		"""
		self._agent.sendCommand("setPitch " + str(0))
		self.wield_axe()

		if self.does_robot_have_axe():
			cutting_time = self.axe_cutting_time
		else:
			cutting_time = self.no_axe_cutting_time
		trees = np.array(self.per_item_dict['log'])
		robot_info, robot_index = self.get_robot_info()

		single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
		dist = self.get_euclidean_distance_to_all_trees(single_point)
		temp = {}
		for e, i in enumerate(dist):
			temp[i] = e
		dist = temp
		counter = 0
		line = sorted(dist)

		dist_to_tree, counter = self.turn_to_object(type, trees, dist, line, single_point, robot_info)
		c = 0
		old_pos = single_point

		while np.abs(dist_to_tree) > 2:
			c += 1
			self._agent.sendCommand('move .5')
			robot_info, robot_index = self.get_robot_info()
			single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
			dist_to_tree = np.linalg.norm(single_point - trees[dist[line[counter]]])
			sleep(0.01)
			# teleport
			if c >= 900:
				self._agent.sendCommand('move 0')
				multiplier = 1
				x_val_to_tp = old_pos[0] - 30 - (old_pos - trees[dist[line[counter]]])[0] * multiplier
				z_val_to_tp = old_pos[1] - 30 - (old_pos - trees[dist[line[counter]]])[1] * multiplier
				dist_to_tree = np.linalg.norm([x_val_to_tp + 30, z_val_to_tp + 30] - trees[dist[line[counter]]])

				while np.abs(dist_to_tree) < 2:
					x_val_to_tp = old_pos[0] - 30 - (old_pos - trees[dist[line[counter]]])[0] * multiplier
					z_val_to_tp = old_pos[1] - 30 - (old_pos - trees[dist[line[counter]]])[1] * multiplier
					dist_to_tree = np.linalg.norm([x_val_to_tp + 30, z_val_to_tp + 30] - trees[dist[line[counter]]])
					multiplier -= 0.01

				self._agent.sendCommand("chat /tp Robot " + str(x_val_to_tp) + " 4 " + str(z_val_to_tp))
				sleep(0.2)
				robot_info, robot_index = self.get_robot_info()

				single_point = [robot_info['x'] + 30, robot_info['z'] + 30]

				angle_to_tree = angle_clockwise((trees[dist[line[counter]]] + np.array([.5, .5])) - np.array(single_point), np.array(
					[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
				# agent_yaw = (robot_info['yaw'] + 180) % 360
				self._agent.sendCommand("setYaw " + str(angle_to_tree + (robot_info['yaw'] % 360)))

				break
		self._agent.sendCommand('move 0')

		# time to chop
		if 'LineOfSight' not in list(self.world_observations.keys()):
			self.movement_perturbation()

		dat = self.get_object_in_LoS()
		if (not dat[1] == type):
			self.rotate_till_facing_object(type, robot_index, single_point)

		pitch = 30
		self._agent.sendCommand("setPitch " + str(pitch))

		self._agent.sendCommand('attack 1')
		old_pitch = pitch
		timeout = time() + 50
		while pitch > -58:
			try:
				if (time() > timeout):
					break
				while self.get_object_in_LoS()[0] != 'log' and self.get_distance_of_LoS() < 2 and pitch > -58:
					pitch = pitch - 1
					self._agent.sendCommand("setPitch " + str(pitch))
					sleep(.05)
				if (self.get_object_in_LoS()[2] < 3 and self.get_object_in_LoS()[0] == 'log'):
					# cutting behavior
					if not (old_pitch == pitch):
						if pitch > 10 and timeout - time() <= 30:
							pitch -= 5
						sleep(cutting_time)
						old_pitch = pitch
					else:
						pitch = pitch - 1
						self._agent.sendCommand("setPitch " + str(pitch))
					continue
				else:
					pitch = pitch - 1
					self._agent.sendCommand("setPitch " + str(pitch))
					continue

			except KeyError:
				pitch = pitch - 1
				self._agent.sendCommand("setPitch " + str(pitch))

		sleep(cutting_time - 1)

		tree_x = trees[dist[line[counter]]][0] - 30
		tree_z = trees[dist[line[counter]]][1] - 30
		self._agent.sendCommand('attack 0')
		self._agent.sendCommand("chat /fill " + str(tree_x) + " 4 " + str(tree_z) + " " + str(tree_x) + " 7 " + str(tree_z) + " air 0 replace log")
		print("Command sent: " + 'chat /fill ' + str(tree_x) + " 4 " + str(tree_z) + " " + str(tree_x) + " 7 " + str(tree_z) + " air 0 replace log")

		# collect wood
		self._agent.sendCommand('move 1')
		sleep(1)
		self._agent.sendCommand('move 0')
		self._agent.sendCommand("setPitch " + str(0))

	def movement_perturbation(self):
		""" 
		Performs a slight random movement of the robot.

		Params:

		Returns:
		"""
		self._agent.sendCommand("setPitch " + str(0))
		sleep(.1)
		self._agent.sendCommand("setYaw " + str(np.random.randint(180)))
		self._agent.sendCommand('move 1')
		sleep(2)

		self._agent.sendCommand('move 0')
		self._agent.sendCommand("setYaw " + str(0))

		sleep(1)

	def mine_stone(self, type='cobblestone'):
		"""
		Mines a column of 4 blocks of stone, of the specified type.

		Params:

		type: the type of stone to mine. Default = cobblestone

		Returns:

		"""
		# self._agent.sendCommand("chat /give Robot wooden_pickaxe")
		self._agent.sendCommand("setPitch " + str(0))
		self.wield_pickaxe()
		if self.does_robot_have_pickaxe():
			cutting_time = self.axe_cutting_time
		else:
			cutting_time = self.no_axe_cutting_time
			self.craft_axe('wooden_pickaxe')
			print("you cant cut stone without a pickaxe")
			return

		robot_info, robot_index = self.get_robot_info()

		single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
		if type == 'cobblestone':
			stone = self.shave_array_to_include_stone_group(np.array(self.per_item_dict['cobblestone']), type)
		else:
			stone = self.shave_array_to_include_stone_group(np.array(self.per_item_dict['stone']), type)

		dist = (stone - single_point) ** 2
		dist = np.sum(dist, axis=1)
		dist = np.sqrt(dist)

		# get closest stone
		closest_stone_index = np.argmin(dist)
		dist_to_stone = np.linalg.norm(single_point - stone[closest_stone_index])

		angle_to_stone = angle_clockwise((stone[closest_stone_index] + np.array([.5, .5])) - np.array(single_point), np.array(
			[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90

		self._agent.sendCommand("setYaw " + str(angle_to_stone + (robot_info['yaw'] % 360)))
		old_yaw = [angle_to_stone + (robot_info['yaw'] % 360)]
		old_pos = single_point
		counter = 0
		while np.abs(dist_to_stone) > 1.5:
			counter += 1
			self._agent.sendCommand('move .5')
			robot_info, robot_index = self.get_robot_info()
			single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
			dist_to_stone = np.linalg.norm(single_point - stone[closest_stone_index])
			sleep(0.01)
			if counter >= 800:
				# teleport
				self._agent.sendCommand('move 0')
				multiplier = 1
				x_val_to_tp = old_pos[0] - 30 - (old_pos - stone[closest_stone_index])[0] * multiplier
				z_val_to_tp = old_pos[1] - 30 - (old_pos - stone[closest_stone_index])[1] * multiplier
				dist_to_stone = np.linalg.norm([x_val_to_tp + 30, z_val_to_tp + 30] - stone[closest_stone_index])
				while np.abs(dist_to_stone) < 2:
					x_val_to_tp = old_pos[0] - 30 - (old_pos - stone[closest_stone_index])[0] * multiplier
					z_val_to_tp = old_pos[1] - 30 - (old_pos - stone[closest_stone_index])[1] * multiplier
					dist_to_stone = np.linalg.norm([x_val_to_tp + 30, z_val_to_tp + 30] - stone[closest_stone_index])
					multiplier -= 0.01
				self._agent.sendCommand("chat /tp Robot " + str(x_val_to_tp) + " 4 " + str(z_val_to_tp))
				sleep(0.2)
				robot_info, robot_index = self.get_robot_info()

				single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
				angle_to_stone = angle_clockwise((stone[closest_stone_index] + np.array([.5, .5])) - np.array(single_point), np.array(
					[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
				# agent_yaw = (robot_info['yaw'] + 180) % 360
				self._agent.sendCommand("setYaw " + str(angle_to_stone + (robot_info['yaw'] % 360)))

				break
		self._agent.sendCommand('move 0')
		self.wield_pickaxe()

		if 'LineOfSight' not in list(self.world_observations.keys()):
			# NOTE: may lead to bug
			print('teleporting on line 923-934')

			self._agent.sendCommand(
				"chat /tp Robot " + str(stone[closest_stone_index][0] - 30 - 1.5) + " 4 " + str(stone[closest_stone_index][1] - 30 - 1.5))
			sleep(0.2)
			robot_info, robot_index = self.get_robot_info()

			single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
			angle_to_stone = angle_clockwise((stone[closest_stone_index] + np.array([.5, .5])) - np.array(single_point), np.array(
				[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
			# agent_yaw = (robot_info['yaw'] + 180) % 360
			self._agent.sendCommand("setYaw " + str(angle_to_stone + (robot_info['yaw'] % 360)))

			self.movement_perturbation()

		dat = self.get_object_in_LoS()

		if (not dat[0] == type):
			self.pitch_till_facing_stone(type, robot_index, single_point)

		pitch = 30
		self._agent.sendCommand("setPitch " + str(pitch))

		self._agent.sendCommand('attack 1')
		old_pitch = pitch
		timeout = time() + 55

		while pitch > -59:
			try:
				if (time() > timeout):
					break
				while self.get_object_in_LoS()[0] != type and pitch > -60:
					# moving up behavior
					pitch = pitch - 1
					self._agent.sendCommand("setPitch " + str(pitch))
					sleep(.05)
				if (self.get_object_in_LoS()[2] < 3 and self.get_object_in_LoS()[0] == type):
					# cutting behavior
					if not (old_pitch == pitch):
						if pitch > 10 and timeout - time() <= 30:
							pitch -= 5
						sleep(cutting_time)
						old_pitch = pitch
					else:
						pitch = pitch - 1
						self._agent.sendCommand("setPitch " + str(pitch))
					continue
				else:
					pitch = pitch - 1
					self._agent.sendCommand("setPitch " + str(pitch))
					continue
			except KeyError:
				pitch = pitch - 1
				self._agent.sendCommand("setPitch " + str(pitch))

		self._agent.sendCommand('attack 0')

		stone_x = stone[closest_stone_index][0] - 30
		stone_z = stone[closest_stone_index][1] - 30
		self._agent.sendCommand(
			"chat /fill " + str(stone_x) + " 4 " + str(stone_z) + " " + str(stone_x) + " 7 " + str(stone_z) + " air 0 replace " + type)
		# collect stone
		self._agent.sendCommand('move 1')
		sleep(1)
		self._agent.sendCommand('move 0')
		self._agent.sendCommand("setPitch " + str(0))

	def get_number_of_robot_planks(self):
		if 'planks' in list(self.robot_inventory.keys()):
			return self.robot_inventory['planks']
		else:
			return 0

	def get_number_of_robot_wood(self):
		"""
		Gets the number of logs in the robot's inventory.

		Params:

		Returns:
		int: the number of logs in the robot's inventory
		"""

		# returns total of all types
		if 'log' in list(self.robot_inventory.keys()):
			return self.robot_inventory['log']
		else:
			return 0

	def get_number_of_robot_sticks(self):
		"""
		Gets the number of sticks in the robot's inventory.

		Params:

		Returns:
		int: the number of sticks in the robot's inventory
		"""
		if 'stick' in list(self.robot_inventory.keys()):
			return self.robot_inventory['stick']
		else:
			return 0

	def get_number_of_robot_stone(self):
		"""
		Gets the number of stone in the robot's inventory.

		Params:

		Returns:
		int: the number of stone in the robot's inventory
		"""
		if 'stone' in list(self.robot_inventory.keys()):
			return self.robot_inventory['stone']
		else:
			return 0

	def get_number_of_robot_cobblestone(self):
		"""
		Gets the number of cobblestone in the robot's inventory.

		Params:

		Returns:
		int: the number of cobblestone in the robot's inventory
		"""
		if 'cobblestone' in list(self.robot_inventory.keys()):
			return self.robot_inventory['cobblestone']
		else:
			return 0

	def get_number_of_chest_planks(self):
		"""
		Gets the number of planks in the chest.

		Params:

		Returns:
		int: the number of planks in the chest.
		"""
		if 'planks' in list(self.chest_state.keys()):
			return self.chest_state['planks']
		else:
			return 0

	def get_number_of_chest_stick(self):
		"""
		Gets the number of sticks in the chest.

		Params:

		Returns:
		int: the number of sticks in the chest.
		"""
		if 'stick' in list(self.chest_state.keys()):
			return self.chest_state['stick']
		else:
			return 0

	def get_number_of_chest_wood(self):
		"""
		Gets the number of logs in the chest.

		Params:

		Returns:
		int: the number of logs in the chest.
		"""
		if 'log' in list(self.chest_state.keys()):
			return self.chest_state['log']
		else:
			return 0

	def get_inventory_number_of_axe(self):
		"""
		Finds which slot of the robot's inventory contains an axe.

		Params:

		Returns:
		int: the slot of the robot's inventory that contains an axe.
		"""
		inventory_keys = ['InventorySlot_0_item', 'InventorySlot_1_item', 'InventorySlot_2_item', 'InventorySlot_3_item',
						  'InventorySlot_4_item', 'InventorySlot_5_item', 'InventorySlot_6_item', 'InventorySlot_7_item',
						  'InventorySlot_8_item', 'InventorySlot_9_item', 'InventorySlot_10_item', 'InventorySlot_11_item',
						  'InventorySlot_12_item', 'InventorySlot_13_item', 'InventorySlot_14_item', 'InventorySlot_15_item',
						  'InventorySlot_16_item', 'InventorySlot_17_item', 'InventorySlot_18_item', 'InventorySlot_19_item',
						  'InventorySlot_20_item', 'InventorySlot_21_item', 'InventorySlot_22_item', 'InventorySlot_23_item',
						  'InventorySlot_24_item', 'InventorySlot_25_item', 'InventorySlot_26_item', 'InventorySlot_27_item',
						  'InventorySlot_28_item', 'InventorySlot_29_item', 'InventorySlot_30_item', 'InventorySlot_31_item',
						  'InventorySlot_32_item', 'InventorySlot_33_item', 'InventorySlot_34_item', 'InventorySlot_35_item',
						  'InventorySlot_36_item', 'InventorySlot_37_item', 'InventorySlot_38_item', 'InventorySlot_39_item',
						  'InventorySlot_40_item']
		for e, i in enumerate(inventory_keys):
			if 'wooden_axe' == str(self.world_observations[i]):
				if e in self.inventory_axes:
					pass
				else:
					self.inventory_axes.append(e)
					return e

		return np.random.choice(self.inventory_axes)

	def get_inventory_number_of_pickaxe(self):
		"""
		Finds which slot of the robot's inventory contains a pickaxe.

		Params:

		Returns:
		int: the slot of the robot's inventory that contains a pickaxe.
		"""
		inventory_keys = ['InventorySlot_0_item', 'InventorySlot_1_item', 'InventorySlot_2_item', 'InventorySlot_3_item',
						  'InventorySlot_4_item', 'InventorySlot_5_item', 'InventorySlot_6_item', 'InventorySlot_7_item',
						  'InventorySlot_8_item', 'InventorySlot_9_item', 'InventorySlot_10_item', 'InventorySlot_11_item',
						  'InventorySlot_12_item', 'InventorySlot_13_item', 'InventorySlot_14_item', 'InventorySlot_15_item',
						  'InventorySlot_16_item', 'InventorySlot_17_item', 'InventorySlot_18_item', 'InventorySlot_19_item',
						  'InventorySlot_20_item', 'InventorySlot_21_item', 'InventorySlot_22_item', 'InventorySlot_23_item',
						  'InventorySlot_24_item', 'InventorySlot_25_item', 'InventorySlot_26_item', 'InventorySlot_27_item',
						  'InventorySlot_28_item', 'InventorySlot_29_item', 'InventorySlot_30_item', 'InventorySlot_31_item',
						  'InventorySlot_32_item', 'InventorySlot_33_item', 'InventorySlot_34_item', 'InventorySlot_35_item',
						  'InventorySlot_36_item', 'InventorySlot_37_item', 'InventorySlot_38_item', 'InventorySlot_39_item',
						  'InventorySlot_40_item']
		for e, i in enumerate(inventory_keys):
			if 'wooden_pickaxe' == str(self.world_observations[i]):
				if e in self.inventory_pickaxes:
					pass
				else:
					self.inventory_pickaxes.append(e)
					return e

		return np.random.choice(self.inventory_pickaxes)

	def craft_axe(self, axe_type='wooden_axe'):
		"""
		Crafts an axe of the specified type.

		Params:

		axe_type: the type of axe to craft. Default = wooden_axe

		Returns:
		"""

		sleep(1)
		self.get_robot_inventory()

		if self.get_number_of_robot_planks() >= 3 and self.get_number_of_robot_sticks() >= 2:

			# take away resources
			plank_info = self.find_plank_inventory_slot()
			tot_planks_needed = 3
			for plank_chest_slot, plank_quantity, plank_variant in plank_info:
				if plank_variant == 'birch':
					mod = ' 2'
				else:
					mod = ' 3'
				if plank_quantity > tot_planks_needed:
					self._agent.sendCommand(
						"chat /replaceitem entity Robot slot.hotbar." + str(plank_chest_slot) + " planks " + str(plank_quantity - tot_planks_needed
																												 ) + mod)
					tot_planks_needed = 0
				else:
					self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(plank_chest_slot) + " air")
					tot_planks_needed -= plank_quantity
				if tot_planks_needed == 0:
					break
				sleep(1)

			# sticks
			sticks_chest_slot, sticks_quantity = self.find_sticks_inventory_slot()
			if sticks_quantity > 2:
				self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(sticks_chest_slot) + " stick " + str(sticks_quantity - 2))
			else:
				self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(sticks_chest_slot) + " air")

			# give axe
			self._agent.sendCommand("chat /give Robot " + str(axe_type) + " 1 45")
			sleep(0.5)
			if axe_type == 'wooden_axe':
				if self.does_robot_have_an_axe_already():
					self.put_axe_in_chest(axe_type)
					self._agent.sendCommand("use 0")
			else:
				if self.does_robot_have_an_pickaxe_already():
					self.put_axe_in_chest(axe_type)
					self._agent.sendCommand("use 0")

		elif self.get_number_of_robot_planks() >= 5:
			plank_info = self.find_plank_inventory_slot()
			tot_planks_needed = 5
			for plank_chest_slot, plank_quantity, plank_variant in plank_info:
				if plank_variant == 'birch':
					mod = ' 2'
				else:
					mod = ' 3'
				if plank_quantity > tot_planks_needed:
					self._agent.sendCommand(
						"chat /replaceitem entity Robot slot.hotbar." + str(plank_chest_slot) + " planks " + str(plank_quantity - tot_planks_needed
																												 ) + mod)
					tot_planks_needed = 0
				else:
					self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(plank_chest_slot) + " air")
					tot_planks_needed -= plank_quantity
				if tot_planks_needed == 0:
					break
				sleep(1)

			# give stick
			self._agent.sendCommand("chat /give Robot stick 2")
			# give axe
			self._agent.sendCommand("chat /give Robot " + str(axe_type) + " 1 48")

			sleep(0.5)
			if axe_type == 'wooden_axe':
				if self.does_robot_have_an_axe_already():
					self.put_axe_in_chest(axe_type)
					self._agent.sendCommand("use 0")
			else:
				if self.does_robot_have_an_pickaxe_already():
					self.put_axe_in_chest(axe_type)
					self._agent.sendCommand("use 0")

		elif self.get_number_of_robot_wood() >= 2:
			wood_info = self.find_wood_inventory_slot()
			tot_wood_needed = 2
			for wood_chest_slot, wood_quantity, wood_variant in wood_info:
				if wood_variant == 'birch':
					mod = ' 2'
				else:
					mod = ' 3'
				if wood_quantity > tot_wood_needed:
					self._agent.sendCommand(
						"chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " log " + str(wood_quantity - tot_wood_needed
																											 ) + mod)
					tot_wood_needed = 0
				else:
					self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " air")
					tot_wood_needed -= wood_quantity
				if tot_wood_needed == 0:
					break
				sleep(1)
			# give axe
			self._agent.sendCommand("chat /give Robot planks 3" + mod)
			self._agent.sendCommand("chat /give Robot stick 2")
			self._agent.sendCommand("chat /give Robot " + str(axe_type) + " 1 48")

			sleep(0.5)
			if axe_type == 'wooden_axe':
				if self.does_robot_have_an_axe_already():
					self.put_axe_in_chest(axe_type)
					self._agent.sendCommand("use 0")
			else:
				if self.does_robot_have_an_pickaxe_already():
					self.put_axe_in_chest(axe_type)
					self._agent.sendCommand("use 0")
		else:
			self.put_resources_in_chest()
			print('robot does not have resources')
			# Getting resources from chest
			self.pull_resources_from_chest()
			if self.get_number_of_robot_wood() >= 2:
				self.craft_axe(axe_type)
			else:
				return

	def craft_planks(self, type='birch'):
		"""
		Makes the robot craft planks.

		Params:

		Returns:
		"""
		self.get_robot_inventory()
		if self.get_number_of_robot_wood() > 0:
			wood_info = self.find_wood_inventory_slot()
			tot_wood_needed = 1
			for wood_chest_slot, wood_quantity, wood_variant in wood_info:
				if wood_variant == 'birch':
					mod = ' 2'
				else:
					mod = ' 3'
				if wood_variant == type:
					if wood_quantity > tot_wood_needed:
						self._agent.sendCommand(
							"chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " log " + str(wood_quantity - tot_wood_needed
																												 ) + mod)
						tot_wood_needed = 0
					else:
						self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " air")
						tot_wood_needed -= wood_quantity
					# give planks
					self._agent.sendCommand("chat /give Robot planks 4 " + mod)
					sleep(4)
				if tot_wood_needed == 0:
					sleep(1)
					break
					return

		print('You do not have the resources to craft planks')
		return

	def craft_sticks(self):
		"""
		Makes the robot craft sticks.

		Params:

		Returns:
		"""
		self.get_robot_inventory()

		if self.get_number_of_robot_planks() >= 2:
			plank_info = self.find_plank_inventory_slot()
			tot_planks_needed = 2
			for plank_chest_slot, plank_quantity, plank_variant in plank_info:
				if plank_variant == 'birch':
					mod = ' 2'
				else:
					mod = ' 3'
				if plank_quantity > tot_planks_needed:
					self._agent.sendCommand(
						"chat /replaceitem entity Robot slot.hotbar." + str(plank_chest_slot) + " planks " + str(plank_quantity - tot_planks_needed
																												 ) + mod)
					tot_planks_needed = 0

				else:
					self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(plank_chest_slot) + " air")
					tot_planks_needed -= plank_quantity
				if tot_planks_needed == 0:
					break
				sleep(1)
				self._agent.sendCommand("chat /give Robot stick 4")
				sleep(0.5)


		elif self.get_number_of_robot_wood() > 0:
			wood_info = self.find_wood_inventory_slot()
			tot_wood_needed = 1
			for wood_chest_slot, wood_quantity, wood_variant in wood_info:
				if wood_variant == 'birch':
					mod = ' 2'
				else:
					mod = ' 3'
				if wood_quantity > tot_wood_needed:
					self._agent.sendCommand(
						"chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " log " + str(wood_quantity - tot_wood_needed
																											 ) + mod)
					tot_wood_needed = 0
				# self._agent.sendCommand("swapInventoryItems chest:" + str(empty_chest_slot) + " inventory:" + str(wood_chest_slot))
				else:
					self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " air")
					tot_wood_needed -= wood_quantity
				if tot_wood_needed == 0:
					break
				sleep(1)
			# give axe
			self._agent.sendCommand("chat /give Robot planks 2 " + mod)
			self._agent.sendCommand("chat /give Robot stick 4")

			sleep(0.5)
		else:
			print('you do not have the resources to craft sticks')

	# self.get_robot_inventory()
	# if self.get_number_of_robot_planks() > 0:
	# 	self._agent.sendCommand("craft stick")
	# 	sleep(0.5)
	# elif self.get_number_of_robot_wood() > 0:
	# 	self._agent.sendCommand("craft planks")
	# 	sleep(0.5)
	# 	self._agent.sendCommand("craft stick")
	# 	sleep(0.5)
	# else:
	# 	print('you do not have the resources to craft planks')

	def update_chest(self):
		"""
		Updates the status of the chest.

		Params:

		Returns:
		"""
		robot_info, robot_index = self.get_robot_info()
		# single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
		# dist = np.linalg.norm(np.array(single_point) - np.array(self.chest_location))
		#
		# angle_to_chest = angle_clockwise((np.array(self.chest_location)) - np.array(single_point), np.array(
		# 	[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
		# # agent_yaw = (robot_info['yaw'] + 180) % 360
		# self._agent.sendCommand("setYaw " + str(angle_to_chest + (robot_info['yaw'] % 360)))
		#
		# pitch = 90 - np.arctan2(dist, 1.5) * 180 / np.pi
		# self._agent.sendCommand("setPitch " + str(pitch))
		if len(self.world_observations['inventoriesAvailable']) > 1:
			inventory_keys = {'container.chestSlot_0_size': 'container.chestSlot_0_item',
							  'container.chestSlot_1_size': 'container.chestSlot_1_item',
							  'container.chestSlot_2_size': 'container.chestSlot_2_item',
							  'container.chestSlot_3_size': 'container.chestSlot_3_item',
							  'container.chestSlot_4_size': 'container.chestSlot_4_item',
							  'container.chestSlot_5_size': 'container.chestSlot_5_item',
							  'container.chestSlot_6_size': 'container.chestSlot_6_item',
							  'container.chestSlot_7_size': 'container.chestSlot_7_item',
							  'container.chestSlot_8_size': 'container.chestSlot_8_item',
							  'container.chestSlot_9_size': 'container.chestSlot_9_item',
							  'container.chestSlot_10_size': 'container.chestSlot_10_item',
							  'container.chestSlot_11_size': 'container.chestSlot_11_item',
							  'container.chestSlot_12_size': 'container.chestSlot_12_item',
							  'container.chestSlot_13_size': 'container.chestSlot_13_item',
							  'container.chestSlot_14_size': 'container.chestSlot_14_item',
							  'container.chestSlot_15_size': 'container.chestSlot_15_item',
							  'container.chestSlot_16_size': 'container.chestSlot_16_item',
							  'container.chestSlot_17_size': 'container.chestSlot_17_item',
							  'container.chestSlot_18_size': 'container.chestSlot_18_item',
							  'container.chestSlot_19_size': 'container.chestSlot_19_item',
							  'container.chestSlot_20_size': 'container.chestSlot_20_item',
							  'container.chestSlot_21_size': 'container.chestSlot_21_item',
							  'container.chestSlot_22_size': 'container.chestSlot_22_item',
							  'container.chestSlot_23_size': 'container.chestSlot_23_item',
							  'container.chestSlot_24_size': 'container.chestSlot_24_item',
							  'container.chestSlot_25_size': 'container.chestSlot_25_item',
							  'container.chestSlot_26_size': 'container.chestSlot_26_item'}
			# resetting chest to zero

			for i in inventory_keys:
				try:
					self.chest_state[self.world_observations[inventory_keys[i]]] = 0
				except KeyError:
					print('keyerror on', i)
					return

			elements = ['log', 'wooden_axe',
						'planks',
						'wooden_pickaxe',
						'cobblestone',
						'stone',
						'stick',
						'air',
						'jungle_fence',
						'birch_fence'
						]
			for i in elements:
				if i not in self.chest_state.keys():
					self.chest_state[i] = 0
				else:
					self.chest_state[i] = 0
			# recounting
			for i in inventory_keys:
				try:
					if self.world_observations[i] != 0:
						self.chest_state[self.world_observations[inventory_keys[i]]] += self.world_observations[i]
				except KeyError:
					continue

	def go_to_chest(self):
		"""
		Moves the robot to the chest.

		Params:

		Returns:
		"""
		robot_info, robot_index = self.get_robot_info()
		single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
		dist = np.linalg.norm(np.array(single_point) - np.array(self.chest_location))

		angle_to_chest = angle_clockwise((np.array(self.chest_location)) - np.array(single_point), np.array(
			[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
		# agent_yaw = (robot_info['yaw'] + 180) % 360
		self._agent.sendCommand("setYaw " + str(angle_to_chest + (robot_info['yaw'] % 360)))

		pitch = 90 - np.arctan2(dist, 1.5) * 180 / np.pi
		self._agent.sendCommand("setPitch " + str(pitch))
		sleep(.5)

		if self.world_observations['LineOfSight']['type'] == 'chest':
			while self.world_observations['LineOfSight']['inRange'] == False:
				self._agent.sendCommand('move .5')
				robot_info, robot_index = self.get_robot_info()
				single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
				dist = np.linalg.norm(np.array(single_point) - np.array(self.chest_location))

				angle_to_chest = angle_clockwise((np.array(self.chest_location)) - np.array(single_point), np.array(
					[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
				# agent_yaw = (robot_info['yaw'] + 180) % 360
				self._agent.sendCommand("setYaw " + str(angle_to_chest + (robot_info['yaw'] % 360)))

				pitch = 90 - np.arctan2(dist, 1.5) * 180 / np.pi
				self._agent.sendCommand("setPitch " + str(pitch))
				sleep(.5)
				print('moving')
				print(self.world_observations['LineOfSight']['inRange'])

			self._agent.sendCommand('move 0')
			self._agent.sendCommand('use 1')

	def put_axe_in_chest(self, axe_type='wooden_axe'):
		"""
		Makes the robot put an axe in the chest of the specified type

		Params:

		axe_type(str): the type of axe to put in the chest. Either wooden_axe or wooden_pickaxe.

		Returns:
		"""
		robot_info, robot_index = self.get_robot_info()
		single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
		dist = np.linalg.norm(np.array(single_point) - np.array(self.chest_location))

		angle_to_chest = angle_clockwise((np.array(self.chest_location)) - np.array(single_point), np.array(
			[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
		# agent_yaw = (robot_info['yaw'] + 180) % 360
		self._agent.sendCommand("setYaw " + str(angle_to_chest + (robot_info['yaw'] % 360)))

		pitch = 90 - np.arctan2(dist, 1.5) * 180 / np.pi
		self._agent.sendCommand("setPitch " + str(pitch))
		sleep(.5)

		if self.world_observations['LineOfSight']['type'] == 'chest':
			while self.world_observations['LineOfSight']['inRange'] == False:
				self._agent.sendCommand('move .5')
				robot_info, robot_index = self.get_robot_info()
				single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
				dist = np.linalg.norm(np.array(single_point) - np.array(self.chest_location))

				angle_to_chest = angle_clockwise((np.array(self.chest_location)) - np.array(single_point), np.array(
					[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
				# agent_yaw = (robot_info['yaw'] + 180) % 360
				self._agent.sendCommand("setYaw " + str(angle_to_chest + (robot_info['yaw'] % 360)))

				pitch = 90 - np.arctan2(dist, 1.5) * 180 / np.pi
				self._agent.sendCommand("setPitch " + str(pitch))
				sleep(.5)

			self._agent.sendCommand('move 0')
		# NOTE We get close to it, and then auto add/subtract
		else:
			self._agent.sendCommand("chat /tp Robot 4 4 9")
			self.put_axe_in_chest(axe_type)
			return

		self.get_robot_inventory()
		if axe_type == 'wooden_axe':
			if self.does_robot_have_an_axe_already():
				empty_inventory_slot = self.find_empty_inventory_slot()
				empty_chest_slot = self.find_empty_chest_slot()
				axe_inventory_number = self.get_inventory_number_of_axe()
				print(axe_inventory_number)
				self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(axe_inventory_number) + " air")
				self._agent.sendCommand("chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " " + str(axe_type) + " 1 48")

				self.inventory_axes.remove(axe_inventory_number)
		else:
			empty_inventory_slot = self.find_empty_inventory_slot()
			empty_chest_slot = self.find_empty_chest_slot()
			axe_inventory_number = self.get_inventory_number_of_pickaxe()
			print(axe_inventory_number)
			self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(axe_inventory_number) + " air")
			self._agent.sendCommand("chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " " + str(axe_type) + " 1 48")

			self.inventory_pickaxes.remove(axe_inventory_number)

	def put_resources_in_chest(self):
		"""
		Puts the robot's resources in the chest while keeping enough for crafting.

		Params:

		Returns:
		"""
		if np.random.uniform(0, 1) > .5:

			self._agent.sendCommand("chat /tp Robot 4 4 9")
			robot_info, robot_index = self.get_robot_info()
			single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
			dist = np.linalg.norm(np.array(single_point) - np.array(self.chest_location))

			angle_to_chest = angle_clockwise((np.array(self.chest_location)) - np.array(single_point), np.array(
				[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
			self._agent.sendCommand("setYaw " + str(-4 + angle_to_chest + (robot_info['yaw'] % 360)))

			pitch = 90 - np.arctan2(dist, 1.5) * 180 / np.pi
			self._agent.sendCommand("setPitch " + str(pitch))
			sleep(.2)
		else:
			self._agent.sendCommand("chat /tp Robot 7.5 4 13.75")
			robot_info, robot_index = self.get_robot_info()
			single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
			dist = np.linalg.norm(np.array(single_point) - np.array(self.chest_location))

			angle_to_chest = angle_clockwise((np.array(self.chest_location)) - np.array(single_point), np.array(
				[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
			self._agent.sendCommand("setYaw " + str(155))

			pitch = 90 - np.arctan2(dist, 1.5) * 180 / np.pi
			self._agent.sendCommand("setPitch " + str(29.2))
			sleep(.2)

		if self.world_observations['LineOfSight']['type'] == 'chest':
			while self.world_observations['LineOfSight']['inRange'] == False:
				self._agent.sendCommand('move .5')
				robot_info, robot_index = self.get_robot_info()
				single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
				dist = np.linalg.norm(np.array(single_point) - np.array(self.chest_location))

				angle_to_chest = angle_clockwise((np.array(self.chest_location)) - np.array(single_point), np.array(
					[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
				# agent_yaw = (robot_info['yaw'] + 180) % 360
				self._agent.sendCommand("setYaw " + str(angle_to_chest + (robot_info['yaw'] % 360)))

				pitch = 90 - np.arctan2(dist, 1.5) * 180 / np.pi
				self._agent.sendCommand("setPitch " + str(pitch))
				sleep(.5)

			self._agent.sendCommand('move 0')
		# NOTE We get close to it, and then auto add/subtract
		else:
			# self._agent.sendCommand("chat /tp Robot 4 4 9")
			self.put_resources_in_chest()
		# self._agent.sendCommand('use 1')
		self.get_robot_inventory()
		# if self.get_number_of_robot_wood() > 2:
		# 	# log 1 2 is 1 log of variant 2 (birch)
		# 	wood_info = self.find_wood_inventory_slot()
		# 	for wood_chest_slot, wood_quantity, wood_variant in wood_info:
		# 		empty_inventory_slot = self.find_empty_inventory_slot()
		# 		empty_chest_slot = self.find_empty_chest_slot()
		# 		if wood_variant == 'birch':
		# 			mod = ' 2'
		# 		else:
		# 			mod = ' 3'
		# 		if wood_quantity > 2:
		# 			# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " birch_log: 2")
		# 			self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " log " + str(2) + mod)
		# 			# self._agent.sendCommand("swapInventoryItems chest:" + str(empty_chest_slot) + " inventory:" + str(wood_chest_slot))
		# 			self._agent.sendCommand(
		# 				"chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " log " + str(wood_quantity - 2) + mod)
		#
		# 		sleep(1)
		#
		# if self.get_number_of_robot_planks() > 2:
		#
		# 	plank_info = self.find_plank_inventory_slot()
		# 	for plank_chest_slot, plank_quantity, plank_variant in plank_info:
		# 		empty_inventory_slot = self.find_empty_inventory_slot()
		# 		empty_chest_slot = self.find_empty_chest_slot()
		# 		if plank_variant == 'birch':
		# 			mod = ' 2'
		# 		else:
		# 			mod = ' 3'
		# 		if plank_quantity > 2:
		# 			# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " birch_log: 2")
		# 			self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(plank_chest_slot) + " planks " + str(2) + mod)
		# 			# self._agent.sendCommand("swapInventoryItems chest:" + str(empty_chest_slot) + " inventory:" + str(wood_chest_slot))
		# 			self._agent.sendCommand(
		# 				"chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " planks " + str(plank_quantity - 2) + mod)
		# 			sleep(1)
		#
		# if self.get_number_of_robot_sticks() > 2:
		# 	empty_inventory_slot = self.find_empty_inventory_slot()
		# 	empty_chest_slot = self.find_empty_chest_slot()
		# 	sticks_chest_slot, sticks_quantity = self.find_sticks_inventory_slot()
		# 	# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(empty_inventory_slot) + " stick 2")
		# 	self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(sticks_chest_slot) + " stick " + str(2))
		# 	self._agent.sendCommand(
		# 		"chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " stick " + str(sticks_quantity - 2))
		# 	sleep(1)
		#
		# if self.get_number_of_robot_stone() > 2:
		# 	stone_info = self.find_stone_inventory_slot()
		# 	for stone_chest_slot, stone_quantity, stone_variant in stone_info:
		# 		empty_inventory_slot = self.find_empty_inventory_slot()
		# 		empty_chest_slot = self.find_empty_chest_slot()
		# 		if stone_quantity > 2:
		# 			# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " birch_log: 2")
		# 			self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(stone_chest_slot) + " stone " + str(2) + ' 6')
		# 			# self._agent.sendCommand("swapInventoryItems chest:" + str(empty_chest_slot) + " inventory:" + str(wood_chest_slot))
		# 			self._agent.sendCommand(
		# 				"chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " stone " + str(stone_quantity - 2) + ' 6')
		# 			sleep(1)
		#
		# if self.get_number_of_robot_cobblestone() > 2:
		# 	empty_chest_slot = self.find_empty_chest_slot()
		# 	cobblestone_chest_slot, cobblestone_quantity = self.find_cobblestone_inventory_slot()
		# 	self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(cobblestone_chest_slot) + " cobblestone " + str(2))
		# 	self._agent.sendCommand(
		# 		"chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " cobblestone " + str(cobblestone_quantity - 2))
		# 	sleep(1)

		self.update_chest()

	# self.save_chest_information()

	def save_chest_information(self):
		"""
		Saves the current state of the chest in a pickle file.

		Params:

		Returns:
		"""
		save_pickle(os.curdir + '/' + str(self.participant_number), [self.chest_state], 'chest_state' + str(self.cur_time) + '.pkl', False)

	def clean_up_inventory(self):
		pass

	def find_empty_inventory_slot(self):
		"""
		Finds an empty slot in the robot's hotbar.

		Params:

		Returns:

		int: the first robot's hotbar slot that is empty. 
		"""
		inventory_of_robot = {'Hotbar_0_size': 'Hotbar_0_item', 'Hotbar_1_size': 'Hotbar_1_item', 'Hotbar_2_size': 'Hotbar_2_item',
							  'Hotbar_3_size': 'Hotbar_3_item',
							  'Hotbar_4_size': 'Hotbar_4_item', 'Hotbar_5_size': 'Hotbar_5_item', 'Hotbar_6_size': 'Hotbar_6_item',
							  'Hotbar_7_size': 'Hotbar_7_item',
							  'Hotbar_8_size': 'Hotbar_8_item'}
		for e, i in enumerate(inventory_of_robot.keys()):
			if self.world_observations[i] == 0:
				return e

	def find_wood_inventory_slot(self):
		""" 
		Finds the slots in the robot's hotbar that contains logs.

		Params:

		Returns:
		alist(int array): the slots in the robot's hotbar that contain logs.
		"""
		inventory_of_robot = {'Hotbar_0_size': 'Hotbar_0_item', 'Hotbar_1_size': 'Hotbar_1_item', 'Hotbar_2_size': 'Hotbar_2_item',
							  'Hotbar_3_size': 'Hotbar_3_item',
							  'Hotbar_4_size': 'Hotbar_4_item', 'Hotbar_5_size': 'Hotbar_5_item', 'Hotbar_6_size': 'Hotbar_6_item',
							  'Hotbar_7_size': 'Hotbar_7_item',
							  'Hotbar_8_size': 'Hotbar_8_item'}
		alist = []
		for e, i in enumerate(inventory_of_robot.keys()):
			if self.world_observations[inventory_of_robot[i]] == 'log':
				alist.append((e, self.world_observations[i], self.world_observations['InventorySlot_' + str(e) + '_variant']))
		return alist

	def find_cobblestone_inventory_slot(self):
		""" 
		Finds the slots in the robot's hotbar that contain cobblestone.

		Params:

		Returns:
		alist(int array): the slots in the robot's hotbar that contain cobblestone.
		"""
		# inventory_of_robot = {'Hotbar_0_size': 'Hotbar_0_item', 'Hotbar_1_size': 'Hotbar_1_item', 'Hotbar_2_size': 'Hotbar_2_item',
		# 					  'Hotbar_3_size': 'Hotbar_3_item',
		# 					  'Hotbar_4_size': 'Hotbar_4_item', 'Hotbar_5_size': 'Hotbar_5_item', 'Hotbar_6_size': 'Hotbar_6_item',
		# 					  'Hotbar_7_size': 'Hotbar_7_item',
		# 					  'Hotbar_8_size': 'Hotbar_8_item'}
		inventory_of_robot = {'InventorySlot_0_size': 'InventorySlot_0_item',
							  'InventorySlot_1_size': 'InventorySlot_1_item',
							  'InventorySlot_2_size': 'InventorySlot_2_item',
							  'InventorySlot_3_size': 'InventorySlot_3_item',
							  'InventorySlot_4_size': 'InventorySlot_4_item',
							  'InventorySlot_5_size': 'InventorySlot_5_item',
							  'InventorySlot_6_size': 'InventorySlot_6_item',
							  'InventorySlot_7_size': 'InventorySlot_7_item',
							  'InventorySlot_8_size': 'InventorySlot_8_item',
							  'InventorySlot_9_size': 'InventorySlot_9_item',
							  'InventorySlot_10_size': 'InventorySlot_10_item',
							  'InventorySlot_11_size': 'InventorySlot_11_item',
							  'InventorySlot_12_size': 'InventorySlot_12_item',
							  'InventorySlot_13_size': 'InventorySlot_13_item',
							  'InventorySlot_14_size': 'InventorySlot_14_item',
							  'InventorySlot_15_size': 'InventorySlot_15_item',
							  'InventorySlot_16_size': 'InventorySlot_16_item',
							  'InventorySlot_17_size': 'InventorySlot_17_item',
							  'InventorySlot_18_size': 'InventorySlot_18_item',
							  'InventorySlot_19_size': 'InventorySlot_19_item',
							  'InventorySlot_20_size': 'InventorySlot_20_item',
							  'InventorySlot_21_size': 'InventorySlot_21_item',
							  'InventorySlot_22_size': 'InventorySlot_22_item',
							  'InventorySlot_23_size': 'InventorySlot_23_item',
							  'InventorySlot_24_size': 'InventorySlot_24_item',
							  'InventorySlot_25_size': 'InventorySlot_25_item',
							  'InventorySlot_26_size': 'InventorySlot_26_item',
							  'InventorySlot_27_size': 'InventorySlot_27_item',
							  'InventorySlot_28_size': 'InventorySlot_28_item',
							  'InventorySlot_29_size': 'InventorySlot_29_item',
							  'InventorySlot_30_size': 'InventorySlot_30_item',
							  'InventorySlot_31_size': 'InventorySlot_31_item',
							  'InventorySlot_32_size': 'InventorySlot_32_item',
							  'InventorySlot_33_size': 'InventorySlot_33_item',
							  'InventorySlot_34_size': 'InventorySlot_34_item',
							  'InventorySlot_35_size': 'InventorySlot_35_item',
							  'InventorySlot_36_size': 'InventorySlot_36_item',
							  'InventorySlot_37_size': 'InventorySlot_37_item',
							  'InventorySlot_38_size': 'InventorySlot_38_item',
							  'InventorySlot_39_size': 'InventorySlot_39_item',
							  'InventorySlot_40_size': 'InventorySlot_40_item'}

		# inventory_of_robot = ['InventorySlot_0_item', 'InventorySlot_1_item', 'InventorySlot_2_item', 'InventorySlot_3_item',
		# 				  'InventorySlot_4_item', 'InventorySlot_5_item', 'InventorySlot_6_item', 'InventorySlot_7_item',
		# 				  'InventorySlot_8_item', 'InventorySlot_9_item', 'InventorySlot_10_item', 'InventorySlot_11_item',
		# 				  'InventorySlot_12_item', 'InventorySlot_13_item', 'InventorySlot_14_item', 'InventorySlot_15_item',
		# 				  'InventorySlot_16_item', 'InventorySlot_17_item', 'InventorySlot_18_item', 'InventorySlot_19_item',
		# 				  'InventorySlot_20_item', 'InventorySlot_21_item', 'InventorySlot_22_item', 'InventorySlot_23_item',
		# 				  'InventorySlot_24_item', 'InventorySlot_25_item', 'InventorySlot_26_item', 'InventorySlot_27_item',
		# 				  'InventorySlot_28_item', 'InventorySlot_29_item', 'InventorySlot_30_item', 'InventorySlot_31_item',
		# 				  'InventorySlot_32_item', 'InventorySlot_33_item', 'InventorySlot_34_item', 'InventorySlot_35_item',
		# 				  'InventorySlot_36_item', 'InventorySlot_37_item', 'InventorySlot_38_item', 'InventorySlot_39_item',
		# 				  'InventorySlot_40_item']
		alist = []
		for e, i in enumerate(inventory_of_robot.keys()):
			if self.world_observations[inventory_of_robot[i]] == 'cobblestone':
				return (e, self.world_observations[i])

	def find_plank_inventory_slot(self):
		""" 
		Finds the slots in the robot's hotbar that contain planks.

		Params:

		Returns:
		alist(int array): the slots in the robot's hotbar that contain planks.
		"""
		inventory_of_robot = {'Hotbar_0_size': 'Hotbar_0_item', 'Hotbar_1_size': 'Hotbar_1_item', 'Hotbar_2_size': 'Hotbar_2_item',
							  'Hotbar_3_size': 'Hotbar_3_item',
							  'Hotbar_4_size': 'Hotbar_4_item', 'Hotbar_5_size': 'Hotbar_5_item', 'Hotbar_6_size': 'Hotbar_6_item',
							  'Hotbar_7_size': 'Hotbar_7_item',
							  'Hotbar_8_size': 'Hotbar_8_item'}

		alist = []
		for e, i in enumerate(inventory_of_robot.keys()):
			if self.world_observations[inventory_of_robot[i]] == 'planks':
				alist.append((e, self.world_observations[i], self.world_observations['InventorySlot_' + str(e) + '_variant']))
		return alist

	def find_stone_inventory_slot(self):
		""" 
		Finds the slots in the robot's hotbar that contain stone.

		Params:

		Returns:
		alist(int array): the slots in the robot's hotbar that contain stone.
		"""
		inventory_of_robot = {'Hotbar_0_size': 'Hotbar_0_item', 'Hotbar_1_size': 'Hotbar_1_item', 'Hotbar_2_size': 'Hotbar_2_item',
							  'Hotbar_3_size': 'Hotbar_3_item',
							  'Hotbar_4_size': 'Hotbar_4_item', 'Hotbar_5_size': 'Hotbar_5_item', 'Hotbar_6_size': 'Hotbar_6_item',
							  'Hotbar_7_size': 'Hotbar_7_item',
							  'Hotbar_8_size': 'Hotbar_8_item'}

		alist = []
		for e, i in enumerate(inventory_of_robot.keys()):
			if self.world_observations[inventory_of_robot[i]] == 'stone':
				alist.append((e, self.world_observations[i], self.world_observations['InventorySlot_' + str(e) + '_variant']))
		return alist

	def find_sticks_inventory_slot(self):
		"""
		Finds a slot in the robot's hotbar that contains sticks.

		Params:

		Returns:

		int: the first robot's hotbar slot that contains sticks. 
		"""
		inventory_of_robot = {'Hotbar_0_size': 'Hotbar_0_item', 'Hotbar_1_size': 'Hotbar_1_item', 'Hotbar_2_size': 'Hotbar_2_item',
							  'Hotbar_3_size': 'Hotbar_3_item',
							  'Hotbar_4_size': 'Hotbar_4_item', 'Hotbar_5_size': 'Hotbar_5_item', 'Hotbar_6_size': 'Hotbar_6_item',
							  'Hotbar_7_size': 'Hotbar_7_item',
							  'Hotbar_8_size': 'Hotbar_8_item'}

		for e, i in enumerate(inventory_of_robot.keys()):
			if self.world_observations[inventory_of_robot[i]] == 'stick':
				return e, self.world_observations[i]

	def find_empty_chest_slot(self):
		""" 
		Finds the first empty slot in the resources chest.

		Params:

		Returns:
		int: the first empty slot in the resources chest.
		"""
		inventory_keys = {'container.chestSlot_0_size': 'container.chestSlot_0_item',
						  'container.chestSlot_1_size': 'container.chestSlot_1_item',
						  'container.chestSlot_2_size': 'container.chestSlot_2_item',
						  'container.chestSlot_3_size': 'container.chestSlot_3_item',
						  'container.chestSlot_4_size': 'container.chestSlot_4_item',
						  'container.chestSlot_5_size': 'container.chestSlot_5_item',
						  'container.chestSlot_6_size': 'container.chestSlot_6_item',
						  'container.chestSlot_7_size': 'container.chestSlot_7_item',
						  'container.chestSlot_8_size': 'container.chestSlot_8_item',
						  'container.chestSlot_9_size': 'container.chestSlot_9_item',
						  'container.chestSlot_10_size': 'container.chestSlot_10_item',
						  'container.chestSlot_11_size': 'container.chestSlot_11_item',
						  'container.chestSlot_12_size': 'container.chestSlot_12_item',
						  'container.chestSlot_13_size': 'container.chestSlot_13_item',
						  'container.chestSlot_14_size': 'container.chestSlot_14_item',
						  'container.chestSlot_15_size': 'container.chestSlot_15_item',
						  'container.chestSlot_16_size': 'container.chestSlot_16_item',
						  'container.chestSlot_17_size': 'container.chestSlot_17_item',
						  'container.chestSlot_18_size': 'container.chestSlot_18_item',
						  'container.chestSlot_19_size': 'container.chestSlot_19_item',
						  'container.chestSlot_20_size': 'container.chestSlot_20_item',
						  'container.chestSlot_21_size': 'container.chestSlot_21_item',
						  'container.chestSlot_22_size': 'container.chestSlot_22_item',
						  'container.chestSlot_23_size': 'container.chestSlot_23_item',
						  'container.chestSlot_24_size': 'container.chestSlot_24_item',
						  'container.chestSlot_25_size': 'container.chestSlot_25_item',
						  'container.chestSlot_26_size': 'container.chestSlot_26_item'}
		for e, i in enumerate(inventory_keys):
			if i in self.world_observations.keys():
				if self.world_observations[i] == 0:
					return e

	def find_axe_slot_in_chest(self, type):
		inventory_keys = {'container.chestSlot_0_size': 'container.chestSlot_0_item',
						  'container.chestSlot_1_size': 'container.chestSlot_1_item',
						  'container.chestSlot_2_size': 'container.chestSlot_2_item',
						  'container.chestSlot_3_size': 'container.chestSlot_3_item',
						  'container.chestSlot_4_size': 'container.chestSlot_4_item',
						  'container.chestSlot_5_size': 'container.chestSlot_5_item',
						  'container.chestSlot_6_size': 'container.chestSlot_6_item',
						  'container.chestSlot_7_size': 'container.chestSlot_7_item',
						  'container.chestSlot_8_size': 'container.chestSlot_8_item',
						  'container.chestSlot_9_size': 'container.chestSlot_9_item',
						  'container.chestSlot_10_size': 'container.chestSlot_10_item',
						  'container.chestSlot_11_size': 'container.chestSlot_11_item',
						  'container.chestSlot_12_size': 'container.chestSlot_12_item',
						  'container.chestSlot_13_size': 'container.chestSlot_13_item',
						  'container.chestSlot_14_size': 'container.chestSlot_14_item',
						  'container.chestSlot_15_size': 'container.chestSlot_15_item',
						  'container.chestSlot_16_size': 'container.chestSlot_16_item',
						  'container.chestSlot_17_size': 'container.chestSlot_17_item',
						  'container.chestSlot_18_size': 'container.chestSlot_18_item',
						  'container.chestSlot_19_size': 'container.chestSlot_19_item',
						  'container.chestSlot_20_size': 'container.chestSlot_20_item',
						  'container.chestSlot_21_size': 'container.chestSlot_21_item',
						  'container.chestSlot_22_size': 'container.chestSlot_22_item',
						  'container.chestSlot_23_size': 'container.chestSlot_23_item',
						  'container.chestSlot_24_size': 'container.chestSlot_24_item',
						  'container.chestSlot_25_size': 'container.chestSlot_25_item',
						  'container.chestSlot_26_size': 'container.chestSlot_26_item'}
		for e, i in enumerate(inventory_keys):
			if i in self.world_observations.keys():
				if self.world_observations[inventory_keys[i]] == type:
					return e

	def find_chest_slot(self, item):
		"""
		Finds where in the resources chest the specified item type is located.

		Params:

		item(str): the type of item to find.

		Returns:

		int: the location of the specified item type.
		"""
		inventory_keys = {'container.chestSlot_0_size': 'container.chestSlot_0_item',
						  'container.chestSlot_1_size': 'container.chestSlot_1_item',
						  'container.chestSlot_2_size': 'container.chestSlot_2_item',
						  'container.chestSlot_3_size': 'container.chestSlot_3_item',
						  'container.chestSlot_4_size': 'container.chestSlot_4_item',
						  'container.chestSlot_5_size': 'container.chestSlot_5_item',
						  'container.chestSlot_6_size': 'container.chestSlot_6_item',
						  'container.chestSlot_7_size': 'container.chestSlot_7_item',
						  'container.chestSlot_8_size': 'container.chestSlot_8_item',
						  'container.chestSlot_9_size': 'container.chestSlot_9_item',
						  'container.chestSlot_10_size': 'container.chestSlot_10_item',
						  'container.chestSlot_11_size': 'container.chestSlot_11_item',
						  'container.chestSlot_12_size': 'container.chestSlot_12_item',
						  'container.chestSlot_13_size': 'container.chestSlot_13_item',
						  'container.chestSlot_14_size': 'container.chestSlot_14_item',
						  'container.chestSlot_15_size': 'container.chestSlot_15_item',
						  'container.chestSlot_16_size': 'container.chestSlot_16_item',
						  'container.chestSlot_17_size': 'container.chestSlot_17_item',
						  'container.chestSlot_18_size': 'container.chestSlot_18_item',
						  'container.chestSlot_19_size': 'container.chestSlot_19_item',
						  'container.chestSlot_20_size': 'container.chestSlot_20_item',
						  'container.chestSlot_21_size': 'container.chestSlot_21_item',
						  'container.chestSlot_22_size': 'container.chestSlot_22_item',
						  'container.chestSlot_23_size': 'container.chestSlot_23_item',
						  'container.chestSlot_24_size': 'container.chestSlot_24_item',
						  'container.chestSlot_25_size': 'container.chestSlot_25_item',
						  'container.chestSlot_26_size': 'container.chestSlot_26_item'}
		alist = []
		for e, i in enumerate(inventory_keys):
			if i in self.world_observations.keys():
				if self.world_observations[inventory_keys[i]] == item:
					alist.append((e, self.world_observations[i], self.world_observations['container.chestSlot_' + str(e) + '_variant']))
		return alist

	def get_resources_from_chest(self, item, counter=0):
		sleep(.2)
		self.get_robot_inventory()
		if counter >= 5:
			return
		if item == 'wood':
			counter += 1
			if self.get_number_of_robot_wood() > 0:
				return
			else:
				# NOTE: SHOULD NEVER GO INTO THIS BLOCK
				self.go_to_chest()
				sleep(.1)
				empty_inventory_slot = self.find_empty_inventory_slot()
				wood_chest_slot = self.find_chest_slot('log')
				self._agent.sendCommand("swapInventoryItems chest:" + str(wood_chest_slot) + " inventory:" + str(empty_inventory_slot))
				sleep(.3)
				wood_inventory_slot, wood_quantity = self.find_wood_inventory_slot()
				empty_inventory_slot = self.find_empty_inventory_slot()
				if wood_quantity > 1:
					self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(empty_inventory_slot) + " log 1")
					sleep(.1)
					self._agent.sendCommand(
						"chat /replaceitem entity Robot slot.hotbar." + str(wood_inventory_slot) + " log " + str(wood_quantity - 1))
					sleep(.1)
					self._agent.sendCommand("swapInventoryItems chest:" + str(wood_chest_slot) + " inventory:" + str(wood_inventory_slot))
				else:
					self._agent.sendCommand("swapInventoryItems chest:" + str(wood_chest_slot) + " inventory:" + str(wood_inventory_slot))


		elif item == 'planks':
			counter += 1
			if self.get_number_of_robot_wood() > 0:
				if self.get_number_of_robot_planks() > 5:
					return
				else:
					self._agent.sendCommand("craft planks")
					return
			else:
				self.get_resources_from_chest('wood', counter)
				self.get_resources_from_chest('planks', counter)

		elif item == 'stick':
			counter += 1
			if self.get_number_of_robot_planks() >= 3 or self.get_number_of_robot_wood() > 0:
				if self.get_number_of_robot_sticks() > 5:
					return
				else:
					self._agent.sendCommand("craft stick")
					return
			else:
				self.get_resources_from_chest('wood', counter)
				self.get_resources_from_chest('planks', counter)
				self.get_resources_from_chest('stick', counter)


		elif item == 'wooden_axe':
			counter += 1
			if self.get_number_of_robot_planks() >= 3 and self.get_number_of_robot_sticks() > 2:
				return
			else:
				self.get_resources_from_chest('planks', counter)
				self.get_resources_from_chest('stick', counter)
				self.get_resources_from_chest('wooden_axe', counter)
				self._agent.sendCommand("use 0")

	def do_ours(self, action_id, cur_time=0):
		"""
		function that executes action
		"""
		action = action_id
		self.cur_time = cur_time
		if self._turn_based:
			if self._turn.can_play:
				self._agent.sendCommand(str(action), str(self._turn.key))
				self._turn.has_played = True
				self._previous_action = action
				self._action_count += 1
		else:
			self.per_item_dict = pickle.load(open('current_per_item_dict.pkl', 'rb'))

			self.get_robot_inventory()

			if action == 'chop_birch_wood':
				self.chop_wood(type='birch')
				self.transfer_resources_to_chest()
				# self.put_resources_in_chest()
				self._previous_action = action
				self._action_count += 1

			elif action == 'chop_jungle_wood':
				self.chop_wood(type='jungle')
				self.transfer_resources_to_chest()
				self._previous_action = action
				self._action_count += 1

			elif action == 'mine_stone':
				self.mine_stone(type='stone')
				self.transfer_resources_to_chest()
				self._previous_action = action
				self._action_count += 1

			elif action == 'mine_cobblestone':
				self.mine_stone(type='cobblestone')
				self.transfer_resources_to_chest()
				self._previous_action = action
				self._action_count += 1

			elif action == 'craft_axe':
				# self.get_resources_from_chest('wooden_axe')
				self.craft_axe()
				self._agent.sendCommand("use 0")
				sleep(self.craft_delay)
				self._previous_action = action
				self._action_count += 1

			elif action == 'craft_pickaxe':
				# self.get_resources_from_chest('wooden_axe')
				self.craft_axe(axe_type='wooden_pickaxe')
				sleep(self.craft_delay)
				self._agent.sendCommand("use 0")
				self._previous_action = action
				self._action_count += 1

			elif action == 'put_resources_in_chest':
				self.put_resources_in_chest()
				self._agent.sendCommand("use 0")
				sleep(self.store_resources_delay)
				self._previous_action = action
				self._action_count += 1

			elif action == 'craft_birch_planks':
				# self.get_resources_from_chest('planks')
				self.craft_planks(type='birch')
				self.transfer_resources_to_chest()
				sleep(self.craft_delay)
				self._previous_action = action
				self._action_count += 1

			elif action == 'craft_jungle_planks':
				# self.get_resources_from_chest('planks')
				self.craft_planks(type='jungle')
				self.transfer_resources_to_chest()
				sleep(self.craft_delay)
				self._previous_action = action
				self._action_count += 1

			elif action == 'craft_sticks' or action == 'craft_stick':
				# self.get_resources_from_chest('stick')
				self.craft_sticks()
				self.transfer_resources_to_chest()
				sleep(self.craft_delay)

				self._previous_action = action
				self._action_count += 1
			else:
				print(action, 'misssinggggggggg')

		self._await_next_obs()
		return self.state, sum([reward.getValue() for reward in self._world.rewards]), self.done

	def pull_resources_from_chest(self):

		wood_info = self.find_chest_slot('log')
		if len(wood_info) == 0:
			return
		for wood_chest_slot, wood_quantity, wood_variant in wood_info:
			empty_inventory_slot = self.find_empty_inventory_slot()
			empty_chest_slot = self.find_empty_chest_slot()
			if wood_quantity >= 2 and wood_variant == 'birch':
				mod = ' 2'
				self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(empty_inventory_slot) + " log " + str(2) + mod)
				if wood_quantity == 2:
					self._agent.sendCommand(
						"chat /replaceitem block 6 4 11 slot.container." + str(wood_chest_slot) + " air")
				else:
					self._agent.sendCommand(
						"chat /replaceitem block 6 4 11 slot.container." + str(wood_chest_slot) + " log " + str(wood_quantity - 2) + mod)
				# self._agent.sendCommand(
				# 	"chat /replaceitem block 6 4 11 slot.container." + str(26) + " log " + str(
				# 		self.chest_unknown_birch_wood_quant - 2) + mod)
				if wood_chest_slot == 26:
					self.chest_unknown_birch_wood_quant = self.chest_unknown_birch_wood_quant - 2
				sleep(0.1)
				return
			elif wood_quantity >= 2 and wood_variant == 'jungle':
				mod = ' 3'
				self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(empty_inventory_slot) + " log " + str(2) + mod)
				self._agent.sendCommand(
					"chat /replaceitem block 6 4 11 slot.container." + str(wood_chest_slot) + " log " + str(
						self.chest_unknown_jungle_wood_quant - 2) + mod)
				if wood_chest_slot == 23:
					self.chest_unknown_jungle_wood_quant = self.chest_unknown_jungle_wood_quant - 2
				sleep(0.1)
				return

	def transfer_resources_to_chest(self):
		"""
		Copy of put_resources in chest without motion
		:return:
		"""
		# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar.5 stone 6 6")
		# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar.3 cobblestone 6")
		self.put_resources_in_chest()
		# robot_info, robot_index = self.get_robot_info()

		# single_point = [robot_info['x'] + 30, robot_info['z'] + 30]
		# dist = np.linalg.norm(np.array(single_point) - np.array(self.chest_location))
		#
		# angle_to_chest = angle_clockwise((np.array(self.chest_location)) - np.array(single_point), np.array(
		# 	[np.cos((robot_info['yaw']) % 360 * np.pi / 180), np.sin((robot_info['yaw']) % 360 * np.pi / 180)])) - 90
		# # agent_yaw = (robot_info['yaw'] + 180) % 360
		# self._agent.sendCommand("setYaw " + str(angle_to_chest + (robot_info['yaw'] % 360)))
		#
		# pitch = 90 - np.arctan2(dist, 1.5) * 180 / np.pi
		# self._agent.sendCommand("setPitch " + str(pitch))
		# sleep(.2)

		if self.world_observations['LineOfSight']['type'] == 'chest' and self.find_empty_chest_slot() != None:

			self.get_robot_inventory()
			if self.get_number_of_robot_wood() > 2:
				# log 1 2 is 1 log of variant 2 (birch)
				wood_info = self.find_wood_inventory_slot()
				for wood_chest_slot, wood_quantity, wood_variant in wood_info:
					empty_inventory_slot = self.find_empty_inventory_slot()
					empty_chest_slot = self.find_empty_chest_slot()
					if wood_variant == 'birch':
						mod = ' 2'
					else:
						mod = ' 3'
					if wood_quantity > 2:
						# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " birch_log: 2")
						self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " log " + str(2) + mod)
						# self._agent.sendCommand("swapInventoryItems chest:" + str(empty_chest_slot) + " inventory:" + str(wood_chest_slot))
						self._agent.sendCommand(
							"chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " log " + str(wood_quantity - 2) + mod)

					sleep(0.1)

			if self.get_number_of_robot_planks() > 2:

				plank_info = self.find_plank_inventory_slot()
				for plank_chest_slot, plank_quantity, plank_variant in plank_info:
					empty_inventory_slot = self.find_empty_inventory_slot()
					empty_chest_slot = self.find_empty_chest_slot()
					if plank_variant == 'birch':
						mod = ' 2'
					else:
						mod = ' 3'
					if plank_quantity > 2:
						# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " birch_log: 2")
						self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(plank_chest_slot) + " planks " + str(2) + mod)
						# self._agent.sendCommand("swapInventoryItems chest:" + str(empty_chest_slot) + " inventory:" + str(wood_chest_slot))
						self._agent.sendCommand(
							"chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " planks " + str(plank_quantity - 2) + mod)
					sleep(0.1)

			if self.get_number_of_robot_sticks() > 2:
				empty_chest_slot = self.find_empty_chest_slot()
				sticks_chest_slot, sticks_quantity = self.find_sticks_inventory_slot()
				self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(sticks_chest_slot) + " stick " + str(2))
				self._agent.sendCommand(
					"chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " stick " + str(sticks_quantity - 2))
				sleep(0.1)

			if self.get_number_of_robot_stone() > 2:
				stone_info = self.find_stone_inventory_slot()
				for stone_chest_slot, stone_quantity, stone_variant in stone_info:
					empty_chest_slot = self.find_empty_chest_slot()
					if stone_quantity > 2:
						# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " birch_log: 2")
						self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(stone_chest_slot) + " stone " + str(2) + ' 6')
						# self._agent.sendCommand("swapInventoryItems chest:" + str(empty_chest_slot) + " inventory:" + str(wood_chest_slot))
						self._agent.sendCommand(
							"chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " stone " + str(stone_quantity - 2) + ' 6')
						sleep(1)

			if self.get_number_of_robot_cobblestone() > 2:
				empty_chest_slot = self.find_empty_chest_slot()
				cobblestone_chest_slot, cobblestone_quantity = self.find_cobblestone_inventory_slot()
				if cobblestone_chest_slot > 9:
					self._agent.sendCommand(
						"chat /replaceitem entity Robot slot.inventory." + str(cobblestone_chest_slot - 9) + " cobblestone " + str(2))
				else:
					self._agent.sendCommand(
						"chat /replaceitem entity Robot slot.hotbar." + str(cobblestone_chest_slot) + " cobblestone " + str(2))
				self._agent.sendCommand(
					"chat /replaceitem block 6 4 11 slot.container." + str(empty_chest_slot) + " cobblestone " + str(cobblestone_quantity - 2))
				sleep(1)
			self.update_chest()
			if not self.does_robot_have_one_pickaxe():
				if self.chest_state['wooden_pickaxe'] >= 1:
					slot = self.find_axe_slot_in_chest('wooden_pickaxe')
					self._agent.sendCommand("chat /give Robot " + str('wooden_pickaxe') + " 1 48")
					self._agent.sendCommand(
						"chat /replaceitem block 6 4 11 slot.container." + str(slot) + " air")
			if not self.does_robot_have_one_axe():
				if self.chest_state['wooden_axe'] >= 1:
					slot = self.find_axe_slot_in_chest('wooden_axe')
					self._agent.sendCommand("chat /give Robot " + str('wooden_axe') + " 1 48")
					self._agent.sendCommand(
						"chat /replaceitem block 6 4 11 slot.container." + str(slot) + " air")

			self.update_chest()
			self.save_chest_information()

		else:
			# YOU COULD NOT SEE THE CHEST
			self.get_robot_inventory()
			if self.get_number_of_robot_wood() > 2:
				# log 1 2 is 1 log of variant 2 (birch)
				wood_info = self.find_wood_inventory_slot()
				for wood_chest_slot, wood_quantity, wood_variant in wood_info:
					empty_inventory_slot = self.find_empty_inventory_slot()
					empty_chest_slot = self.find_empty_chest_slot()
					if wood_variant == 'birch':
						mod = ' 2'
					else:
						mod = ' 3'
					if wood_quantity > 2:
						# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " birch_log: 2")
						self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " log " + str(2) + mod)
						# self._agent.sendCommand("swapInventoryItems chest:" + str(empty_chest_slot) + " inventory:" + str(wood_chest_slot))
						if wood_variant == "birch":
							self._agent.sendCommand(
								"chat /replaceitem block 6 4 11 slot.container." + str(26) + " log " + str(
									self.chest_unknown_birch_wood_quant + wood_quantity - 2) + mod)
							self.chest_unknown_birch_wood_quant = self.chest_unknown_birch_wood_quant + wood_quantity - 2
						else:
							self._agent.sendCommand(
								"chat /replaceitem block 6 4 11 slot.container." + str(23) + " log " + str(
									self.chest_unknown_jungle_wood_quant + wood_quantity - 2) + mod)
							self.chest_unknown_jungle_wood_quant = self.chest_unknown_jungle_wood_quant + wood_quantity - 2
					sleep(0.1)

			if self.get_number_of_robot_planks() > 2:

				plank_info = self.find_plank_inventory_slot()
				for plank_chest_slot, plank_quantity, plank_variant in plank_info:
					empty_inventory_slot = self.find_empty_inventory_slot()
					empty_chest_slot = self.find_empty_chest_slot()
					if plank_variant == 'birch':
						mod = ' 2'
					else:
						mod = ' 3'
					if plank_quantity > 2:
						# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " birch_log: 2")
						self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(plank_chest_slot) + " planks " + str(2) + mod)
						# self._agent.sendCommand("swapInventoryItems chest:" + str(empty_chest_slot) + " inventory:" + str(wood_chest_slot))
						if plank_variant == "birch":
							self._agent.sendCommand(
								"chat /replaceitem block 6 4 11 slot.container." + str(25) + " planks " + str(
									self.chest_unknown_birch_plank_quant + plank_quantity - 2) + mod)
							self.chest_unknown_birch_plank_quant = self.chest_unknown_birch_plank_quant + plank_quantity - 2
						else:
							self._agent.sendCommand(
								"chat /replaceitem block 6 4 11 slot.container." + str(22) + " planks " + str(
									self.chest_unknown_jungle_plank_quant + plank_quantity - 2) + mod)
							self.chest_unknown_jungle_plank_quant = self.chest_unknown_jungle_plank_quant + plank_quantity - 2

					sleep(0.1)

			if self.get_number_of_robot_sticks() > 2:
				empty_chest_slot = self.find_empty_chest_slot()
				sticks_chest_slot, sticks_quantity = self.find_sticks_inventory_slot()
				self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(sticks_chest_slot) + " stick " + str(2))
				self._agent.sendCommand(
					"chat /replaceitem block 6 4 11 slot.container." + str(24) + " stick " + str(
						self.chest_unknown_stick_quant + sticks_quantity - 2))
				self.chest_unknown_stick_quant = self.chest_unknown_stick_quant + sticks_quantity - 2
				sleep(0.1)

			if self.get_number_of_robot_stone() > 2:
				stone_info = self.find_stone_inventory_slot()
				for stone_chest_slot, stone_quantity, stone_variant in stone_info:
					empty_chest_slot = self.find_empty_chest_slot()
					if stone_quantity > 2:
						# self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(wood_chest_slot) + " birch_log: 2")
						self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(stone_chest_slot) + " stone " + str(2) + ' 6')
						# self._agent.sendCommand("swapInventoryItems chest:" + str(empty_chest_slot) + " inventory:" + str(wood_chest_slot))
						self._agent.sendCommand(
							"chat /replaceitem block 6 4 11 slot.container." + str(21) + " stone " + str(
								self.chest_unknown_stone_quant + stone_quantity - 2) + ' 6')
						self.chest_unknown_stone_quant = self.chest_unknown_stone_quant + stone_quantity - 2
						sleep(1)

			if self.get_number_of_robot_cobblestone() > 2:
				empty_chest_slot = self.find_empty_chest_slot()
				cobblestone_chest_slot, cobblestone_quantity = self.find_cobblestone_inventory_slot()
				self._agent.sendCommand("chat /replaceitem entity Robot slot.hotbar." + str(cobblestone_chest_slot) + " cobblestone " + str(2))
				self._agent.sendCommand(
					"chat /replaceitem block 6 4 11 slot.container." + str(20) + " cobblestone " + str(
						self.chest_unknown_cobblestone_quant + cobblestone_quantity - 2))
				self.chest_unknown_cobblestone_quant = self.chest_unknown_cobblestone_quant + cobblestone_quantity - 2
				sleep(1)

			self.update_chest()
			self.save_chest_information()

	def do(self, action_id):
		assert 0 <= action_id <= self.available_actions, \
			"action %d is not valid (should be in [0, %d[)" % (action_id,
															   self.available_actions)
		action = self._actions[action_id]
		assert isinstance(action, six.string_types)

		if self._turn_based:
			if self._turn.can_play:
				self._agent.sendCommand(str(action), str(self._turn.key))
				self._turn.has_played = True
				self._previous_action = action
				self._action_count += 1
		else:
			self._agent.sendCommand(action)
			self._previous_action = action
			self._action_count += 1

		self._await_next_obs()
		return self.state, sum([reward.getValue() for reward in self._world.rewards]), self.done

	def reset(self):
		super(MalmoEnvironment, self).reset()

		if self._force_world_reset:
			self._mission.forceWorldReset()

		self._world = None
		self._world_obs = None
		self._previous_action = None
		self._action_count = 0
		self._turn = TurnState()
		self._end_result = None

		# Wait for the server (role = 0) to start
		sleep(.5)

		for i in range(MalmoEnvironment.MAX_START_MISSION_RETRY):
			try:
				self._agent.startMission(self._mission,
										 self._clients,
										 self._recorder,
										 self._role,
										 self._exp_name)
				break
			except Exception as e:
				if i == MalmoEnvironment.MAX_START_MISSION_RETRY - 1:
					raise Exception("Unable to connect after %d tries %s" %
									(self.MAX_START_MISSION_RETRY, e))
				else:
					sleep(log(i + 1) + 1)

		# wait for mission to begin
		self._await_next_obs()
		return self.state

	def _await_next_obs(self):
		"""
		Ensure that an update to the world state is received
		:return:
		"""
		# Wait until we have everything we need
		current_state = self._agent.peekWorldState()
		while not self.is_valid(current_state) or not self._ready_to_act(current_state):

			if current_state.has_mission_begun and not current_state.is_mission_running:
				if not current_state.is_mission_running and len(current_state.mission_control_messages) > 0:
					# Parse the mission ended message:
					mission_end_tree = xml.etree.ElementTree.fromstring(current_state.mission_control_messages[-1].text)
					ns_dict = {"malmo": "http://ProjectMalmo.microsoft.com"}
					hr_stat = mission_end_tree.find("malmo:HumanReadableStatus", ns_dict).text
					self._end_result = hr_stat
				break

			# Peek fresh world state from socket
			current_state = self._agent.peekWorldState()

		# Flush current world as soon as we have the entire state
		self._world = self._agent.getWorldState()

		if self._world.is_mission_running:
			new_world = json.loads(self._world.observations[-1].text)
			if new_world is not None:
				self._world_obs = new_world

		# Update video frames if any
		if hasattr(self._world, 'video_frames') and len(self._world.video_frames) > 0:
			self._last_frame = self._world.video_frames[-1]

	def is_valid(self, world_state):
		"""
		Check whether the provided world state is valid.
		@override to customize checks
		"""

		# observation cannot be empty
		return world_state is not None \
			   and world_state.has_mission_begun \
			   and len(world_state.observations) > 0
# and (len(world_state.rewards) > 0 or self._action_count == 0)
