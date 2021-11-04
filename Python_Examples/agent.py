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
import os

x = 0
y = 600
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)
import sys
from collections import Iterable
from time import sleep
import numpy as np
from PIL import Image
import pygame

from malmopy.visualization import Visualizable


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


class BaseAgent(Visualizable):
	"""
	Represents an agent that interacts with an environment
	"""

	def __init__(self, name, nb_actions, visualizer=None):
		assert nb_actions > 0, 'Agent should at least have 1 action (got %d)' % nb_actions

		super(BaseAgent, self).__init__(visualizer)

		self.name = name
		self.nb_actions = nb_actions

	def act(self, new_state, reward, done, is_training=False):
		raise NotImplementedError()

	def save(self, out_dir):
		pass

	def load(self, out_dir):
		pass

	def inject_summaries(self, idx):
		pass


class AdaptiveAgent(BaseAgent):
	"""
	An agent that selects actions based on the adaptive policies found in the packet
	"""

	def __init__(self, name, env, nb_actions, participant_number='tmp', delay_between_action=3, visualizer=None, which_env=1,factor=0,display_code='1', start_time=None):
		super(AdaptiveAgent, self).__init__(name, nb_actions, visualizer)
		print('AdaptiveAgent AGENT CHOSEN')
		self._delay = delay_between_action
		self.start_time = start_time
		self.state = None
		self.env = env
		self.env_board = None
		self.phase = 0
		self.curr_house_andesite = 0
		self.human_inventory = None
		self.house_location = [34.0, 33.0]  # center of house
		self.crafting_table_location = [35.5, 41.5]
		self.chest_location = [36.5, 41.5]
		# which random environment is the user playing in
		self.which_env = which_env
		print('env is ', self.which_env)

		self.chest_state = {'log': 0, 'wooden_axe': 0, 'planks': 0, 'wooden_pickaxe': 0, 'cobblestone': 0, 'stick': 0}

		self.prev_phase = 0
		self.cur_time = 0

		# THRESHOLDS
		self.tree_threshold = 5
		self.stone_threshold = 6.5
		self.house_threshold = 4.5
		self.house_andesite_threshold = 10
		self.craft_threshold = 4
		self.craft_wood_threshold = 3
		self.robot_resources_threshold = 10
		self.plank_threshold = 5
		self.time_threshold = 5
		self.wood_threshold = 5


		# Tracking policy phases and show tree phases
		self.old_tree_phase = -1
		self.tree_phase = -1
		self.time_offset = 0
		self.sleep_time = 250


		self.score_threshold = 2

		self.possible_human_behaviors = ['chopping', 'crafting', 'mining', 'building']
		self.possible_robot_behaviors = ['chop_wood_any', 'chop_wood_birch', 'chop_wood_jungle',
										 'mine_cobble_stone', 'mine_andesite_stone',
										 'craft_axe', 'craft_pickaxe', 'put_resources_in_chest', 'craft_planks',
										 'craft stick']

		pygame.init()

		self.screen = pygame.display.set_mode((1800, 500))  # Whatever resolution you want.
		# If factor = 0, blank display
		# if factor = 1, only showing human tree
		# if factor = 2, only showing robot tree
		# if factor = 3, we show both
		self.factor = factor
		# if display code is 0, we are in SA level 0. ELSE SA=2-3
		self.display_code = display_code
		self.robot_resources_tracker = {}
		# chest info saved by malmo.py
		self.build_progress_tracker = {}
		self.human_resource_tracker = {}
		self.robot_resources_verbose_tracker = {}
		self.chest_state_resources_verbose_tracker = {}

		# change for each participant

		self.participant_number = participant_number
		if not os.path.exists(str(participant_number) + "A"):
			os.mkdir(str(participant_number) + "A")


		if not os.path.exists(str(participant_number) + "B"):
			os.mkdir(str(participant_number) + "B")


		if not os.path.exists(str(participant_number) + "C"):
			os.mkdir(str(participant_number) + "C")

		if not os.path.exists(str(self.participant_number) + "E"):
			os.mkdir(str(self.participant_number) + "E")

		if not os.path.exists(str(self.participant_number) + "F"):
			os.mkdir(str(self.participant_number) + "F")


		if self.display_code == '1' and self.factor == 0:
			self.participant_number = participant_number + "A"
		if self.display_code == '0' and self.factor == 0:
			self.participant_number = participant_number + "A"
		if self.display_code == '0' and self.factor == 3:
			self.participant_number = participant_number + "B"
		if self.display_code == '1' and self.factor == 3:
			self.participant_number = participant_number + "C"
		if self.display_code == '0' and self.factor == 2:
			self.participant_number = participant_number + "E"
		if self.display_code == '1' and self.factor == 2:
			self.participant_number = participant_number + "F"



	def act(self, new_state, reward, done, is_training=False, cur_time=0):

		# per_action_delay
		if self._delay > 0:
			from time import sleep
			sleep(self._delay)

		# takes care of time offsets due to pausing
		cur_time = cur_time - self.time_offset
		print('cur_time is ', cur_time)

		self.cur_time = cur_time

		self.state = new_state
		self.load_in_new_human_inventory()

		# set phases
		human_behavior, human_code = self.determine_human_behavior()
		self.set_phase(cur_time)
		# self.tree_phase = self.is_cur_time_near_switch_times(cur_time, self.tree_phase)
		#
		# # show tree pausing
		# if self.tree_phase != self.old_tree_phase:
		# 	self.old_tree_phase = self.tree_phase

		# get human behavior, and action based on trees
		macro_action = self.determine_robot_behavior(human_behavior, human_code)

		# if phase changed, save and put resources in chest
		if self.phase != self.prev_phase:
			self.prev_phase = self.phase

		return macro_action

	def play_with_house_thresholds(self):
		human_info = None
		for i in self.env.world_observations['entities']:
			if i['name'] == 'Human':
				human_info = i
		single_point = [human_info['x'] + 30, human_info['z'] + 30]
		print('your location: ', single_point)
		print('house location: ', self.house_location)
		dist = np.linalg.norm(np.array(single_point) - np.array(self.house_location))
		print('distance to house: ', dist)
		dx = max(abs(single_point[0] - self.house_location[0]) - 6 / 2, 0)
		dy = max(abs(single_point[1] - self.house_location[1]) - 9 / 2, 0)
		print(max(dx,dy))
		if dist < self.house_threshold:
			print('close to house')
			return True
		else:
			print('NOT close to house')
			return False

	def play_with_crafting_table_thresholds(self):
		human_info = None
		for i in self.env.world_observations['entities']:
			if i['name'] == 'Human':
				human_info = i
		single_point = [human_info['x'] + 30, human_info['z'] + 30]
		print('your location: ', single_point)
		dist = np.linalg.norm(np.array(single_point) - np.array(self.crafting_table_location))
		print('distance to table: ', dist)
		if dist < self.craft_threshold:
			print('close to table')
			return True
		else:
			print('NOT close to table')
			return False

	def play_with_mining_thresholds(self):
		human_info = None
		for i in self.env.world_observations['entities']:
			if i['name'] == 'Human':
				human_info = i
		single_point = [human_info['x'] + 30, human_info['z'] + 30]
		print('your location: ', single_point)
		dist1 = self.get_euclidean_distance_to_stone(single_point)
		dist2 = self.get_euclidean_distance_to_cobblestone(single_point)
		print('distance to stone: ', min(dist1))
		print('distance to cobble: ', min(dist2))
		if (dist1 < self.stone_threshold).any() or (dist2 < self.stone_threshold).any():
			print('close to stone')
			return True
		else:
			print('NOT close to stone')
			return False

	def save_inventory(self, cur_time):
		"""
		saves inventory of robot and human in a list
		"""
		inventorys = []  # human, robot
		self.get_robot_inventory()
		self.load_in_new_human_inventory()
		inventorys.append(self.robot_inventory)
		inventorys.append(self.human_inventory)
		save_pickle(os.curdir + '/' + str(self.participant_number), inventorys, str(cur_time) + 'inventory.pkl')

	def is_cur_time_near_switch_times(self, cur_time, phase):
		# check if you are passed 45ish and that you are not in another phase. If so, change phase
		score = self.check_house_progress()
		if 14 - self.score_threshold <= score:
			if phase == 0 or phase == 1 or phase == 2 or phase == 3 or phase == 4:
				pass
			else:
				return 0

		if 42 - self.score_threshold <= cur_time:
			if phase == 1 or phase == 2 or phase == 3 or phase == 4:
				pass
			else:
				return 1

		if 70 - self.score_threshold <= cur_time:
			if phase == 2 or phase == 3 or phase == 4:
				pass
			else:
				return 2

		if 98 - self.score_threshold <= cur_time:
			if phase == 3 or phase == 4:
				pass
			else:
				return 3

		if 126 - self.score_threshold <= cur_time:
			if phase == 4:
				pass
			else:
				return 4

		return phase

	def get_number_of_human_wood(self):
		"""
		Gets the number of logs in the humans's inventory.

		Params:

		Returns:
		int: the number of logs in the humans's inventory
		"""
		if 'log' in list(self.human_inventory.keys()):
			return self.human_inventory['log']
		else:
			return 0

	def get_number_of_human_planks(self):
		"""
		Gets the number of planks in the humans's inventory.

		Params:

		Returns:
		int: the number of planks in the humans's inventory.
		"""

		if 'planks' in list(self.human_inventory.keys()):
			return self.human_inventory['planks']
		else:
			return 0

	def get_number_of_human_stone(self):
		"""
		Gets the number of stone in the human's inventory.

		Params:

		Returns:
		int: the number of stone in the human's inventory
		"""
		if 'cobblestone' in list(self.human_inventory.keys()):
			cob = self.human_inventory['cobblestone']
		else:
			cob = 0
		if 'stone' in list(self.human_inventory.keys()):
			sto = self.human_inventory['stone']
		else:
			sto = 0
		return cob + sto

	def get_number_of_robot_wood(self):
		"""
		Gets the number of logs in the robot's inventory.

		Params:

		Returns:
		int: the number of logs in the robot's inventory
		"""
		if 'log' in list(self.robot_inventory.keys()):
			return self.robot_inventory['log']
		else:
			return 0

	def get_number_of_robot_planks(self):
		"""
		Gets the number of planks in the robot's inventory.

		Params:

		Returns:
		int: the number of planks in the robot's inventory.
		"""
		if 'planks' in list(self.robot_inventory.keys()):
			return self.robot_inventory['planks']
		else:
			return 0

	def get_tot_number_of_robot_stone(self):
		"""
		Gets the total number of stone in the robot's inventory.

		Params:

		Returns:
		int: the total number of stone in the robot's inventory. Both cobblestone and other stone.
		"""
		if 'cobblestone' in list(self.robot_inventory.keys()):
			cob = self.robot_inventory['cobblestone']
		else:
			cob = 0
		if 'stone' in list(self.robot_inventory.keys()):
			sto = self.robot_inventory['stone']
		else:
			sto = 0

		return cob + sto

	def get_tot_number_of_chest_stone(self):
		"""
		Gets the total number of stone in the robot's inventory.

		Params:

		Returns:
		int: the total number of stone in the robot's inventory. Both cobblestone and other stone.
		"""
		if 'cobblestone' in list(self.chest_state.keys()):
			cob = self.chest_state['cobblestone']
		else:
			cob = 0
		if 'stone' in list(self.chest_state.keys()):
			sto = self.chest_state['stone']
		else:
			sto = 0

		return cob + sto

	def determine_robot_behavior(self, human_behavior, human_behavior_code='0'):
		display_code = self.display_code
		# takes in human behavior, and determines robot behavior
		self.get_robot_inventory()
		if display_code == '1':
			image_dir = '../minecraft_images/robot_policies/'
		else:
			image_dir = "../minecraft_images/single_action_images/robot_actions/"
		human_behavior, human_behavior_code = self.determine_human_behavior()
		human_info = self.get_human_info()
		single_point = [human_info['x'] + 30, human_info['z'] + 30]

		if self.phase == 0:
			if human_behavior == 'chopping':
				if self.get_number_of_robot_wood() + self.get_number_of_chest_wood() < self.tree_threshold:
					self.display_image(image_dir, '1_1_0', str(3) + '_' + human_behavior_code, 'chop_birch_wood', human_behavior,
									   display_code)
					return 'chop_birch_wood'
				else:

					self.display_image(image_dir, '1_1_1', str(3) + '_' + human_behavior_code, 'craft_axe', human_behavior, display_code)
					return 'craft_axe'
			else:
				if self.get_number_of_robot_wood() < self.tree_threshold:

					self.display_image(image_dir, '1_0_0', str(3) + '_' + human_behavior_code, 'chop_birch_wood', human_behavior,
									   display_code)
					return 'chop_birch_wood'
				else:

					self.display_image(image_dir, '1_0_1', str(3) + '_' + human_behavior_code, 'craft_pickaxe', human_behavior,
									   display_code)
					return 'craft_pickaxe'

		elif self.phase == 1:

			if human_behavior == 'crafting':
				if self.get_number_of_robot_wood() + self.get_number_of_chest_wood() < 4:

					self.display_image(image_dir, '2_1_0', str(3) + '_' + human_behavior_code, 'chop_jungle_wood', human_behavior,
									   display_code)
					return 'chop_jungle_wood'
				else:

					self.display_image(image_dir, '2_1_1', str(3) + '_' + human_behavior_code, 'craft_birch_planks', human_behavior,
									   display_code)
					return 'craft_birch_planks'
			else:
				if self.get_number_of_robot_wood() + self.get_number_of_chest_wood() < 4:
					self.display_image(image_dir, '2_0_0', str(3) + '_' + human_behavior_code, 'chop_birch_wood', human_behavior,
									   display_code)
					return 'chop_birch_wood'
				else:
					self.display_image(image_dir, '2_0_1', str(3) + '_' + human_behavior_code, 'craft_pickaxe', human_behavior,
									   display_code)
					return 'craft_pickaxe'
			# else:
			# 	if human_behavior == "chopping":
			# 		if self.get_number_of_robot_wood() + self.get_number_of_chest_wood() < self.craft_wood_threshold:
			# 			self.display_image(image_dir, '2_1_0', str(self.phase + 1) + '_' + human_behavior_code, 'chop_birch_wood', human_behavior,
			# 							   display_code)
			# 			return 'chop_birch_wood'
			# 		else:
			# 			self.display_image(image_dir, '2_1_1', str(self.phase + 1) + '_' + human_behavior_code, 'craft_birch_planks', human_behavior,
			# 							   display_code)
			# 			return 'craft_birch_planks'
			# 	else:
			# 		if self.get_number_of_robot_wood() < 5:
			# 			return 'chop_birch_wood'
			# 		else:
			# 			return 'craft_pickaxe'

		elif self.phase == 2:
			if self.get_number_of_robot_wood() + self.get_number_of_chest_wood() < self.tree_threshold:
				if self.does_robot_have_pickaxe():
					self.display_image(image_dir, '3_0_1', str(3) + '_' + human_behavior_code, 'mine_stone', human_behavior,
									   display_code)
					return 'mine_stone'
				else:
					self.display_image(image_dir, '3_0_0', str(3) + '_' + human_behavior_code, 'chop_jungle_wood', human_behavior,
									   display_code)
					return 'chop_jungle_wood'
			else:
				if human_behavior == 'mining':
					self.display_image(image_dir, '3_1_1', str(3) + '_' + human_behavior_code, 'craft_pickaxe', human_behavior,
									   display_code)
					return 'craft_pickaxe'
				else:
					self.display_image(image_dir, '3_1_0', str(3) + '_' + human_behavior_code, 'chop_jungle_wood', human_behavior,
									   display_code)
					return 'chop_jungle_wood'


		elif self.phase == 3:

			if self.does_robot_have_pickaxe():
				if self.curr_house_andesite <= 10:
					self.display_image(image_dir, '4_1_1', str(3) + '_' + human_behavior_code, 'mine_stone', human_behavior,
									   display_code)
					return 'mine_stone'
				else:
					self.display_image(image_dir, '4_1_0', str(3) + '_' + human_behavior_code, 'mine_cobblestone', human_behavior,
									   display_code)
					return 'mine_cobblestone'
			else:
				if human_behavior == 'chopping':
					self.display_image(image_dir, '4_0_1', str(3) + '_' + human_behavior_code, 'chop_jungle_wood', human_behavior,
									   display_code)
					return 'chop_jungle_wood'
				else:
					self.display_image(image_dir, '4_0_0', str(3) + '_' + human_behavior_code, 'chop_birch_wood', human_behavior,
									   display_code)
					return 'chop_birch_wood'


		elif self.phase == 4:
			if human_behavior == 'crafting':
				if not self.does_robot_have_pickaxe():
					self.display_image(image_dir, '5_1_0', str(3) + '_' + human_behavior_code, 'chop_jungle_wood', human_behavior,
									   display_code)
					return 'chop_jungle_wood'
				else:
					self.display_image(image_dir, '5_1_1', str(3) + '_' + human_behavior_code, 'mine_cobblestone', human_behavior,
									   display_code)
					return 'mine_cobblestone'
			else:
				if self.get_number_of_robot_wood() + self.get_number_of_chest_wood() < 5:
					self.display_image(image_dir, '5_0_0', str(3) + '_' + human_behavior_code, 'chop_jungle_wood', human_behavior,
									   display_code)
					return 'chop_jungle_wood'
				else:
					self.display_image(image_dir, '5_0_1', str(3) + '_' + human_behavior_code, 'craft_sticks', human_behavior, display_code)
					return 'craft_sticks'

		else:
			print('Entered a phase that is not part of structure')

	def display_image(self, image_dir, img_name, human_behavior_code, robot_behavior, human_behavior, display_code='0'):
		"""
		Displays an image using a pygame window.

		Params:
		image_dir(str): the directory where the image is located
		img_name(str): the name of the image.
		human_behavior_code(int): the current behavior code of the human.

		Returns:
		"""
		if display_code == '1':
			human_img_dir = '../minecraft_images/human_policies/'

			if self.factor == 0:
				# blank display
				pass
			elif self.factor == 1:
				self.screen.fill((0, 0, 0))
				pygame.display.update()
				# display only human policies
				image = pygame.image.load(image_dir + img_name + '.png').convert()
				# image = pygame.transform.scale(image, (900, 500))
				self.screen.blit(image, (900, 0))
				self.text_to_screen(self.screen, 'Robot Inference of Human Action', 0, 0)
				pygame.display.flip()
			elif self.factor == 2:
				self.screen.fill((0, 0, 0))
				pygame.display.update()
				# display both robot and human policies
				# image2 = pygame.image.load(human_img_dir + human_behavior_code + '.png').convert()
				# image2 = pygame.transform.scale(image2, (900, 500))
				image = pygame.image.load(image_dir + img_name + '.png')
				image = pygame.transform.scale(image, (900, 325))
				self.screen.blit(image, (900, 45))
				self.text_to_screen(self.screen, 'Robot Policy', 900, 0)
				pygame.display.flip()
			else:
				# show both
				# pygame.display.set_caption('Pac Man')
				self.screen.fill((0, 0, 0))
				pygame.display.update()
				image = pygame.image.load(image_dir + img_name + '.png')
				image = pygame.transform.scale(image, (900, 325))
				self.screen.blit(image, (900, 45))
				image2 = pygame.image.load(human_img_dir + human_behavior_code + '.png')
				image2 = pygame.transform.scale(image2, (900, 325))
				self.screen.blit(image2, (0, 45))
				font = pygame.font.Font(pygame.font.get_default_font(), 36)

				# now print the text
				self.text_to_screen(self.screen, 'Robot Inference of Human Action', 0, 0)
				self.text_to_screen(self.screen, 'Robot Policy', 900, 0)
				pygame.display.flip()

		elif display_code == '0':

			human_img_dir = '../minecraft_images/single_action_images/human_actions/'

			if self.factor == 0:
				# blank display
				pass
			elif self.factor == 1:
				self.screen.fill((0, 0, 0))
				pygame.display.update()
				# display only human policies
				image = pygame.image.load(image_dir + robot_behavior + '.png').convert()
				# image = pygame.transform.scale(image, (900, 500))
				self.screen.blit(image, (900, 0))
				self.text_to_screen(self.screen, 'Robot Inference of Human Action', 0, 0)
				pygame.display.flip()
			elif self.factor == 2:
				self.screen.fill((0, 0, 0))
				pygame.display.update()
				# display both robot and human policies
				image = pygame.image.load(image_dir + robot_behavior + '.png')
				# image2 = pygame.transform.scale(image2, (900, 500))
				self.text_to_screen(self.screen, 'Robot Policy', 900, 0)
				self.screen.blit(image, (900, 30))
				pygame.display.flip()
			else:
				# show both
				# pygame.display.set_caption('Pac Man')
				self.screen.fill((0, 0, 0))
				pygame.display.update()
				image = pygame.image.load(image_dir + robot_behavior + '.png')
				# image = pygame.transform.scale(image, (900, 500))
				self.screen.blit(image, (900, 30))
				image2 = pygame.image.load(human_img_dir + human_behavior + '.png')
				# image2 = pygame.transform.scale(image2, (900, 500))
				self.screen.blit(image2, (0, 30))
				font = pygame.font.Font(pygame.font.get_default_font(), 36)

				# now print the text
				self.text_to_screen(self.screen, 'Robot Inference of Human Action', 0, 0)
				self.text_to_screen(self.screen, 'Robot Policy', 900, 0)
				pygame.display.flip()

	@staticmethod
	def text_to_screen(screen, text, x, y, size=30,
					   color=(200, 000, 000)):
		"""
		from https://stackoverflow.com/questions/20842801/how-to-display-text-in-pygame
		:param screen:
		:param text:
		:param x:
		:param y:
		:param size:
		:param color:
		:return:
		"""
		text = str(text)
		font = pygame.font.Font(pygame.font.get_default_font(), size)
		text = font.render(text, True, color)
		screen.blit(text, (x, y))

	def get_human_info(self):
		human_info = None
		while human_info is None:
			# print(self.state['entities'])
			for i in self.state['entities']:
				if i['name'] == 'Human':
					human_info = i
		return human_info

	def get_euclidean_distance_to_all_trees(self, single_point):
		# from https://stackoverflow.com/questions/4370975/python-numpy-euclidean-distance-calculation-between-matrices-of-row-vectors/38976555


		trees = np.array(self.per_item_dict['log'])


		dist = (trees - single_point) ** 2
		dist = np.sum(dist, axis=1)
		dist = np.sqrt(dist)
		return dist

	def get_euclidean_distance_to_cobblestone(self, single_point):
		stone = np.array(self.per_item_dict['cobblestone'])

		dist = (stone - single_point) ** 2
		dist = np.sum(dist, axis=1)
		dist = np.sqrt(dist)
		return dist

	def get_euclidean_distance_to_stone(self, single_point):
		stone = np.array(self.per_item_dict['stone'])

		dist = (stone - single_point) ** 2
		dist = np.sum(dist, axis=1)
		dist = np.sqrt(dist)
		return dist

	def does_human_have_axe(self):
		"""
		Returns whether the human currently has an axe.

		Params:

		Returns:
		bool: whether or not the human has an axe.
		"""
		if 'wooden_axe' in list(self.human_inventory.keys()):
			return True
		else:
			return False

	def does_human_have_pickaxe(self):
		"""
		Returns whether the human currently has a pickaxe.

		Params:

		Returns:
		bool: whether or not the human has a pickaxe.
		"""
		if 'wooden_pickaxe' in list(self.human_inventory.keys()):
			return True
		else:
			return False

	def does_robot_have_pickaxe(self):
		"""
		Returns whether the robot currently has a pickaxe.

		Params:

		Returns:
		bool: whether or not the robot has a pickaxe.
		"""
		if 'wooden_pickaxe' in list(self.robot_inventory.keys()):
			return True
		else:
			return False

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

	def is_human_near_stone(self, single_point):
		"""
		Determines and returns whether or not the human is near a block of stone.

		Params:
		single_point(float tuple): (x,y) location of the human.

		Returns:
		bool: whether or not the human is near a block of stone
		"""
		dist1 = self.get_euclidean_distance_to_stone(single_point)
		dist2 = self.get_euclidean_distance_to_cobblestone(single_point)

		if (dist1 < self.stone_threshold).any() or (dist2 < self.stone_threshold).any():
			return True
		else:
			return False

	def is_human_near_trees(self, single_point):
		"""
		Determines and returns whether or not the human is near a tree

		Params:
		single_point(float tuple): (x,y) location of the human.

		Returns:
		bool: whether or not the human is near a tree
		"""
		dist = self.get_euclidean_distance_to_all_trees(single_point)
		if (dist < self.tree_threshold).any():
			return True
		else:
			return False

	def is_human_near_house(self, single_point):
		"""
		Determines and returns whether or not the human is near the house

		Params:
		single_point(float tuple): (x,y) location of the human.

		Returns:
		bool: whether or not the human is near the house
		"""
		# dist = np.linalg.norm(np.array(single_point) - np.array(self.house_location))
		# if dist < self.house_threshold:
		# 	return True
		# else:
		# 	return False
		# human_info = None
		# for i in self.env.world_observations['entities']:
		# 	if i['name'] == 'Human':
		# 		human_info = i
		# single_point = [human_info['x'] + 30, human_info['z'] + 30]
		# print('your location: ', single_point)
		# print('house location: ', self.house_location)
		# dist = np.linalg.norm(np.array(single_point) - np.array(self.house_location))
		# print('distance to house: ', dist)
		dx = max(abs(single_point[0] - self.house_location[0]) - 6 / 2, 0)
		dy = max(abs(single_point[1] - self.house_location[1]) - 9 / 2, 0)
		dist = max(dx, dy)
		if dist < self.house_threshold:
			# print('close to house')
			return True
		else:
			# print('NOT close to house')
			return False

	def is_human_near_crafting_table(self, single_point):
		"""
		Determines and returns whether or not the human is near the crafting table.

		Params:
		single_point(float tuple): (x,y) location of the human.

		Returns:
		bool: whether or not the human is near the crafting table.
		"""
		dist = np.linalg.norm(np.array(single_point) - np.array(self.crafting_table_location))
		if dist < self.craft_threshold:
			return True
		else:
			return False

	# def get_number_of_chest_stone(self):
	# 	"""
	# 	Returns the amount of cobblestone in the resources chest
	#
	# 	Params:
	#
	# 	Returns:
	#
	# 	int: the number of cobblestones in the resources chest.
	# 	"""
	# 	if 'cobblestone' in list(self.chest_state.keys()):
	# 		return self.chest_state['cobblestone']
	# 	else:
	# 		return 0

	def get_number_of_chest_planks(self):
		"""
		Returns the number of planks in the resources chest.

		Params:

		Returns:

		int: the number of planks in the resources chest.
		"""
		if 'planks' in list(self.chest_state.keys()):
			return self.chest_state['planks']
		else:
			return 0

	def get_number_of_chest_wood(self):
		"""
		Returns the number of logs in the resources chest.

		Params:

		Returns:

		int: the number of logs in the resources chest.
		"""

		if 'log' in list(self.chest_state.keys()):
			return self.chest_state['log']
		else:
			return 0

	def determine_human_behavior(self):
		"""
		Returns the current human behavior (actions the human is undertaking) based on observations of the minecraft environment.

		Params:

		Returns:

		str: the current human behavior.
		"""

		self.convert_state_to_dict(list(self.state['board']))
		# if self.phase == 0:
		# 	human_info = self.get_human_info()
		# 	single_point = [human_info['x'] + 30, human_info['z'] + 30]
		# 	dist = self.get_euclidean_distance_to_all_trees(single_point)
		#
		# 	if (dist < self.tree_threshold).any():
		# 		if self.get_number_of_human_wood() >= 5:
		# 			return 'chopping', '1_1'
		# 		else:
		# 			return 'chopping', '1_0'
		# 		# return self.possible_human_behaviors[0], '1'  # chopping
		# 	else:
		# 		if self.get_number_of_human_wood() >= 5:
		# 			return 'crafting', '0_1'
		# 		else:
		# 			return 'crafting', '0_0'
		# 		# return self.possible_human_behaviors[1], '0'  # crafting
		#
		#
		# elif self.phase == 1:
		# 	human_info = self.get_human_info()
		# 	single_point = [human_info['x'] + 30, human_info['z'] + 30]
		# 	dist = self.get_euclidean_distance_to_all_trees(single_point)
		#
		# 	if self.does_human_have_axe():
		# 		if (dist < self.tree_threshold).any():
		# 			return 'chopping', '1_1'
		# 		else:
		# 			return 'crafting', '1_0'
		# 		# return self.possible_human_behaviors[0], '1'  # chopping
		# 	else:
		# 		if self.is_human_near_stone(single_point):
		# 			return 'mining', '0_1'
		# 		else:
		# 			return 'crafting', '0_0'


		human_info = self.get_human_info()
		single_point = [human_info['x'] + 30, human_info['z'] + 30]

		if self.is_human_near_house(single_point):
			if self.is_human_near_crafting_table(single_point):
				return 'crafting', '1_1'
			else:
				return 'building', '1_0'
		# return self.possible_human_behaviors[0], '1'  # chopping
		else:
			if self.is_human_near_stone(single_point):
				return 'mining', '0_1'
			else:
				return 'chopping', '0_0'

		# elif self.phase == 3:
		# 	human_info = self.get_human_info()
		# 	single_point = [human_info['x'] + 30, human_info['z'] + 30]
		#
		# 	if self.is_human_near_house(single_point):
		# 		if self.is_human_near_crafting_table(single_point):
		# 			return 'crafting', '1_1'
		# 		else:
		# 			return 'building', '1_0'
		# 	# return self.possible_human_behaviors[0], '1'  # chopping
		# 	else:
		# 		if self.is_human_near_trees(single_point):
		# 			return 'chopping', '0_1'
		# 		else:
		# 			return 'mining', '0_0'
		#
		# elif self.phase == 4:
		# 	human_info = self.get_human_info()
		# 	single_point = [human_info['x'] + 30, human_info['z'] + 30]
		#
		# 	if self.does_human_have_pickaxe():
		# 		if self.is_human_near_stone(single_point):
		# 			return 'mining', '1_1'
		# 		else:
		# 			return 'building', '1_0'
		# 	# return self.possible_human_behaviors[0], '1'  # chopping
		# 	else:
		# 		if self.is_human_near_trees(single_point):
		# 			return 'chopping', '0_1'
		# 		else:
		# 			return 'building', '0_0'
		# else:
		# 	print('you messed up')

	def set_phase(self, cur_time):
		"""
		Sets the current phase of the game based on the score.

		Params:

		Returns:
		"""
		phase = self.check_house_progress()
		if phase >= self.phase:
			self.phase = phase
		print('Current phase is :', self.phase)

	def load_in_new_human_inventory(self):
		import pickle

		try:
			self.human_inventory = pickle.load(
				open('./Python_Examples/inventory_of_human.pkl', 'rb'))
		except EOFError:
			sleep(.5)
			self.human_inventory = pickle.load(
				open('./Python_Examples/inventory_of_human.pkl', 'rb'))

	# self.give_bad_axe()

	def get_robot_inventory(self):
		"""
		Sets the robot's inventory based on it's hotbar items.

		Params:

		Returns:
		"""
		# robot_resources = {}
		# inventory_of_robot = {'Hotbar_0_size': 'Hotbar_0_item', 'Hotbar_1_size': 'Hotbar_1_item', 'Hotbar_2_size': 'Hotbar_2_item',
		# 					  'Hotbar_3_size': 'Hotbar_3_item',
		# 					  'Hotbar_4_size': 'Hotbar_4_item', 'Hotbar_5_size': 'Hotbar_5_item', 'Hotbar_6_size': 'Hotbar_6_item',
		# 					  'Hotbar_7_size': 'Hotbar_7_item',
		# 					  'Hotbar_8_size': 'Hotbar_8_item'}
		# for i in inventory_of_robot.keys():
		# 	if self.state[i] != 0:
		# 		if robot_resources[self.state[inventory_of_robot[i]]] != 0:
		# 			robot_resources[self.state[inventory_of_robot[i]]] += self.state[i]
		# 		else:
		# 			robot_resources[self.state[inventory_of_robot[i]]] = self.state[i]
		#
		# self.robot_inventory = robot_resources
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
			if self.state[i] != 0:
				if self.state[inventory_of_robot[i]] not in list(robot_resources.keys()):
					robot_resources[self.state[inventory_of_robot[i]]] = self.state[i]
				else:
					robot_resources[self.state[inventory_of_robot[i]]] += self.state[i]
		self.robot_inventory = robot_resources

	def convert_state_to_dict(self, env_board):
		self.env_dict = {}
		self.progress_env_dict = {}
		self.per_item_dict = {}
		location_of_stone = pickle.load(open(os.curdir + '/location_of_stones.pkl', 'rb'))
		location_of_cobblestones = pickle.load(open(os.curdir + '/location_of_cobblestones.pkl', 'rb'))
		for i in set(env_board):
			self.per_item_dict[i] = []
			self.progress_env_dict[i] = []
		c = 0
		for i in range(6):
			for j in range(61):
				for k in range(61):
					x = k
					y = j
					z = i
					if env_board[c] == 'air':
						c += 1
						continue
					if env_board[c] == 'bedrock':
						c += 1
						continue
					if env_board[c] == 'stone':
						if (x, y, z) in location_of_stone:
							self.progress_env_dict[(x, y, z)] = env_board[c]
							c += 1
							continue
					if env_board[c] == 'cobblestone':
						if (x, y, z) in location_of_cobblestones:
							self.progress_env_dict[(x, y, z)] = env_board[c]
							c += 1
							continue
					self.env_dict[(x, y, z)] = env_board[c]
					self.progress_env_dict[(x, y, z)] = env_board[c]

					if (x, y) not in self.per_item_dict[env_board[c]]:
						self.per_item_dict[env_board[c]].append((x, y))

					c += 1
		save_pickle(os.curdir, self.per_item_dict, 'current_per_item_dict.pkl', False)

	def check_house_progress(self, output_phase=True):
		"""
		Checks the current progress of the house and sets the current score based on the progress. Also returns the current phase by default.

		Params:
		output_phase(bool): whether or not to output the current phase based on the score. Default is True

		Returns:
		int: the current phase.
		"""
		self.load_in_new_human_inventory()
		if output_phase:
			location_of_house_planks = pickle.load(open(os.curdir + '/location_of_planks.pkl', 'rb'))
			location_of_stone = pickle.load(open(os.curdir + '/location_of_stones.pkl', 'rb'))
			location_of_cobblestones = pickle.load(open(os.curdir + '/location_of_cobblestones.pkl', 'rb'))
			location_of_fence = pickle.load(open(os.curdir + '/location_of_fence.pkl', 'rb'))
			location_of_fence_gate = pickle.load(open(os.curdir + '/location_of_fence_gate.pkl', 'rb'))
			location_of_door = [(33, 36, 0), (34, 36, 0)]
			location_of_stairs = pickle.load(open(os.curdir + '/location_of_stairs.pkl', 'rb'))

			first_layer = 0
			third_layer = 0
			second_layer = 0
			fourth_layer = 0
			fence = 0
			door = 0
			stairs = 0
			for i in location_of_house_planks:
				if i[2] == 0:
					if i in self.env_dict.keys():
						if self.env_dict[i] == 'planks':
							first_layer += 1

				if i[2] == 2:
					if i in self.env_dict.keys():
						if self.env_dict[i] == 'planks':
							third_layer += 1

			if self.which_env == 1:
				for i in location_of_stone:
					if i in self.progress_env_dict.keys():
						if self.progress_env_dict[i] == 'stone':
							second_layer += 1
			else:
				for i in location_of_stone:
					if i in self.progress_env_dict.keys():
						if self.progress_env_dict[i] == 'cobblestone':
							second_layer += 1

			if self.which_env == 1:
				for i in location_of_cobblestones:
					if i in self.progress_env_dict.keys():
						if self.progress_env_dict[i] == 'cobblestone':
							fourth_layer += 1
			else:
				for i in location_of_stone:
					if i in self.progress_env_dict.keys():
						if self.progress_env_dict[i] == 'stone':
							fourth_layer += 1

			for i in location_of_fence:
				if i in self.env_dict.keys():
					if 'fence' in self.env_dict[i]:
						fence += 1

			for i in location_of_fence_gate:
				if i in self.env_dict.keys():
					if 'fence_gate' in self.env_dict[i]:
						fence += 1

			for i in location_of_door:
				if i in self.env_dict.keys():
					if 'door' in self.env_dict[i]:
						door += 1

			for i in location_of_stairs:
				if i in self.env_dict.keys():
					if 'stair' in self.env_dict[i]:
						stairs += 1


			tot_score = len(location_of_house_planks) \
						+ len(location_of_stone) \
						+ len(location_of_cobblestones) \
						+ len(location_of_fence) \
						+ len(location_of_fence_gate) \
						+ len(location_of_door) \
						+ len(location_of_stairs)

			self.get_robot_inventory()
			score = (self.get_number_of_human_wood() * 4
					 + self.get_number_of_human_planks()
					 + self.get_number_of_robot_wood() * 4
					 + self.get_number_of_robot_planks()
					 + self.get_tot_number_of_robot_stone()
					 + self.get_number_of_human_stone()
					+ self.get_number_of_chest_wood() * 4
					+ self.get_number_of_chest_planks()
					+ self.get_tot_number_of_chest_stone()
					 + first_layer + second_layer + third_layer + fourth_layer + fence + door + stairs)

			self.curr_house_andesite = second_layer
			building_score = score - (self.get_number_of_human_wood() * 4
					 + self.get_number_of_human_planks()
					 + self.get_number_of_robot_wood() * 4
					 + self.get_number_of_robot_planks()
					 + self.get_tot_number_of_robot_stone()
					 + self.get_number_of_human_stone()
					+ self.get_number_of_chest_wood() * 4
					+ self.get_number_of_chest_planks()
					+ self.get_tot_number_of_chest_stone())
			robot_resources_score = (
					 + self.get_number_of_robot_wood() * 4
					 + self.get_number_of_robot_planks()
					 + self.get_tot_number_of_robot_stone()
					 + self.get_number_of_chest_wood() * 4
					 + self.get_number_of_chest_planks()
					 + self.get_tot_number_of_chest_stone()
			)
			human_resources_score = (self.get_number_of_human_wood() * 4
					 + self.get_number_of_human_planks()
					 + self.get_number_of_human_stone())
			print("Current house andesite: ", second_layer)
			print('score is: ', score)
			print('building score is ', building_score, '/', tot_score)
			# update at participant 44
			import time
			current_real_time = time.time() - self.start_time
			self.build_progress_tracker[(self.cur_time,current_real_time)] = building_score
			self.robot_resources_tracker[(self.cur_time,current_real_time)] = robot_resources_score
			self.human_resource_tracker[(self.cur_time,current_real_time)] = human_resources_score
			self.robot_resources_verbose_tracker[(self.cur_time,current_real_time)] = self.robot_inventory
			self.chest_state_resources_verbose_tracker[(self.cur_time,current_real_time)] = self.chest_state
			save_pickle(os.curdir + '/' + str(self.participant_number),
						[self.build_progress_tracker, self.robot_resources_tracker, self.human_resource_tracker], 'build_progress.pkl', False)
			save_pickle(os.curdir + '/' + str(self.participant_number),
						[self.robot_resources_verbose_tracker, self.chest_state_resources_verbose_tracker], 'robot_inventory_chest_state_progress.pkl', False)
			if score < 28:
				return 0
			elif building_score >= tot_score:
				# house is completed
				# save time
				save_pickle(os.curdir + '/' + str(self.participant_number), [(self.cur_time,current_real_time)], 'end_time.pkl', False)
				# save resources of each entity
				save_pickle(os.curdir + '/' + str(self.participant_number), [self.human_inventory], 'human_resources_end_game.pkl', False)
				save_pickle(os.curdir + '/' + str(self.participant_number), [self.robot_inventory], 'robot_resources_end_game.pkl', False)
				save_pickle(os.curdir + '/' + str(self.participant_number), [self.chest_state], 'chest_resources_end_game.pkl', False)
				save_pickle(os.curdir + '/' + str(self.participant_number), [self.build_progress_tracker, self.robot_resources_tracker, self.human_resource_tracker], 'build_progress_end_game.pkl', False)
				save_pickle(os.curdir + '/' + str(self.participant_number),
							[self.robot_resources_verbose_tracker, self.chest_state_resources_verbose_tracker],
							'end_game_robot_inventory_chest_state_progress.pkl', False)
				# close simulation and game.
				print('game complete')
				raise SystemExit
			elif 28 <= score < 54:
				return 1
			elif 54 <= score < 80:
				return 2

			elif 80 <= score < 115:
				return 3
			else:
				# house is not completed, but you have enough resources
				return 4
		else:
			location_of_house_planks = pickle.load(open(os.curdir + '/location_of_planks.pkl', 'rb'))
			location_of_stone = pickle.load(open(os.curdir + '/location_of_stones.pkl', 'rb'))
			location_of_cobblestones = pickle.load(open(os.curdir + '/location_of_cobblestones.pkl', 'rb'))
			location_of_fence = pickle.load(open(os.curdir + '/location_of_fence.pkl', 'rb'))
			location_of_fence_gate = pickle.load(open(os.curdir + '/location_of_fence_gate.pkl', 'rb'))

			first_layer = 0
			third_layer = 0
			second_layer = 0
			fourth_layer = 0
			fence = 0
			for i in location_of_house_planks:
				if i[2] == 0:
					if i in self.env_dict.keys():
						if self.env_dict[i] == 'planks':
							first_layer += 1

				if i[2] == 2:
					if i in self.env_dict.keys():
						if self.env_dict[i] == 'planks':
							third_layer += 1

			if self.which_env == 1:
				for i in location_of_stone:
					if i in self.env_dict.keys():
						if self.env_dict[i] == 'stone':
							second_layer += 1
			else:
				for i in location_of_stone:
					if i in self.env_dict.keys():
						if self.env_dict[i] == 'cobblestone':
							second_layer += 1

			if self.which_env == 1:
				for i in location_of_cobblestones:
					if i in self.env_dict.keys():
						if self.env_dict[i] == 'cobblestone':
							fourth_layer += 1
			else:
				for i in location_of_stone:
					if i in self.env_dict.keys():
						if self.env_dict[i] == 'stone':
							fourth_layer += 1

			for i in location_of_fence:
				if i in self.env_dict.keys():
					if self.env_dict[i] == 'fence':
						fence += 1

			for i in location_of_fence_gate:
				if i in self.env_dict.keys():
					if self.env_dict[i] == 'fence_gate':
						fence += 1
			tot_score = len(location_of_house_planks) \
						+ len(location_of_stone) \
						+ len(location_of_cobblestones) \
						+ len(location_of_fence) \
						+ len(location_of_fence_gate)

			self.get_robot_inventory()
			score = (self.get_number_of_human_wood() * 4
					 + self.get_number_of_human_planks()
					 + self.get_number_of_robot_wood() * 4
					 + self.get_number_of_robot_planks() + first_layer + second_layer + third_layer + fourth_layer + fence)
			print('score is: ', score)
			return score

	def save_world_model(self, env_board):
		self.env_dict = {}
		self.per_item_dict = {}
		for i in set(env_board):
			self.per_item_dict[i] = []
		c = 0
		for i in range(6):
			for j in range(61):
				for k in range(61):
					x = k
					y = j
					z = i
					if env_board[c] == 'air':
						c += 1
						continue
					self.env_dict[(x, y, z)] = env_board[c]

					if (x, y, z) not in self.per_item_dict[env_board[c]]:
						self.per_item_dict[env_board[c]].append((x, y, z))

					c += 1
		save_pickle(os.curdir, [self.per_item_dict, self.env_dict], 'end_of_game_world.pkl', False)


class RandomAgent(BaseAgent):
	"""
	An agent that selects actions uniformly at random
	"""

	def __init__(self, name, nb_actions, delay_between_action=0, visualizer=None):
		super(RandomAgent, self).__init__(name, nb_actions, visualizer)
		print('RANDOM AGENT $$ CHOSEN')
		self._delay = delay_between_action

	def act(self, new_state, reward, done, is_training=False):
		if self._delay > 0:
			from time import sleep
			sleep(self._delay)
		return np.random.randint(0, self.nb_actions)


class ConsoleAgent(BaseAgent):
	""" Provide a console interface for mediating human interaction with
   an environment

	Users are prompted for input when an action is required:

	Agent-1, what do you want to do?
		1: action1
		2: action2
		3: action3
		...
		N: actionN
	Agent-1: 2
	...
	"""

	def __init__(self, name, actions, stdin=None):
		assert isinstance(actions, Iterable), 'actions need to be iterable (e.g., list, tuple)'
		assert len(actions) > 0, 'actions need at least one element'

		super(ConsoleAgent, self).__init__(name, len(actions))

		self._actions = actions

		if stdin is not None:
			sys.stdin = os.fdopen(stdin)

	def act(self, new_state, reward, done, is_training=False):
		user_action = None

		while user_action is None:
			self._print_choices()
			try:
				user_input = input("%s: " % self.name)
				user_action = int(user_input)
				if user_action < 0 or user_action > len(self._actions) - 1:
					user_action = None
					print("Provided input is not valid should be [0, %d]" % (len(self._actions) - 1))
			except ValueError:
				user_action = None
				print("Provided input is not valid should be [0, %d]" % (len(self._actions) - 1))

		return user_action

	def _print_choices(self):
		print("\n%s What do you want to do?" % self.name)

		for idx, action in enumerate(self._actions):
			print("\t%d : %s" % (idx, action))


class ReplayMemory(object):
	"""
	Simple representation of agent memory
	"""

	def __init__(self, max_size, state_shape):
		assert max_size > 0, 'size should be > 0 (got %d)' % max_size

		self._pos = 0
		self._count = 0
		self._max_size = max_size
		self._state_shape = state_shape
		self._states = np.empty((max_size,) + state_shape, dtype=np.float32)
		self._actions = np.empty(max_size, dtype=np.uint8)
		self._rewards = np.empty(max_size, dtype=np.float32)
		self._terminals = np.empty(max_size, dtype=np.bool)

	def append(self, state, action, reward, is_terminal):
		"""
		Appends the specified memory to the history.
		:param state: The state to append (should have the same shape as defined at initialization time)
		:param action: An integer representing the action done
		:param reward: An integer reprensenting the reward received for doing this action
		:param is_terminal: A boolean specifying if this state is a terminal (episode has finished)
		:return:
		"""
		assert state.shape == self._state_shape, \
			'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

		self._states[self._pos, ...] = state
		self._actions[self._pos] = action
		self._rewards[self._pos] = reward
		self._terminals[self._pos] = is_terminal

		self._count = max(self._count, self._pos + 1)
		self._pos = (self._pos + 1) % self._max_size

	def __len__(self):
		"""
		Number of elements currently stored in the memory (same as #size())
		See #size()
		:return: Integer : max_size >= size() >= 0
		"""
		return self.size

	@property
	def last(self):
		"""
		Return the last observation from the memory
		:return: Tuple (state, action, reward, terminal)
		"""
		idx = self._pos
		return self._states[idx], self._actions[idx], self._rewards[idx], self._terminals[idx]

	@property
	def size(self):
		"""
		Number of elements currently stored in the memory
		:return: Integer : max_size >= size >= 0
		"""
		return self._count

	@property
	def max_size(self):
		"""
		Maximum number of elements that can fit in the memory
		:return: Integer > 0
		"""
		return self._max_size

	@property
	def history_length(self):
		"""
		Number of states stacked along the first axis
		:return: int >= 1
		"""
		return 1

	def sample(self, size, replace=False):
		"""
		Generate a random sample of desired size (if available) from the current memory
		:param size: Number of samples
		:param replace: True if sampling with replacement
		:return: Integer[size] representing the sampled indices
		"""
		return np.random.choice(self._count, size, replace=replace)

	def get_state(self, index):
		"""
		Return the specified state
		:param index: State's index
		:return: state : (input_shape)
		"""
		index %= self.size
		return self._states[index]

	def get_action(self, index):
		"""
		Return the specified action
		:param index: Action's index
		:return: Integer
		"""
		index %= self.size
		return self._actions[index]

	def get_reward(self, index):
		"""
		Return the specified reward
		:param index: Reward's index
		:return: Integer
		"""
		index %= self.size
		return self._rewards[index]

	def minibatch(self, size):
		"""
		Generate a minibatch with the number of samples specified by the size parameter.
		:param size: Minibatch size
		:return: Tensor[minibatch_size, input_shape...)
		"""
		indexes = self.sample(size)

		pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
		post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
		actions = self._actions[indexes]
		rewards = self._rewards[indexes]
		terminals = self._terminals[indexes]

		return pre_states, actions, post_states, rewards, terminals

	def save(self, out_file):
		"""
		Save the current memory into a file in Numpy format
		:param out_file: File storage path
		:return:
		"""
		np.savez_compressed(out_file, states=self._states, actions=self._actions,
							rewards=self._rewards, terminals=self._terminals)

	def load(self, in_dir):
		pass
