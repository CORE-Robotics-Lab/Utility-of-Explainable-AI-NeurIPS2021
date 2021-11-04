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

from __future__ import division
import os
import sys
import time
import pickle
from collections import namedtuple
from tkinter import ttk, Canvas, W

import numpy as np
from ehis_utils import visualize_training, Entity, ENV_TARGET_NAMES, ENV_ENTITIES, ENV_AGENT_NAMES, \
    ENV_ACTIONS, ENV_CAUGHT_REWARD, ENV_BOARD_SHAPE, ENV_AGENT_TYPES
from six.moves import range

from malmopy.agent import AStarAgent
from malmopy.agent import QLearnerAgent, BaseAgent, RandomAgent
from malmopy.agent.gui import GuiAgent

P_FOCUSED = .75
CELL_WIDTH = 20


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


def get_agent_type(agent):
    if isinstance(agent, RandomAgent):
        return ENV_AGENT_TYPES.RANDOM
    elif isinstance(agent, PigChaseHumanAgent):
        return ENV_AGENT_TYPES.HUMAN
    else:
        return ENV_AGENT_TYPES.OTHER


class PigChaseHumanAgent(GuiAgent):

    def __init__(self, name, environment, keymap, max_episodes, max_actions,
                 visualizer, quit,factor=0,display_code='1',participant_number='tmp', start_time = None):
        """
        :param name: <String> Name of the human agent
        :param environment: <Malmo Environment> MalmoEnvironment that creates the minecraft world
        :param keymap: <Dict> Mapping keys to minecraft commands
        :param max_episodes: <Int> Not really sure if we need this
        :param max_actions: <Int> Not really sure if we need this
        :param visualizer:
        :param quit: <Event()> Not really sure what this does???
        """
        self._max_episodes = max_episodes
        self._max_actions = max_actions
        self._action_taken = 0
        self._episode = 1
        self._scores = []
        self._rewards = []
        self._episode_has_ended = False
        self._episode_has_started = False
        self._quit_event = quit
        self.inventory_axes = []
        self.inventory_pickaxes = []
        self.old_inventory = {}
        self.factor = factor
        self.display_code = display_code
        self.participant_number = participant_number
        self.start_time = start_time

        if not os.path.exists(str(self.participant_number) + "A"):
            os.mkdir(str(self.participant_number) + "A")

        if not os.path.exists(str(self.participant_number) + "B"):
            os.mkdir(str(self.participant_number) + "B")

        if not os.path.exists(str(self.participant_number) + "C"):
            os.mkdir(str(self.participant_number) + "C")

        if not os.path.exists(str(self.participant_number) + "E"):
            os.mkdir(str(self.participant_number) + "E")

        if not os.path.exists(str(self.participant_number) + "F"):
            os.mkdir(str(self.participant_number) + "F")

        if self.display_code == '1' and self.factor == 0:
            self.participant_number = self.participant_number + "A"
        if self.display_code == '0' and self.factor == 0:
            self.participant_number = self.participant_number + "A"
        if self.display_code == '0' and self.factor == 3:
            self.participant_number = self.participant_number + "B"
        if self.display_code == '1' and self.factor == 3:
            self.participant_number = self.participant_number + "C"
        if self.display_code == '0' and self.factor == 2:
            self.participant_number = participant_number + "E"
        if self.display_code == '1' and self.factor == 2:
            self.participant_number = participant_number + "F"

        super(PigChaseHumanAgent, self).__init__(name, environment, keymap,
                                                 visualizer=visualizer)

    def _on_key_pressed(self, e):
        """
        Main callback for keyboard events
        :param e:
        :return:
        """
        if e.keysym == 'Escape':
            self._quit()

        if e.keysym == 'Return':
            print("Trying to restart")
            self._env.reset()

        if e.keysym == 'p':
            self._wait()

        if e.keysym == 'u':
            self._start()

    def _build_layout(self, root):
        # Left part of the GUI, first person view
        self._first_person_header = ttk.Label(root, text='Controller', font=(None, 14, 'bold')) \
            .grid(row=0, column=0)
        self._first_person_view = ttk.Label(root)
        self._first_person_view.grid(row=1, column=0, rowspan=10)

        self._first_person_header = ttk.Label(root, text='Controller', font=(None, 14, 'bold')) \
            .grid(row=0, column=1)
        self._symbolic_view = Canvas(root)
        self._symbolic_view.configure(width=ENV_BOARD_SHAPE[0] * CELL_WIDTH,
                                      height=ENV_BOARD_SHAPE[1] * CELL_WIDTH)
        self._symbolic_view.grid(row=1, column=1)

        self._information_panel = ttk.Label(root, text='Esc == Quit; Return == Restart', font=(None, 14, 'bold'))
        self._information_panel.grid(row=1, column=0)

        # Main rendering callback
        self._pressed_binding = root.bind('<Key>', self._on_key_pressed)
        self._user_pressed_enter = False

        root.after(self._tick, self._poll_frame)

    def _poll_frame(self):
        """
        Main callback for UI rendering.
        Called at regular intervals.
        The method will ask the environment to provide a frame if available (not None).
        :return:
        """
        # print(self._env.world_observations['TotalTime'] / 20)
        cell_width = CELL_WIDTH
        circle_radius = 10

        self._env.env_dict = {}
        self._env.per_item_dict = {}
        # for i in set(self._env.world_observations['board']):
        #     self._env.per_item_dict[i] = []
        human_resources = {}
        inventory_of_human = {'InventorySlot_0_size': 'InventorySlot_0_item',
                             'InventorySlot_1_size': 'InventorySlot_1_item',
                             'InventorySlot_2_size': 'InventorySlot_2_item',
                             'InventorySlot_3_size': 'InventorySlot_3_item',
                             'InventorySlot_4_size': 'InventorySlot_4_item',
                             'InventorySlot_5_size': 'InventorySlot_5_item',
                             'InventorySlot_6_size': 'InventorySlot_6_item',
                             'InventorySlot_7_size': 'InventorySlot_7_item',
                             'InventorySlot_8_size': 'InventorySlot_8_item',
                             'InventorySlot_9_size':  'InventorySlot_9_item',
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
        for e, i in enumerate(inventory_of_human.keys()):

            # bookkeeping
            if e in self.inventory_axes:
                if self._env.world_observations[inventory_of_human[i]] != 'wooden_axe':
                    self.inventory_axes.remove(e)
            if e in self.inventory_pickaxes:
                if self._env.world_observations[inventory_of_human[i]] != 'wooden_pickaxe':
                    self.inventory_pickaxes.remove(e)

            if self._env.world_observations[i] != 0:
                if self._env.world_observations[inventory_of_human[i]] in human_resources.keys():
                    human_resources[self._env.world_observations[inventory_of_human[i]]] += self._env.world_observations[i]
                else:
                    human_resources[self._env.world_observations[inventory_of_human[i]]] = self._env.world_observations[i]


                # change axe to weakened version
                if e<=8:
                    if self._env.world_observations[inventory_of_human[i]] == 'wooden_axe':
                        if e not in self.inventory_axes:
                            self.inventory_axes.append(e)
                            self._env._agent.sendCommand("chat /replaceitem entity Human slot.hotbar." + str(e) + " air")
                            # self._env._agent.sendCommand("chat /give Human wooden_axe 1 48")
                            self._env._agent.sendCommand("chat /replaceitem entity Human slot.hotbar." + str(e) + " wooden_axe 1 48")
                    elif self._env.world_observations[inventory_of_human[i]] == 'wooden_pickaxe':
                        if e not in self.inventory_pickaxes:
                            self.inventory_pickaxes.append(e)
                            self._env._agent.sendCommand("chat /replaceitem entity Human slot.hotbar." + str(e) + " air")
                            # self._env._agent.sendCommand("chat /give Human wooden_pickaxe 1 48")
                            self._env._agent.sendCommand("chat /replaceitem entity Human slot.hotbar." + str(e) + " wooden_pickaxe 1 48")

        # if self.old_inventory != human_resources:
        #     self.old_inventory = human_resources
        #     for e, i in enumerate(inventory_of_human.keys()):
        #         if self._env.world_observations[inventory_of_human[i]] == 'wooden_axe':
        #             if e not in self.inventory_axes:
        #                 self.inventory_axes.append(e)
        #                 self._env._agent.sendCommand("chat /replaceitem entity Human slot.hotbar." + str(e) + " air")
        #                 self._env._agent.sendCommand("chat /give Human wooden_axe 1 48")
        #         if self._env.world_observations[inventory_of_human[i]] == 'wooden_pickaxe':
        #             if e not in self.inventory_pickaxes:
        #                 self.inventory_pickaxes.append(e)
        #                 self._env._agent.sendCommand("chat /replaceitem entity Human slot.hotbar." + str(e) + " air")
        #                 self._env._agent.sendCommand("chat /give Human wooden_pickaxe 1 48")

        if human_resources == {} or 'wooden_axe' not in human_resources.keys():
            self.inventory_axes = []

        if human_resources == {} or 'wooden_pickaxe' not in human_resources.keys():
            self.inventory_pickaxes = []

        import time
        current_real_time = time.time() - self.start_time
        print('current real time is ', current_real_time)
        save_pickle(os.curdir, human_resources, 'inventory_of_human.pkl', want_to_print=False)
        save_pickle(os.curdir + '/' + str(self.participant_number), human_resources,
                    'human_resources_at_time' + str([self._env.world_observations['TotalTime']/20])+'_real_time' + str(current_real_time) + '.pkl', False)

        # end checking
        env_board = self._env.world_observations['board']
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


        for i in location_of_stone:
            if i in self.progress_env_dict.keys():
                if self.progress_env_dict[i] == 'stone':
                    second_layer += 1



        for i in location_of_cobblestones:
            if i in self.progress_env_dict.keys():
                if self.progress_env_dict[i] == 'cobblestone':
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


        building_score = first_layer + second_layer + third_layer + fourth_layer + fence + door + stairs

        # print('Human: building score is ', building_score, '/', tot_score)
        if building_score >= 87:
            # house is completed
            # save time
            print('human saving at ', [self._env.world_observations['TotalTime']/20])
            import time
            current_real_time = time.time() - self.start_time
            save_pickle(os.curdir + '/' + str(self.participant_number), [self._env.world_observations['TotalTime']/20, current_real_time], 'human_agent_recorded_end_time.pkl', False)
            import time
            time.sleep(100)
            # self._quit()
            raise SystemExit

        # for c in range(len(self._env.world_observations['board'])):
        #     y = int(c/61)
        #     x = int((c - y) % 60)
        #     if c ==  1890:
        #         print('hi')
        #     z = int(c / 1891)
        #
        #     self._env.env_dict[(x, y, z)] = self._env.world_observations['board'][c]

        # c = 0
        # for i in range(6):
        #     for j in range(61):
        #         for k in range(61):
        #             x = k
        #             y = j
        #             z = i
        #             self._env.env_dict[(x, y, z)] = self._env.world_observations['board'][c]
        #
        #             if (x, y) not in self._env.per_item_dict[self._env.world_observations['board'][c]]:
        #                 self._env.per_item_dict[self._env.world_observations['board'][c]].append((x, y))
        # if (x, y, z) not in self._env.per_item_dict[self._env.world_observations['board'][c]]:
        #     self._env.per_item_dict[self._env.world_observations['board'][c]].append((x, y, z))
        #
        # c += 1

        self._root.after(self._tick, self._poll_frame)

    def _quit(self):
        print("Quitting from human input")
        self._quit_event.set()
        self._root.quit()
        sys.exit()

    def _wait(self):
        print("Pausing the game")
        self._quit_event.set()

    def _start(self):
        print("Unpausing the game")
        self._quit_event.clear()
