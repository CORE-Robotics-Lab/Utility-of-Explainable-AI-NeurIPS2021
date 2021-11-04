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

import json
import re

import numpy as np

from Python_Examples.ehis_utils import Entity, ENV_ACTIONS, ENV_BOARD, ENV_ENTITIES, \
    ENV_BOARD_SHAPE, ENV_AGENT_NAMES, ENV_AGENT_TYPES

from MalmoPython import MissionSpec
from malmo import MalmoEnvironment, MalmoStateBuilder

class PigChaseEnvironment(MalmoEnvironment):
    """
    Represent the Pig chase with two agents and a pig. Agents can try to catch
    the pig (high reward), or give up by leaving the pig pen (low reward).
    """

    def __init__(self, remotes,
                 actions=ENV_ACTIONS,
                 role=0, exp_name="",
                 human_speed=False, randomize_positions=False):

        self._mission_xml = open('experiment_world_easier.xml', 'r').read()
        # self._mission_xml = open('ehis_tutorial.xml', 'r').read()
        # self._mission_xml = open('experiment_world_easier_fully_built.xml', 'r').read()    #TODO: Change world

        # override tics per ms to play at human speed
        if human_speed:
            print('Setting mission to run at human speed')
            self._mission_xml = re.sub('<MsPerTick>\d+</MsPerTick>',
                                       '<MsPerTick>70</MsPerTick>',
                                       self._mission_xml)
        super(PigChaseEnvironment, self).__init__(self._mission_xml, actions,
                                                  remotes, role, exp_name)

        self._agent_type = None
        self._print_state_diagnostics = True

    @property
    def state(self):
        return self.world_observations

    @property
    def done(self):
        """
        Done if we have caught the pig
        """
        return super(PigChaseEnvironment, self).done

    def _construct_mission(self):
        # set agent helmet
        original_helmet = "iron_helmet"
        if self._role == 0:
            original_helmet = "diamond_helmet"
        new_helmet = original_helmet
        if self._agent_type == ENV_AGENT_TYPES.RANDOM:
            new_helmet = "golden_helmet"
        elif self._agent_type == ENV_AGENT_TYPES.FOCUSED:
            new_helmet = "diamond_helmet"
        elif self._agent_type == ENV_AGENT_TYPES.HUMAN:
            new_helmet = "leather_helmet"
        else:
            new_helmet = "iron_helmet"

        xml = re.sub(r'type="%s"' % original_helmet,
                     r'type="%s"' % new_helmet, self._mission_xml)
        return MissionSpec(xml, True)

    def _get_pos_dist(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def reset(self, agent_type=None, agent_positions=None):
        """ Overrides reset() to allow changes in agent appearance between
        missions."""
        # if agent_type and agent_type != self._agent_type:
        # self._agent_type = agent_type
        self._mission = self._construct_mission()
        return super(PigChaseEnvironment, self).reset()

    def do(self, action):
        """
        Do the action
        """
        state, reward, done = super(PigChaseEnvironment, self).do(action)
        return state, reward, self.done

    def _debug_output(self, str):
        if self._print_state_diagnostics:
            print(str)

    def is_valid(self, world_state):
        """ Pig Chase Environment is valid if the the board and entities are present """
        if not super(PigChaseEnvironment, self).is_valid(world_state):
            return False
        return True
