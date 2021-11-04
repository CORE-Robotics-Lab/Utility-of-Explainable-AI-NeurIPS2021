"""
Created by Rohan Paleja on Month Day, Year
Purpose:
"""

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

import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Process, Event
from os import path
from time import sleep

from agent import RandomAgent, AdaptiveAgent
from malmopy.agent.gui import ARROW_KEYS_MAPPING, DISCRETE_KEYS_MAPPING, CONTINUOUS_KEYS_MAPPING
from malmopy.visualization import ConsoleVisualizer
import time
# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

from Python_Examples.ehis_utils import parse_clients_args, ENV_AGENT_NAMES, ENV_ACTIONS, ENV_AGENT_TYPES
from Python_Examples.human_agent import PigChaseHumanAgent, get_agent_type
from Python_Examples.experiment_environment import PigChaseEnvironment
import pygame

MAX_ACTIONS = 25 # this should match the mission definition, used for display only

def agent_factory(name, role, kind, clients, max_episodes, max_actions, logdir, quit, model_file):
    """
    Initializes each agent, and starts game loop.
    Note this is an ASYNC function
    :param name:
    :param role:
    :param kind:
    :param clients:
    :param max_episodes:
    :param max_actions:
    :param logdir:
    :param quit:
    :param model_file:
    :return:
    """
    # if two malmo clients
    assert len(clients) >= 2, 'There are not enough Malmo clients in the pool (need at least 2)'

    participant_number = 'tmp'
    factor = 3
    display_code = '1'
    start_time = time.time()
    clients = parse_clients_args(clients)
    visualizer = ConsoleVisualizer(prefix=name)    #handles_agent's display name

    if role == 0:
        print("Initializing Robot ...")
        env = PigChaseEnvironment(clients,
                                  actions=ENV_ACTIONS, role=role,
                                  human_speed=True, randomize_positions=False)
        # initialize Adaptive agent (see adaptive class for details)

        agent = AdaptiveAgent(name, env, env.available_actions, participant_number, which_env=1, factor=factor, display_code=display_code, start_time=start_time)
        env.participant_number = agent.participant_number
        obs = env.reset(ENV_AGENT_TYPES.ADAPTIVE)
        reward = 0
        rewards = []
        episode = 0
        done = False
        sleep(10)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!STARTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        # set always light
        env._agent.sendCommand("chat /gamerule doDaylightCycle false")
        # start = time.time()
        while True:
            # select an action
            # end = time.time()
            cur_time = (env.world_observations['TotalTime'])/20
            print('current time is ', cur_time)
            # print('current other time is, ', (end - start) )

            action = agent.act(env.world_observations, reward, done, False, cur_time)
            # action = 'mine_cobblestone'
            if done:
                print("Game is over ... this is for Robot")
                # visualizer << (episode + 1, 'Reward', sum(rewards))
                rewards = []
                episode += 1
                obs = env.reset(ENV_AGENT_TYPES.RANDOM)
            # action = 'chop_birch_wood'
            print("Robot Action: ", action)

            # performs action
            obs, reward, done = env.do_ours(action, cur_time)
            agent.chest_state = env.chest_state

            rewards.append(reward)


    else:
        print("Initializing Human ...")
        env = PigChaseEnvironment(clients,
                                  actions=list(ARROW_KEYS_MAPPING.values()),
                                  human_speed=True, role=role, randomize_positions=False)
        env.reset(ENV_AGENT_TYPES.HUMAN)
        env.participant_number = participant_number

        agent = PigChaseHumanAgent(name, env, list(ARROW_KEYS_MAPPING.keys()),
                                   max_episodes, max_actions, visualizer, quit, factor=factor, display_code=display_code,participant_number=participant_number,start_time=start_time)

        agent.show()


def run_mission(agents_def):
    assert len(agents_def) == 2, 'Incompatible number of agents (required: 2, got: %d)' % len(agents_def)
    quit = Event()
    processes = []
    for agent in agents_def:
        agent['quit'] = quit
        p = Process(target=agent_factory, kwargs=agent)
        p.daemon = True
        p.start()

        if agent['role'] == 0:
            sleep(1)  # Just to let time for the server to start

        processes.append(p)
    quit.wait()
    tet = input("testing")
    quit.wait()
    for process in processes:
        print("DID we reach here?")
        process.terminate()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-e', '--episodes', type=int, default=10, help='Number of episodes to run.')
    arg_parser.add_argument('-k', '--kind', type=str, default='random', choices=['astar', 'random', 'tabq', 'challenge', 'adaptive'],
                            help='The kind of agent to play with (random, astar, tabq or challenge).')
    arg_parser.add_argument('-m', '--model_file', type=str, default='', help='Model file with which to initialise agent, if appropriate')
    arg_parser.add_argument('clients', nargs='*',
                            default=['127.0.0.1:10000', '127.0.0.1:10001'],
                            help='Malmo clients (ip(:port)?)+')
    args = arg_parser.parse_args()

    logdir = path.join('results/pig-human', datetime.utcnow().isoformat())
    agents = [{'name': agent, 'role': role, 'kind': args.kind, 'model_file': args.model_file,
               'clients': args.clients, 'max_episodes': args.episodes,
               'max_actions': MAX_ACTIONS, 'logdir': logdir}
              for role, agent in enumerate(ENV_AGENT_NAMES)]

    run_mission(agents)