#!/usr/bin/python

#to do:

#are my action commands in the train the same as in expert?
#logs and videos being done?



#my losses are so fucking large, not sure how the fuck this is trainable
#why the fuck are my sample losses so much larger than the loss outputted by train_on_batch?
    #DONE my targets are only one for the max action
    #DONE also then for doing the sample losses, only look at predictions vs. that one column not vs. all 7 action columns
    #DONE read the dueling DQN, do I still train my model this way?
    #slmc_output/batch_size seems to be equal to the 4th loss but not sure wtf is going on with the others

#DONE don't get why my model outputs 4 listed losses in metrics only though 3 inputs (and only takes 3 loss calculations...)
    #the first loss term and the slmc loss terms dominate the middle 2 CNN loss terms
    #WTF is the first loss term? assuming it's a sum of other 3 plus L2 reg
    #doing some math, the first three terms plus a small value (0.09 in the small batgch I looked at) made up the total loss
        #maybe that is the L2 regularization?
#DONE my parse is never getting any reward
#DONE do the expert replay save/load nonsense
#DONE do the load trained model shit

import numpy as np
from collections import deque
import random
import time
import argparse
from os import listdir
from os.path import isfile, join, isdir

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Lambda, Conv2D
from keras.optimizers import SGD, Adam
from keras import initializers
from keras import regularizers
import keras.backend as K

import gym
import retro
from retro_contest.local import make #added for local
from baselines.common.atari_wrappers import WarpFrame, FrameStack

import per_replay as replay
import tensorflow as tf



def fill_expert_buffer(env,exp_buffer,args,nsteps=10):
    # iterate through the demo files
    # iterate through each frame of the demo file
    # convert to format used by the buffer
    # add frame to buffer
    # return expert buffer

    if isdir(args.demodir):
        onlyfiles = [f for f in listdir(args.demodir) if isfile(join(args.demodir, f))]
        onlyfiles.sort()

        demo_files_to_parse = int(args.demonumber)
        for file in onlyfiles:
            if demo_files_to_parse < 1:
                break
            if ".bk2" in file and args.game in file and args.state in file:
                exp_buffer = parse_demo(env, exp_buffer, "{}{}".format(args.demodir,file), nsteps)
                demo_files_to_parse -= 1

        if demo_files_to_parse == args.demonumber:
            print("Error: no demos found to parse")

        return exp_buffer

    else:
        print("Error: directory not found")

def parse_demo(env,rep_buffer,movie_path,nsteps=10):
    print("Parsing demo:",movie_path)
    movie = retro.Movie(movie_path)
    movie.step()
    env.initial_state = movie.get_state()
    curr_obs = env.reset()

    button_dict = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
    num_buttons = len(button_dict)
    action_dict = define_action_dict()

    parse_ts = 0
    episode_start_ts = 0
    nstep_gamma = 0.99
    nstep_state_deque = deque()
    nstep_action_deque = deque()
    nstep_rew_list = []
    nstep_nexts_deque = deque()
    nstep_done_deque = deque()
    total_rew = 0.
    while movie.step():
        #env.render()
        #time.sleep(0.01)
        keys = []
        for i in range(num_buttons):
            keys.append(movie.get_key(i))
        game_a = action_dict[game_get_dict_key(keys)]

        _obs, _rew, _done, _info = env.step(keys)
        episode_start_ts += 1
        parse_ts += 1

        #paper limits reward
        _rew = np.sign(_rew) * np.log(1. + np.abs(_rew))
        #total_rew += _rew
        #print(total_rew,_rew)
        # print(keys)
        # print(game_a)

        nstep_state_deque.append(curr_obs)
        nstep_action_deque.append(game_a)
        nstep_rew_list.append(_rew)
        nstep_nexts_deque.append(_obs)
        nstep_done_deque.append(_done)

        if episode_start_ts > 10:
            add_transition(rep_buffer,nstep_state_deque,nstep_action_deque,nstep_rew_list,nstep_nexts_deque,
                           nstep_done_deque,_obs,False,nsteps,nstep_gamma)

        # if episode done we reset
        if _done:
            #emptying the deques
            add_transition(rep_buffer, nstep_state_deque, nstep_action_deque, nstep_rew_list, nstep_nexts_deque,
                           nstep_done_deque, _obs, True, nsteps, nstep_gamma)

            # reset the environment, get the current state
            curr_obs = env.reset()

            nstep_state_deque.clear()
            nstep_action_deque.clear()
            nstep_rew_list.clear()
            nstep_nexts_deque.clear()
            nstep_done_deque.clear()

            episode_start_ts = 0
        else:
            curr_obs = _obs  # resulting state becomes the current state

    #replay is over emptying the deques
    add_transition(rep_buffer, nstep_state_deque, nstep_action_deque, nstep_rew_list, nstep_nexts_deque,
                   nstep_done_deque, _obs, True, nsteps, nstep_gamma)
    print('Parse finished. {} expert samples added.'.format(parse_ts))

    return rep_buffer

#handles transitions to add to replay buffer and expert buffer
#next step reward (ns_rew) is a list, the rest are deques
def add_transition(rep_buffer, ns_state,ns_action,ns_rew,
                   ns_nexts,ns_done, current_state, empty_deque=False, ns=10, ns_gamma=0.99,is_done=True):
    ns_rew_sum = 0.
    trans = {}
    if empty_deque:
        # emptying the deques
        while len(ns_rew) > 0:
            for i in range(len(ns_rew)):
                ns_rew_sum += ns_rew[i] * ns_gamma ** i
            # state,action,reward,
            # next_state,done, n_step_rew_sum, n_steps later
            # don't use done value because at this point the episode is done
            trans['sample'] = [ns_state.popleft(), ns_action.popleft(), ns_rew.pop(0),
                                      ns_nexts.popleft(), is_done, ns_rew_sum, current_state]
            rep_buffer.add_sample(trans)
    else:
        for i in range(ns):
            ns_rew_sum += ns_rew[i] * ns_gamma ** i
        # state,action,reward,
        # next_state,done, n_step_rew_sum, n_steps later
        trans['sample'] = [ns_state.popleft(), ns_action.popleft(), ns_rew.pop(0),
                           ns_nexts.popleft(), ns_done.popleft(), ns_rew_sum, current_state]
        rep_buffer.add_sample(trans)


#for parsing demos, turns key press from each frame into a key for action_dictionary
#b, down, left, right,
def game_get_dict_key(keys):
    # if A, B, C active activates B since all three jump
    if keys[1] or keys[8]:
        keys[0] = True
    # b, down, left, right
    key = (keys[0], keys[5], keys[6], keys[7])

    return key

#converts key presses from game_get_dict_key into an action
#action options here must match action options in SonicDescretizer()
#actions = [['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'], ['DOWN'], ['NOOP'], ['B']]
def define_action_dict():

    temp_dict = {}
    temp_dict['B'] = 6
    temp_dict['DOWN'] = 5
    temp_dict['RIGHT', 'B'] = 4
    temp_dict['LEFT', 'B'] = 3
    temp_dict['RIGHT'] = 2
    temp_dict['LEFT'] = 1
    temp_dict['NOOP'] = 0

    ret_dict = {} #B,DOWN,LEFT,RIGHT
    ret_dict[(True, True, True, True)] = temp_dict['B']
    ret_dict[(True, True, True, False)] = temp_dict['LEFT', 'B']
    ret_dict[(True, True, False, True)] = temp_dict['RIGHT', 'B']
    ret_dict[(True, True, False, False)] = temp_dict['B']
    ret_dict[(True, False, True, True)] = temp_dict['B']
    ret_dict[(True, False, True, False)] = temp_dict['LEFT', 'B']
    ret_dict[(True, False, False, True)] = temp_dict['RIGHT', 'B']
    ret_dict[(True, False, False, False)] = temp_dict['B']
    ret_dict[(False, True, True, True)] = temp_dict['DOWN']
    ret_dict[(False, True, True, False)] = temp_dict['LEFT']
    ret_dict[(False, True, False, True)] = temp_dict['RIGHT']
    ret_dict[(False, True, False, False)] = temp_dict['DOWN']
    ret_dict[(False, False, True, True)] = temp_dict['NOOP']
    ret_dict[(False, False, True, False)] = temp_dict['LEFT']
    ret_dict[(False, False, False, True)] = temp_dict['RIGHT']
    ret_dict[(False, False, False, False)] = temp_dict['NOOP']

    return ret_dict



def make_env(game='SonicTheHedgehog-Genesis',state='GreenHillZone.Act1',stack=True, scale_rew=True, scenario = 'contest',
             is_contest_env=True, action_list=[['LEFT'], ['RIGHT'], ['B']]):
    if is_contest_env:
        env = make(game=game, state=state, bk2dir='videos', monitordir='logs',scenario=scenario)
        env = SonicDiscretizer(env,action_list)
        env = AllowBacktracking(env)
    else:
        env = retro.make(game=game, state=state, scenario=scenario)

    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env, len(action_list)


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env, action_list):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        # actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']]
        #actions = [['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'],  ['NOOP'],['B']]
        #actions = [['LEFT'], ['RIGHT'], ['DOWN'], ['NOOP'], ['B']]
        #actions = [['LEFT'], ['RIGHT'], ['B']]
        #actions = [['NOOP'], ['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'], ['DOWN'],['B']]
        actions = action_list

        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            if action == ['NOOP']:
                self._actions.append(arr)
                continue
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


#using network shape/size from baselines:
    #https://github.com/openai/baselines/blob/24fe3d6576dd8f4cdd5f017805be689d6fa6be8c/baselines/deepq/experiments/run_atari.py
    #double DQN is done during training
    #there are four losses:
        #L2 loss is from paper
        #slmc loss in computed using expert action choices, expert_margin, and the dqn loss
        #dqn loss and nstep loss both use the CNN. share the weights between them
def build_model(action_len, img_rows=84, img_cols=84, img_channels=4, dueling=True, clip_value=1.0,
                learning_rate=1e-4, nstep_reg=1.0, slmc_reg=1.0, l2_reg=10e-5):

    input_img = Input(shape=(img_rows, img_cols, img_channels), name='input_img', dtype='float32')
    scale_img = Lambda(lambda x: x/255.)(input_img) #scales the image. input is in ints of 0 to 255
    layer_1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='same',
                     activation='relu', input_shape=(img_rows, img_cols, img_channels),
                     kernel_initializer=initializers.glorot_normal(seed=31),
                     kernel_regularizer=regularizers.l2(l2_reg),
                     bias_regularizer=regularizers.l2(l2_reg))(scale_img)#(input_img)
    layer_2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                     kernel_initializer=initializers.glorot_normal(seed=31),
                     kernel_regularizer=regularizers.l2(l2_reg),
                     bias_regularizer=regularizers.l2(l2_reg))(layer_1)
    layer_3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=initializers.glorot_normal(seed=31),
                     kernel_regularizer=regularizers.l2(l2_reg),
                     bias_regularizer=regularizers.l2(l2_reg))(layer_2)
    x = Flatten()(layer_3)
    x = Dense(256, activation='relu',
              kernel_initializer=initializers.glorot_normal(seed=31),
              kernel_regularizer=regularizers.l2(l2_reg),
              bias_regularizer=regularizers.l2(l2_reg))(x)
    if not dueling:
        cnn_output = Dense(action_len,
                           kernel_initializer=initializers.glorot_normal(seed=31),
                           kernel_regularizer=regularizers.l2(l2_reg),
                           bias_regularizer=regularizers.l2(l2_reg), name='cnn_output')(x)
    else:
        dueling_values = Dense(1,
                               kernel_initializer=initializers.glorot_normal(seed=31),
                               kernel_regularizer=regularizers.l2(l2_reg),
                               bias_regularizer=regularizers.l2(l2_reg), name='dueling_values')(x)
        dueling_actions = Dense(action_len,
                                kernel_initializer=initializers.glorot_normal(seed=31),
                                kernel_regularizer=regularizers.l2(l2_reg),
                                bias_regularizer=regularizers.l2(l2_reg), name='dq_actions')(x)

        # https://github.com/keras-team/keras/issues/2364
        def dueling_operator(duel_input):
            duel_v = duel_input[0]
            duel_a = duel_input[1]
            return duel_v + (duel_a - K.mean(duel_a, axis=1, keepdims=True))

        cnn_output = Lambda(dueling_operator, name='cnn_output')([dueling_values, dueling_actions])
        # alternate way: https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py
    cnn_model = Model(input_img, cnn_output)

    input_img_dq = Input(shape=(img_rows, img_cols, img_channels), name='input_img_dq', dtype='float32')
    input_img_nstep = Input(shape=(img_rows, img_cols, img_channels), name='input_img_nstep', dtype='float32')
    dq_output = cnn_model(input_img_dq)
    nstep_output = cnn_model(input_img_nstep)

    # supervised large margin classifier loss
    # max[Q(s,a)+l(ae,a)] - Q(s,ae) if expert replay, 0 otherwise
    # minimize it with mean absolute error vs. fake target values of 0
    input_is_expert = Input(shape=(1,), name='input_is_expert')
    input_expert_action = Input(shape=(2,), name='input_expert_action', dtype='int32')
    input_expert_margin = Input(shape=(action_len,), name='input_expert_margin')

    def slmc_operator(slmc_input):
        is_exp = slmc_input[0]
        sa_values = slmc_input[1]
        exp_act = K.cast(slmc_input[2], dtype='int32')
        exp_margin = slmc_input[3]

        #exp_val = tf.gather(sa_values, exp_act, axis=-1)
        exp_val = tf.gather_nd(sa_values, exp_act)
        # not sure how to use arange with Keras like you can with numpy so convert to numpy then back to Keras
        # I want state,action values along the expert action choice. easy with np.arange, so do that part outside of model
        #         exp_identity = K.ones(shape=K.shape(sa_values)) * expert_margin
        #         exp_identity[K.arange(sa_len),K.eval(exp_act)] = 0.
        #         exp_val = sa_values[K.arange(sa_len),exp_act]


        max_margin = K.max(sa_values + exp_margin, axis=1)
        max_margin_2 = max_margin - exp_val
        max_margin_3 = K.reshape(max_margin_2,K.shape(is_exp))
        max_margin_4 = tf.multiply(is_exp,max_margin_3)
        return max_margin_4

        # print(K.shape(is_exp)[0])
        # print(K.shape(is_exp)[1])
        # print("shape of tensor sa_values")
        # print(K.shape(sa_values))
        # print("shape of tensor exp_act")
        # print(K.shape(exp_act))
        # print("shape of tensor exp_marging")
        # print(K.shape(exp_margin))
        # print("shape of ex_val")
        # print(K.shape(exp_val))
        #return is_exp * (K.max(sa_values + exp_margin, axis=1) - exp_val)

    slmc_output = Lambda(slmc_operator, name='slmc_output')([input_is_expert, dq_output,
                                                             input_expert_action, input_expert_margin])

    model = Model(inputs=[input_img_dq, input_img_nstep, input_is_expert, input_expert_action, input_expert_margin],
                  outputs=[dq_output, nstep_output, slmc_output])

    if clip_value is not None:
        adam = Adam(lr=learning_rate, clipvalue=clip_value)
    else:
        adam = Adam(lr=learning_rate)

    model.compile(optimizer=adam,
                  loss=['mse', 'mse', 'mae'],
                  loss_weights=[1., nstep_reg, slmc_reg])

    return model


def train_network(env, train_model, target_model, exp_buffer, rep_buffer, action_len,max_timesteps=1000000,min_buffer_size=20000,
                  epsilon_start = 0.99,epsilon_min=0.01,nsteps = 10, batch_size = 32,expert_margin=0.8,
                  gamma=0.99,nstep_gamma=0.99):

    update_every = 10000  # update target_model after this many training steps
    time_int = int(time.time())  # for saving models
    nstep_state_deque = deque()
    nstep_action_deque = deque()
    nstep_rew_list = []
    nstep_nexts_deque = deque()
    nstep_done_deque = deque()
    empty_by_one = np.zeros((1, 1))
    empty_exp_action_by_one = np.zeros((1, 2))
    empty_action_len_by_one = np.zeros((1, action_len))

    episode_start_ts = 0 #when this reaches n_steps, can start populating n_step_maxq_deque

    train_ts = -1
    explore_ts = max_timesteps * 0.8

    loss = np.zeros((4,))
    epsilon = epsilon_start
    curr_obs = env.reset()

    # paper samples expert and self generated samples by weights, I used fixed proportion like Ape-X DQfD
    exp_batch_size = int(batch_size / 4)
    gen_batch_size = batch_size - exp_batch_size
    episode = 1
    total_rew = 0.

    while train_ts < max_timesteps:
        train_ts += 1
        episode_start_ts += 1

        # get action
        # action_command used to actually input the command
        if random.random() <= epsilon:
            action_command = env.action_space.sample()
        else:
            temp_curr_obs = np.array(curr_obs)
            temp_curr_obs = temp_curr_obs.reshape(1, temp_curr_obs.shape[0], temp_curr_obs.shape[1], temp_curr_obs.shape[2])
            q, _, _ = train_model.predict([temp_curr_obs, temp_curr_obs,empty_by_one, empty_exp_action_by_one,empty_action_len_by_one])
            action_command = np.argmax(q)

        # reduce exploration rate epsilon
        if epsilon > epsilon_min:
            epsilon -= (epsilon_start - epsilon_min) / explore_ts

        # do action
        _obs, _rew, _done, _info = env.step(action_command)

        # reward clip value from paper = sign(r) * log(1+|r|)
        _rew = np.sign(_rew) * np.log(1.+np.abs(_rew))
        #total_rew += _rew
        #print(action_command, _rew, epsilon)
        nstep_state_deque.append(curr_obs)
        nstep_action_deque.append(action_command)
        nstep_rew_list.append(_rew)
        nstep_nexts_deque.append(_obs)
        nstep_done_deque.append(_done)

        if episode_start_ts > 10:
            add_transition(rep_buffer, nstep_state_deque, nstep_action_deque, nstep_rew_list, nstep_nexts_deque,
                           nstep_done_deque, _obs, False, nsteps, nstep_gamma)

        #if episode done we reset
        if _done:
            #print('episode done {}'.format(total_rew))
            # emptying the deques
            add_transition(rep_buffer, nstep_state_deque, nstep_action_deque, nstep_rew_list, nstep_nexts_deque,
                           nstep_done_deque, _obs, True, nsteps, nstep_gamma)

            episode += 1

            # reset the environment, get the current state
            curr_obs = env.reset()

            nstep_state_deque.clear()
            nstep_action_deque.clear()
            nstep_rew_list.clear()
            nstep_nexts_deque.clear()
            nstep_done_deque.clear()

            episode_start_ts = 0
        else:
            curr_obs = _obs  # resulting state becomes the current state

        # train the network using expert and experience replay
            #I fix the sample between the two while paper samples based on priority
        if train_ts > min_buffer_size:
            #sample from expert and experience replay and concatenate into minibatches
            #get target network and train network predictions
            #use Double DQN

            exp_minibatch = exp_buffer.sample(exp_batch_size)
            exp_zip_batch = []
            for i in exp_minibatch:
                exp_zip_batch.append(i['sample'])
            exp_states_batch, exp_action_batch, exp_reward_batch, exp_next_states_batch, \
            exp_done_batch, exp_nstep_rew_batch, exp_nstep_next_batch = map(np.array, zip(*exp_zip_batch))

            is_expert_input = np.zeros((batch_size, 1))
            is_expert_input[0:exp_batch_size, 0] = 1
            # expert action made into a 2d array for when tf.gather_nd is called during training
            input_exp_action = np.zeros((batch_size, 2))
            input_exp_action[:, 0] = np.arange(batch_size)
            input_exp_action[0:exp_batch_size, 1] = exp_action_batch
            expert_margin_array = np.ones((batch_size,action_len)) * expert_margin
            expert_margin_array[np.arange(exp_batch_size),exp_action_batch] = 0. #expert chosen actions don't have margin

            minibatch = rep_buffer.sample(gen_batch_size)
            zip_batch = []
            for i in minibatch:
                zip_batch.append(i['sample'])
            states_batch, action_batch, reward_batch, next_states_batch, done_batch, \
                nstep_rew_batch, nstep_next_batch = map(np.array, zip(*zip_batch))

            #concatenating expert and generated replays
            concat_states = np.concatenate((exp_states_batch, states_batch), axis=0)
            concat_next_states = np.concatenate((exp_next_states_batch, next_states_batch), axis=0)
            concat_nstep_states = np.concatenate((exp_nstep_next_batch, nstep_next_batch), axis=0)
            concat_reward = np.concatenate((exp_reward_batch, reward_batch), axis=0)
            concat_done = np.concatenate((exp_done_batch, done_batch), axis=0)
            concat_action = np.concatenate((exp_action_batch, action_batch), axis=0)
            concat_nstep_rew = np.concatenate((exp_nstep_rew_batch, nstep_rew_batch), axis=0)

            loss += inner_train_function(train_model, target_model, exp_buffer, rep_buffer,
                                        concat_states, concat_action, concat_reward, concat_next_states,
                                        concat_done, concat_nstep_rew, concat_nstep_states, is_expert_input,
                                        input_exp_action, expert_margin_array, action_len,exp_minibatch,minibatch,
                                         batch_size, gamma, nstep_gamma,exp_batch_size)

        # save model weights and update target model
        if train_ts % update_every == 0 and train_ts >= min_buffer_size:
            print("Saving model weights at DQfD timestep {}. Loss is {}".format(train_ts,loss))
            loss = np.zeros((4,))
            zString = "dqfd_training_weights/model_{}_{}.h5".format(time_int,train_ts)
            train_model.save_weights(zString, overwrite=True)
            # updating fixed Q network weights
            target_model.load_weights(zString)

        #info logged and videos recorded through env
    print("Saving final model weights. Loss is {}".format(loss))
    zString = "dqfd_training_weights/final_model_{}_{}.h5".format(time_int, train_ts)
    train_model.save_weights(zString, overwrite=True)


def inner_train_function(train_model,target_model,exp_buffer,rep_buffer,
                         states_batch,action_batch,reward_batch,
                         next_states_batch,done_batch,nstep_rew_batch,nstep_next_batch,
                         is_expert_input,expert_action_batch,expert_margin,
                         action_len,exp_minibatch,minibatch,
                         batch_size=32,gamma=0.99,nstep_gamma=0.99,exp_batch_size=8):

    empty_batch_by_one = np.zeros((batch_size,1))
    empty_action_batch = np.zeros((batch_size,2))
    empty_action_batch[:,0] = np.arange(batch_size)
    empty_batch_by_action_len = np.zeros((batch_size, action_len))
    ti_tuple = tuple([i for i in range(batch_size)])  # used for indexing a array down below, probably a better way to do this
    nstep_final_gamma = nstep_gamma ** 10

    #getting Double DQN values
        #only for the CNN loss, not including the slmc loss
        # inputs=[input_img_dq, input_img_nstep, input_is_expert, input_expert_action, input_expert_margin]
    q_values_next_target, nstep_q_values_next_target, _ = target_model.predict(
        [next_states_batch, nstep_next_batch,
         empty_batch_by_one, empty_action_batch,
         empty_batch_by_action_len])

    q_values_next_train, nstep_q_values_next_train, _ = train_model.predict(
        [next_states_batch, nstep_next_batch,
         empty_batch_by_one, empty_action_batch,
         empty_batch_by_action_len])

    action_max = np.argmax(q_values_next_train, axis=1)
    nstep_action_max = np.argmax(nstep_q_values_next_train, axis=1)

    #taking loss of outputs vs. what model predicts
        #training only done on actions selected (hence action batch column modified by reward)
        #loss will be 0 for actions not taken
        #tensorflow models typically do this by feeding in array of 1 for action taken and the rest 0's then doing loss based on that one action
    dq_targets, nstep_targets, _ = train_model.predict([states_batch,states_batch,
                                                        is_expert_input, expert_action_batch, expert_margin])
    dq_targets[ti_tuple, action_batch] = reward_batch + \
                                          (1 - done_batch) * gamma \
                                          * q_values_next_target[np.arange(batch_size), action_max]

    nstep_targets[ti_tuple, action_batch] = nstep_rew_batch + \
                                             (1 - done_batch) * nstep_final_gamma \
                                             * nstep_q_values_next_target[np.arange(batch_size), nstep_action_max]

    # apparently can't get sample loss by row in keras?
        # instead get predictions, compare to target values to get weights
        # inputs=[input_img_dq, input_img_nstep, input_is_expert, input_expert_action, input_expert_margin]
        # tried to match up the preds with the targets to get loss but was unsuccessful except with slmc loss
        # so weighting it by value
    dq_pred, nstep_pred, slmc_pred = train_model.predict_on_batch([states_batch, states_batch,
                                                                   is_expert_input, expert_action_batch, expert_margin])

    dq_loss = np.square(dq_pred[np.arange(batch_size),action_batch]-dq_targets[np.arange(batch_size),action_batch])
    nstep_loss = np.square(nstep_pred[np.arange(batch_size), action_batch] - nstep_targets[np.arange(batch_size), action_batch])
    #this should be the same as the model loss but is not for some reason. only slmc loss is. instead treating it down below
    #sample_losses = np.reshape(dq_loss,(batch_size,1))+np.reshape(nstep_loss,(batch_size,1))+np.abs(slmc_pred)

    # training on model
    # outputs=[dq_output, nstep_output, slmc_output]
    loss = train_model.train_on_batch([states_batch, states_batch,is_expert_input, expert_action_batch, expert_margin],
                                                                [dq_targets, nstep_targets, empty_batch_by_one])

    dq_loss_weighted = np.reshape(dq_loss,(batch_size,1))/np.sum(dq_loss) * loss[1] * batch_size
    nstep_loss_weighted = np.reshape(nstep_loss,(batch_size,1))/np.sum(nstep_loss) * loss[2] * batch_size
    #weighting the loss by the model batch loss
    sample_losses = dq_loss_weighted + nstep_loss_weighted + np.abs(slmc_pred)

    if rep_buffer is not None:
        exp_buffer.update_weights(exp_minibatch, sample_losses[:exp_batch_size])
        rep_buffer.update_weights(minibatch, sample_losses[-(batch_size-exp_batch_size):])
    else:
        exp_buffer.update_weights(exp_minibatch, sample_losses)

    # print(np.sum(sample_losses))
    # print(np.sum(dq_loss_weighted))
    # print(np.sum(nstep_loss_weighted))
    # print(np.sum(np.abs(slmc_pred)))
    # print(loss)
    return np.array(loss) #np.sum(loss)



def train_expert_model(train_model, target_model, exp_buffer, action_len, batch_size=32,train_steps=750000,update_every=10000,
                       gamma=0.99, nstep_gamma=0.99,exp_margin_constant=0.8):
    time_int = int(time.time())
    loss = np.zeros((4,))
    print('Training expert model')

    for current_step in range(train_steps):
        #samples stored as a list of dictionaries. have to get the samples from that dict list
        exp_minibatch = exp_buffer.sample(batch_size)
        exp_zip_batch = []
        for i in exp_minibatch:
            exp_zip_batch.append(i['sample'])

        exp_states_batch, exp_action_batch, exp_reward_batch, exp_next_states_batch, \
        exp_done_batch, exp_nstep_rew_batch, exp_nstep_next_batch = map(np.array, zip(*exp_zip_batch))

        is_expert_input = np.ones((batch_size, 1))
        #expert action made into a 2d array for when tf.gather_nd is called during training
        input_exp_action = np.zeros((batch_size, 2))
        input_exp_action[:, 0] = np.arange(batch_size)
        input_exp_action[:, 1] = exp_action_batch
        exp_margin = np.ones((batch_size, action_len)) * exp_margin_constant
        exp_margin[np.arange(batch_size), exp_action_batch] = 0.  # expert chosen actions don't have margin

        loss += inner_train_function(train_model,target_model,exp_buffer,None,
                                    exp_states_batch,exp_action_batch,exp_reward_batch,exp_next_states_batch,
                                    exp_done_batch,exp_nstep_rew_batch,exp_nstep_next_batch,is_expert_input,
                                    input_exp_action,exp_margin,action_len,exp_minibatch,None,
                                     batch_size,gamma,nstep_gamma)

        if current_step % update_every == 0 and current_step >= update_every:
            print("Saving expert training weights at step {}. Loss is {}".format(current_step,loss))
            loss = np.zeros((4,))
            zString = "dqfd_training_weights/expert_model_{}_{}.h5".format(time_int, current_step)
            train_model.save_weights(zString, overwrite=True)
            # updating fixed Q network weights
            target_model.load_weights(zString)


    print('Saving expert final weights. Loss is {}'.format(loss))
    zString = "dqfd_training_weights/expert_model_final_{}_{}.h5".format(time_int, train_steps)
    train_model.save_weights(zString, overwrite=True)

    return train_model, exp_buffer



if __name__ == '__main__':
    # build model, pre train it on human demos, copy it so it is target and initial model, then train network

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='game name', default='SonicTheHedgehog-Genesis')
    parser.add_argument('--state', help='state', default='GreenHillZone.Act1')
    parser.add_argument('--demodir', help='demo directory', default='dqfd_demos/')
    parser.add_argument('--demonumber', help='number or demo files to parse', default=10)
    parser.add_argument('--train_expert',help='train expert only',default=0)
    parser.add_argument('--model_weights', help='weights to load', default='dqfd_training_weights/expert_model_final.h5')
    parser.add_argument('--ebuff_save_name', help='file name for saving expert buffer', default='expert_transitions')
    parser.add_argument('--ebuff_load_name', help='file name for loading expert buffer', default='expert_transitions')
    args = parser.parse_args()

    action_list = [['NOOP'], ['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'], ['DOWN'], ['B']]
    #can only have 1 environment and need a different env for parsing demos vs. running the contest
        #hence can either train the expert (and save replay buffer and trained network) or run the agent and load in the expert training
    if bool(int(args.train_expert)):
        env,n_action = make_env(game=args.game, state=args.state, stack=True, scale_rew=False, scenario='contest',
                       is_contest_env=False,action_list=action_list)

        #unsure on paper's replay buffer size, alpha and beta from the paper. Not sure on epsilon (paper mentions 2 values, this PER only has one)
        expert_buffer = replay.PrioritizedReplayBuffer(500000, alpha=0.4, beta=0.6, epsilon=0.001)
        expert_buffer = fill_expert_buffer(env, expert_buffer, args)

        temp_model = build_model(n_action)
        tgt_model = build_model(n_action)
        tgt_model, expert_buffer = train_expert_model(temp_model, tgt_model, expert_buffer, n_action,
                                                         batch_size=32,
                                                         train_steps=750000)  #10 #750000
        expert_buffer.save_samples(args.ebuff_save_name)

    else:
        print('Training DQfD model')
        env, n_action = make_env(game=args.game, state=args.state, stack=True, scale_rew=False, scenario='contest',
                       is_contest_env=True,action_list=action_list)

        # load expert_replay
        expert_buffer = replay.PrioritizedReplayBuffer(500000, alpha=0.4, beta=0.6, epsilon=0.001)
        expert_buffer.load_samples(args.ebuff_load_name)
        replay_buffer = replay.PrioritizedReplayBuffer(500000, alpha=0.4, beta=0.6, epsilon=0.001)
        # load trained model
        tgt_model = build_model(n_action)
        tgt_model.load_weights(args.model_weights)
        init_model = build_model(n_action)
        init_model.load_weights(args.model_weights)
        #train model
        train_network(env, init_model, tgt_model,expert_buffer,replay_buffer, n_action,max_timesteps=1000000)#,min_buffer_size=50

    print('Training finished')




