#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import copy

def plot_belief(belief):
    
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


def motion_model(action, belief):
    correct_prob = 0.7
    not_move_prob = 0.2
    opposite_prob = 0.1
    tmp_belief = copy.deepcopy(belief)
    forward_move_belief = Forward_move(tmp_belief[:])
    tmp_belief = copy.deepcopy(belief)
    backward_move_belief = Backward_move(tmp_belief[:])
    if action == 1:
        # Forward
        belief = correct_prob * forward_move_belief + not_move_prob * belief + opposite_prob * backward_move_belief
    else:
        # Backwrad
        belief = opposite_prob * forward_move_belief + not_move_prob * belief + correct_prob * backward_move_belief
    return belief

    
def sensor_model(observation, belief, world):
    white_prob = 0.7
    black_prob = 0.9
    for i in range(len(world)):
        if world[i] == 0:
            if observation == 0:
                belief[i] = white_prob * belief[i]
            else:
                belief[i] = (1-white_prob) * belief[i]
        else:
            if observation == 0:
                belief[i] = (1-black_prob) * belief[i]
            else:
                belief[i] = black_prob * belief[i]
    belief = belief / sum(belief)
    return belief

def recursive_bayes_filter(actions, observations, belief, world):
    for i in range(len(actions)):
        belief = motion_model(actions[i], belief)
        belief = sensor_model(observations[i], belief, world)
    return belief

def Forward_move(belief):
    for i in reversed(range(len(belief)-1)):
        belief[i+1] = belief[i]
    belief[0] = 0
    return belief

def Backward_move(belief):
    for i in (range(len(belief)-1)):
        belief[i] = belief[i+1]
    belief[-1] = 0
    return belief
