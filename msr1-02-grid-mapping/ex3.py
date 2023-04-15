#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

def plot_gridmap(gridmap):
    plt.figure(figsize=(10,10))
    plt.imshow(gridmap, cmap='Greys',vmin=0, vmax=1)
    
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape)/2
    if pose.size == 3:
        new_pose = np.zeros(2)
        
        new_pose[0] = np.round(pose[0]/map_res) + origin[0]
        new_pose[1] = np.round(pose[1]/map_res) + origin[1]
        
    else:
        new_pose = np.zeros((2, pose.shape[1]))
        
        new_pose[0, :] = np.round(pose[0, :]/map_res) + origin[0]
        new_pose[1, :] = np.round(pose[1, :]/map_res) + origin[1]
    return new_pose.astype(int)

def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr    

def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle + (num_beams*angular_res), num_beams)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2,:]
    return m_points

def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose  

def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l
    
def logodds2prob(log_odds):
    prob = 1 -  (1 / (1 + np.exp(log_odds)))
    return prob

def prob2logodds(prob):
    if prob == 1:
        prob = 0.99
    log_odds = np.log(prob / (1 - prob))
    return log_odds
    
def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    check_cell = bresenham(cell[0], cell[1], endpoint[0], endpoint[1])
    if check_cell.shape[0] == 0:
        free_cell = np.array([])
        occ_cell = np.array([])
    elif check_cell.shape[0] == 1:
        free_cell = np.array([])
        occ_cell = np.array([check_cell[0,0], check_cell[0, 1]])
    elif check_cell.shape[0] > 1:
        free_cell = np.array(check_cell[0:-1,:])
        occ_cell = np.array(check_cell[-1,:])
    return free_cell, occ_cell

def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior):
    m_current_point = poses2cells(poses_raw, occ_gridmap, map_res)
    m_points = ranges2cells(ranges_raw, poses_raw, occ_gridmap, map_res)

    for i in range(m_points.shape[1]):
        x_ = m_points[0, i]
        y_ = m_points[1, i]
        
        free_cell, occ_cell = inv_sensor_model(m_current_point, [x_, y_], prob_occ, prob_free)
        
        # free_cell
        for j in range(free_cell.shape[0]):
            log_odds_ = prob2logodds(occ_gridmap[free_cell[j,0], free_cell[j,1]]) + prob2logodds(prob_free) - prob2logodds(prior)
            occ_gridmap[free_cell[j,0], free_cell[j,1]] = logodds2prob(log_odds_)
        # occ_cell
        log_odds_ = prob2logodds(occ_gridmap[occ_cell[0], occ_cell[1]]) + prob2logodds(prob_occ) - prob2logodds(prior)
        occ_gridmap[occ_cell[0], occ_cell[1]] = logodds2prob(log_odds_)
    
    return occ_gridmap