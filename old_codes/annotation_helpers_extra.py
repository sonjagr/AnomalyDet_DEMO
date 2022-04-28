from common import *
import os,re, sys
import pandas as pd
import numpy as np

def overlap(box_id, boxes):
    c_x, c_y = tuple(box_id.split("-"))
    c_x, c_y = int(c_x), int(c_y)
    ret = False
    for k in range(0,len(boxes)):
        b_x, b_y =  tuple(boxes[k].split("-"))
        b_x, b_y = int(b_x), int(b_y)
        x1min, y1min = c_x, c_y
        x1max, y1max= c_x+BOXSIZE_X, c_y+BOXSIZE_Y
        x2min, y2min = b_x, b_y
        x2max, y2max= b_x+BOXSIZE_X, b_y+BOXSIZE_Y
        if (x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max):
            print('ol')
            ret = True
    return ret

def overlap_click(click_id, boxes, dX, dY):
    c_x, c_y = tuple(click_id.split("-"))
    x1, y1 = int(c_x), int(c_y)
    ret = False
    for k in range(0,len(boxes)):
        b_x, b_y =  tuple(boxes[k].split("-"))
        b_x, b_y = int(b_x), int(b_y)
        x2min, y2min = b_x, b_y
        x2max, y2max= b_x+int(dX), b_y+int(dY)
        if (x1 < x2max and x2min < x1 and y1 < y2max and y2min < y1):
            print('ol')
            ret = True
    return ret

def overlap_bb(box_id, boxes):
    c_x, c_y = tuple(box_id.split("-"))
    c_x, c_y = int(c_x), int(c_y)
    ret = False
    for k in range(0,len(boxes)):
        b_x, b_y =  tuple(boxes[k].split("-"))
        b_x, b_y = int(b_x), int(b_y)
        x1min, y1min = c_x, c_y
        x1max, y1max= c_x+160, c_y+160
        x2min, y2min = b_x, b_y
        x2max, y2max= b_x+160, b_y+160
        if (x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max):
            print('ol')
            ret = True
    return ret

def clicks_to_boxes(clicksX, clicksY, anchors=ANCHOR_GRID):
    box_xs, box_ys = [], []
    dxs, dys = [], []
    for c_x, c_y in zip(clicksX, clicksY):
        c_x = float(c_x)
        c_y = float(c_y)
        x_top, y_top, dx, dy = closest_node(node=[c_x, c_y], nodes=anchors)
        dxs.append(dx)
        dys.append(dy)
        x_top = x_top - 80
        y_top = y_top - 80
        box_x, box_y = x_top, y_top
        box_xs.append(box_x)
        box_ys.append(box_y)
    return box_xs, box_ys, dxs, dys