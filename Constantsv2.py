"""Defines constants used.

Libraries like pygame, numpy etc. are imported here.Also, various
constants like IDM parameters, dimensions of vehicles, frame rate,
initial parameter  values of player vehicle are set here.
"""

import pygame
import numpy
import random
from numpy.random import rand
import sys
import itertools

# =========Background Road==============================================================================================
ROAD_IMAGE = pygame.image.load("road.png")
ROAD_IMAGE_RECT = ROAD_IMAGE.get_rect()
ROAD_LENGTH_MULTIPLIER = 3
PIXEL_ROAD_LENGTH = ROAD_LENGTH_MULTIPLIER * ROAD_IMAGE_RECT.height  # in pixel
# =========Dimensions of Display Window in pixels=======================================================================
DISPLAY_HEIGHT = 800
DISPLAY_WIDTH  = ROAD_IMAGE_RECT.width
# TODO: check for changes due to changing the display width from image.width to a constant

# ==========Dimensions of Moving Sprites in m===========================================================================
CAR_LENGTH = 4
CAR_WIDTH = 1.8
TRUCK_LENGTH = 7
TRUCK_WIDTH = 2

# =========Traffic Characteristics======================================================================================
FLOW = 0.1  # in vehicles/second  for all lanes combined
TRUCKS_PROPORTION = 0.3

#============ Time Settings============================================================================================
FPS = 5
DELTA_T = 1 / FPS

# ===============Player Settings=======================================================================================
PLAYER_ACCELERATION_STEP = 0.00000001  # write how these values came here
PLAYER_DECCELERATION_STEP = 0.00000001
PLAYER_FRICTION_DECC = 0.0000000075
PLAYER_MAX_VELOCITY = (150 * 5 / 18)

# ========================= IDM PARAMETERS in m,s ======================================================================
IDM_DELTA = 4
# VEHICLE PARAMETERS-CAR
a_CAR = 0.3
b_CAR = 3
s0_CAR = 2
T_CAR = 1.5
v0_CAR = (80 * 5 / 18)
sigma_a_CAR = 0
sigma_b_CAR = 0
sigma_s0_CAR = 0
sigma_T_CAR = 0
sigma_v0_CAR = 0
# VEHICLE PARAMETERS-TRUCK
a_TRUCK = 0.3
b_TRUCK = 2
s0_TRUCK = 2
T_TRUCK = 1.7
v0_TRUCK = (20 * 5 / 18)
sigma_a_TRUCK = 0
sigma_b_TRUCK = 0
sigma_s0_TRUCK = 0
sigma_T_TRUCK = 0
sigma_v0_TRUCK = 0
#================actual distances in m to pixels==========================================
PIXEL_CONVERSION_FACTOR = 20
#================= x coordinates of middle of lanes=================================
LEFT_LANE_MID = 210  # in pixels
RIGHT_LANE_MID = 280



# Length_of_road=10 #in meters
WARMUP_TIME = 2  # in seconds



PLAYER_START_VELOCITY = (10 * 5 / 18)




PLAYER_ENTRY_POINT = ROAD_IMAGE_RECT.height






