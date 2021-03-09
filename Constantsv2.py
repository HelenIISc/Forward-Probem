import pygame
import numpy
import random
from numpy.random import rand
import sys
import itertools

backgroundtilesize=10

road_image=pygame.image.load("road.png")
road_image_rect=road_image.get_rect()
if road_image_rect.width % backgroundtilesize !=0 and road_image_rect.height % backgroundtilesize!=0:
    print("Background tile size not compatible")
    sys.exit()  #sys.exit("Background tile size not compatible")


FPS=5
delta_t=1/FPS 
pixel_conversion=20  #note: this is set such that length of screen is 800
FLOW=0.1 #in vehicles/second on total 
#Length_of_road=10 #in meters
Warm_up_time=2 #in seconds

proportion_of_trucks=0.3


player_start_velocity=(1*5/18)
player_acceleration_step=0.00000001  # write how these values came here
player_decceleration_step=0.00000001
player_friction_decc = 0.0000000075
player_max_velocity = (150*5/18)




background_creater_constant=2
background_tilesize=int(pixel_conversion/background_creater_constant)
Length_of_Screen=800
Width_of_Screen=road_image_rect.width
entry_point_of_player=road_image_rect.height
length_of_car=4*pixel_conversion
width_of_car= int(1.8*pixel_conversion)
length_of_truck=7*pixel_conversion
width_of_truck=2*pixel_conversion

left_lane_mid=210  #have to do this manually
right_lane_mid=280




delta=4
#VEHICLE PARAMETERS-CAR
a_CAR=0.3
b_CAR=3
s0_CAR=2
T_CAR=1.5
v0_CAR=(80*5/18)
sigma_a_CAR=0.1/3 #sigma chosen such that even in worst case 3sigma, values are realistic
sigma_b_CAR=1/3
sigma_s0_CAR=0.5/3
sigma_T_CAR=0.5/3
sigma_v0_CAR=40/3
#VEHICLE PARAMETERS-TRUCK
a_TRUCK=0.3
b_TRUCK=2
s0_TRUCK=2
T_TRUCK=1.7
v0_TRUCK=(20*5/18)
sigma_a_TRUCK=0.1/3
sigma_b_TRUCK=1/3
sigma_s0_TRUCK=0.5/3
sigma_T_TRUCK=0.5/3
sigma_v0_TRUCK=20/3