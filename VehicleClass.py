"""Defines class objects for different types of sprites.

This defines class objects for vehicle sprites following IDM, player sprite,
and background image sprites. In addition, it defines Camera and Map objects.
A function to identify neighbouring sprites of any sprite is defined here.
"""

from Assests import *


# =====================================================================================================

class OtherVeh(pygame.sprite.Sprite):
    """Class to define vehicle sprites other than player vehicle.

    This class defines vehicle sprites that follow IDM rule. Either a car or truck is created based on the probability
    value `proportion_of_trucks'.

    Attributes:
        image: A pygame image representing the sprite.
        rect: A pygame surface object representing the sprite.
        a: Float variable for the IDM parameter maximum acceleration.
        b: Float variable for the IDM parameter maximum decceleration.
        s0: Float variable for the IDM parameter minimum gap.
        T: Float variable for the IDM parameter time headway.
        v0: Float variable for the IDM parameter desired speed.
        length: Float variable for actual length of vehicle in m.
        width: Float variable for actual width of vehicle in m.
        type: Indicator for type of vehicle ( 1-car, 2-truck).
        lane: Indicator for lane of vehicle (1-left lane, 2-right lane).
        v: Float variable for velocity.
        acc: Float variable for acceleration.
        s: Float variable for gap with front vehicle.
        delta_v: Float variable for difference in velocity with front vehicle.
    """

    def __init__(self, lane, position, velocity=0.5*((TRUCKS_PROPORTION* v0_TRUCK)+((1-TRUCKS_PROPORTION)*v0_CAR)), acceleration=0, spacing=0, delta_v=0):
        pygame.sprite.Sprite.__init__(self)
        # init() creates either a truck or a car depending upon 'TRUCKS_PROPORTION'
        if rand() < TRUCKS_PROPORTION:
            self.image = enemytruck_image
            self.a, self.b, self.s0, self.T, self.v0 = random.gauss(a_TRUCK, sigma_a_TRUCK), random.gauss(b_TRUCK,
                                                                                                          sigma_b_TRUCK), random.gauss(
                s0_TRUCK, sigma_s0_TRUCK), random.gauss(T_TRUCK, sigma_T_TRUCK), random.gauss(v0_TRUCK, sigma_v0_TRUCK)
            self.type = 2
        else:
            # 2 different images are used for IDM cars
            image_number = random.randint(1, 2)
            if image_number == 1:
                self.image = enemycar_image1
            else:
                self.image = enemycar_image2
            self.a, self.b, self.s0, self.T, self.v0 = random.gauss(a_CAR, sigma_a_CAR), random.gauss(b_CAR,
                                                                                                      sigma_b_CAR), random.gauss(
                s0_CAR, sigma_s0_CAR), random.gauss(T_CAR, sigma_T_CAR), random.gauss(v0_CAR, sigma_v0_CAR)
            self.type = 1
        self.rect = self.image.get_rect()
        self.length = self.rect.height  # in pixels
        self.width = self.rect.width
        self.lane = lane
        if lane == 1:
            self.rect.x = LEFT_LANE_MID - (self.width / 2)
        else:
            self.rect.x = RIGHT_LANE_MID - (self.width / 2)
        self.rect.y = position
        self.v, self.acc, self.s, self.delta_v = velocity, acceleration, spacing, delta_v

    def update(self):

        s_star = self.s0 + max(0, (self.v * self.T) + ((self.v * self.delta_v)/(2 * numpy.power(self.a * self.b, 0.5))))

        self.acc = self.a * (1 - numpy.power(self.v / self.v0, IDM_DELTA) - numpy.power(s_star / self.s, 2))
        old_velocity = self.v
        self.v = clamp((old_velocity)+(self.acc*DELTA_T),0,self.v0)
        self.acc = (self.v - old_velocity)/DELTA_T  # adjustment to acc due to clamping of velocity

        """
        # Expected gap
        if self.vehicle_in_front_same_lane == None:
            expected_gap = 100000000000
        else:
            expected_gap = self.s + (self.vehicle_in_front_same_lane.v * DELTA_T) + (
                    0.5 * self.vehicle_in_front_same_lane.acc * DELTA_T * DELTA_T)

        acc_upper_cutoff = min(1.5 * self.a, 2 * (expected_gap + self.s0 - (self.v * DELTA_T)) / (DELTA_T * DELTA_T))
        if self.acc > acc_upper_cutoff:
            self.acc = acc_upper_cutoff
        acc_lower_cutoff = max(-1 * self.b, -1 * self.v / DELTA_T)
        if self.acc < acc_lower_cutoff:
            self.acc = acc_lower_cutoff
        """

        distance_moved = (self.v * DELTA_T) - (0.5 * self.acc * DELTA_T * DELTA_T)  # in meters
        self.rect.y = self.rect.y - convert_to_pixels(distance_moved)

# ========================================================================================================================
class PlayerCar(pygame.sprite.Sprite):
    """Class to define player car.

    This class defines vehicle sprite that is controlled by the player.

    Attributes:
        image: A pygame image representing the sprite.
        rect: A pygame surface object representing the sprite.
        length: Float variable for actual length of vehicle in m.
        width: Float variable for actual width of vehicle in m.
        type: Indicator for type of vehicle ( 1-car, 2-truck).
        lane: Indicator for lane of vehicle (1-left lane, 2-right lane).
        v: Float variable for velocity.
        acc: Float variable for acceleration.
        max_velocity: Float variable for maximum velocity allowed.
    """

    def __init__(self, lane, position, velocity, acceleration):
        pygame.sprite.Sprite.__init__(self)
        self.image = mycar_image
        self.rect = self.image.get_rect()
        self.length = self.rect.height  # length and width are in pixels
        self.width = self.rect.width
        self.lane = lane

        if self.lane == 1:
            self.rect.x = LEFT_LANE_MID - (self.width / 2)
        else:
            self.rect.x = RIGHT_LANE_MID - (self.width / 2)
        self.rect.y = position
        self.v, self.acc = velocity, acceleration

    def update(self):
        x_change = 0
        y_change = 0
        action = 0  # corresponding to no key pressed

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            if self.rect.top < PIXEL_ROAD_LENGTH - (ROAD_LENGTH_MULTIPLIER * ROAD_IMAGE_RECT.height):
                pass
            elif self.lane == 2:
                x_change = -(RIGHT_LANE_MID - LEFT_LANE_MID)
                self.lane = 1
                action = 1
            self.acc = 0 - PLAYER_FRICTION_DECC
            """else:
                game.screen.blit(game.NoLeftShift_message_text,(50,50)) 
                pygame.display.flip()"""
        if keys[pygame.K_RIGHT]:
            if self.rect.top < PIXEL_ROAD_LENGTH - (ROAD_LENGTH_MULTIPLIER * ROAD_IMAGE_RECT.height):
                pass
            elif self.lane == 1:
                x_change = (RIGHT_LANE_MID - LEFT_LANE_MID)
                self.lane = 2
                action = 2
            self.acc = 0 - PLAYER_FRICTION_DECC
            """else:
                game.screen.blit(game.NoRightShift_message_text,(50,50))
                pygame.display.flip()"""
        if keys[pygame.K_UP]:
            self.acc = PLAYER_ACCELERATION_STEP - PLAYER_FRICTION_DECC
            action = 3
        if keys[pygame.K_DOWN]:
            self.acc = - PLAYER_DECCELERATION_STEP - PLAYER_FRICTION_DECC
            action = 4

        self.v = clamp(self.v + (self.acc * DELTA_T), 0, PLAYER_MAX_VELOCITY)
        self.acc = -self.v / DELTA_T  # taking -u/t part of a=(v-u)/t
        self.acc = self.acc + (self.v / DELTA_T)  # adding v/t part of a=(v-u)/t
        y_change = (self.v * DELTA_T) - (0.5 * self.acc * DELTA_T * DELTA_T)  # s= vt-at2/2

        self.rect.x = self.rect.x + x_change
        self.rect.y = self.rect.y - convert_to_pixels(y_change)

        return action

# =========================================================================================================================================
class Camera:
    """Class to define camera.

    This class defines a pygame surface object which keeps its top-left corner coincided with player sprites's top-left
    corner. Camera follows the player car while rendering by shifting all sprites such that top-left corner of camera is
    at the center of screen.

    Attributes:
        camera: A pygame Rect object.
        height: Length of camera.
        width: Width of camera.
    """

    def __init__(self, width, height):
        self.camera = pygame.Rect(0, 0, width, height)
        self.width = width
        self.height = height

    def apply(self, entity):
        return entity.rect.move(self.camera.topleft)

    def update(self, target):
        # x = -target.rect.x + int(Width_of_Screen/2)
        x = 0
        y = -target.rect.y + int(DISPLAY_HEIGHT / 2)
        """x=min(0,x) #left
        y=min(0,y) #top
        x=max(-(self.width-Width_of_Screen),x)
        y=max(-(self.height-Length_of_Screen),y)"""
        self.camera = pygame.Rect(x, y, self.width, self.height)


# ================================================================================================
class Map:
    """Class to define map.

    This class stores the map settings.

    Attributes:
        data: A list of strings to store map.
        height: Length of map.
        width: Width of map.
    """

    def __init__(self, filename):
        self.data = []
        with open(filename, 'rt') as f:
            for line in f:
                self.data.append(line.strip())

        self.width = len(self.data[0]) * BCKGRND_TILESIZE
        self.height = len(self.data) * BCKGRND_TILESIZE


class Background_Sprite(pygame.sprite.Sprite):
    """Class to define Background sprites.

    This class defines sprites that are used as background. Here, the ROAD_IMAGE is not cropped.

    Attributes:
        image: A pygame image object representing the sprite.
        rect: A pygame Rect object representing the sprite.
    """

    def __init__(self, x=0, y=0, image=ROAD_IMAGE):
        pygame.sprite.Sprite.__init__(self)
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
