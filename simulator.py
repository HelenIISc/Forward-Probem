"""Defines the Game class.

Game class is used to create traffic simulator which can add/delete vehicles,
update vehicle positions, speed etc. at every frame, collect data from human driver,
etc.

  Typical usage example:

  game = Game()
  game.new()
  game.main_game(<episode length>, <episode number>)
"""
from os import path
from VehicleClass import *


# =====================================================================================================
class Game:
    """Class to define and use the simulator.

    This class creates a road traffic simulator using sprites of `OtherVeh`, 'PlayerCar' and 'Background_Tile' instances.
    Game class can be used to collect data from player.

    Attributes:
        screen: A pygame window object for rendering.
        clock: A pygame object for operations with time.
        f:  A text file for writing data collected during game.
        map: A text file with background settings.
        all_vehicles: Sprite group for all vehicles.
        other_vehicles: Sprite group for all vehicles except the player vehicle.
        waiting_vehicles_left: Sprite group for all vehicles waiting to enter left lane.
        waiting_vehicles_right: Sprite group for all vehicles waiting to enter right lane.
        background_tiles: Sprite group for all background sprites.
        seconds_elapsed_checkpoint: Float variable to track time.
        vehicle_created_counter: float variable used in generation of vehicles at entry point according to input flow value.
        player_entered: Indicator for player car entering road.
        player_car: Pygame sprite used as player car.
        camera: Camera instance to move screen along with player_car.
    """

    def __init__(self):
        """
        :rtype: object
        """
        # ========declaring sprites and groups for vehicles================
        self.player = pygame.sprite.Sprite()
        self.all_vehicles = pygame.sprite.Group()
        self.other_vehicles = pygame.sprite.Group()
        self.waiting_vehicles_left = pygame.sprite.Group()
        self.waiting_vehicles_right = pygame.sprite.Group()
        # ========declaring pygame display elements========
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.display.set_caption("IDM")
        self.camera = Camera(100, 100)  # entering arbitrary values for length and width of camera
        # ==========background sprites================================
        self.background_tiles = pygame.sprite.Group()
        self.create_background_sprites(ROAD_LENGTH_MULTIPLIER)
        # =============files===================================
        self.f = open("trajectory_data.txt", "a")
        # =============== MDP ===================================
        self.state = None
        self.action = None
        #=========================================================
        self.seconds_elapsed_checkpoint = 0
        self.vehicle_created_counter = 0
        self.player_entered = 0

        # TODO: get rid of this clock, feels it wastes time
        self.clock = pygame.time.Clock()

        # TODO: how to insert message that lasts a few seconds
        """message_font=pygame.font.SysFont("None",20)
        self.NoLeftShift_message_text=message_font.render("Already in Left lane",0,(255,255,255))
        self.NoRightShift_message_text=message_font.render("Already in Right lane",0,(255,255,255))
        self.MaxSpeed_message_text=message_font.render("Attained maximum speed",0,(255,255,255))"""

    def reset(self, seed):
        self.all_vehicles.empty()
        self.other_vehicles.empty()
        self.waiting_vehicles_left.empty()
        self.waiting_vehicles_right.empty()
        pygame.quit()
        Vehicle_data = self.read_vehicle_data(seed)  # note this is a list of strings
        pygame.init()
        self.create_initial_vehicles(Vehicle_data)
        self.draw()
        self.state = self.extract_state_features()
        return self.state

    def read_vehicle_data(self, seed):
        Vehicle_data = []
        seed_text = str(seed)
        file_name = 'Seed_' + seed_text + '.txt'
        file = open(file_name, 'r')
        file_contents = file.read().splitlines()
        for i in range(len(file_contents)):
            Vehicle_data.append(file_contents[i][1:len(file_contents[i]) - 1].split(
                ','))  # range is kept like this to avoid an extra pair of list square brackets
        return Vehicle_data

    def create_background_sprites(self, k):
        """Creates background sprites.

        This functions creates sprites which are used as background image. Each sprite is a road_image asset. `k' number
        of sprites are kept lengthwise to create a very long road background.

        Args:
            k: int variable indicating how many times road_image has to be repeated to create the full road length.

        Returns:
            None

        Raises:
            None
        """

        for y in range(0, (k * ROAD_IMAGE_RECT.height), ROAD_IMAGE_RECT.height):
            self.background_tiles.add(Background_Sprite(0, y))

    def create_initial_vehicles(self, Vehicle_data):
        for vehicle_data in Vehicle_data:
            if int(vehicle_data[0]) == 3:
                self.player = PlayerCar(int(vehicle_data[1]), PIXEL_ROAD_LENGTH - convert_to_pixels(float(vehicle_data[2])),
                                        float(vehicle_data[3]), float(vehicle_data[4]))
                self.all_vehicles.add(self.player)
            else:
                self.all_vehicles.add(
                    OtherVeh(int(vehicle_data[1]), PIXEL_ROAD_LENGTH - convert_to_pixels(float(vehicle_data[2])), float(vehicle_data[3]),
                             float(vehicle_data[4])))
                self.other_vehicles.add(
                    OtherVeh(int(vehicle_data[1]), PIXEL_ROAD_LENGTH - convert_to_pixels(float(vehicle_data[2])), float(vehicle_data[3]),
                             float(vehicle_data[4])))

    def extract_state_features(self):
        vehicle_front_same_lane, vehicle_back_same_lane, vehicle_front_other_lane, vehicle_back_other_lane \
            = self.neighbouring_vehicles(self.player)
        if vehicle_front_same_lane == None:
            spacing_front_same_lane = 100000000000
            velocity_front_same_lane = 0
        else:
            spacing_front_same_lane = (self.player.rect.y - vehicle_front_same_lane.rect.y) / PIXEL_CONVERSION_FACTOR
            velocity_front_same_lane = vehicle_front_same_lane.v
        if vehicle_front_other_lane == None:
            spacing_front_other_lane = 100000000000
            velocity_front_other_lane = 0
        else:
            spacing_front_other_lane = (self.player.rect.y - vehicle_front_other_lane.rect.y) / PIXEL_CONVERSION_FACTOR
            velocity_front_other_lane = vehicle_front_other_lane.v

        if vehicle_back_other_lane == None:
            spacing_back_other_lane = 1000000000000000000
            velocity_back_other_lane = 0
        else:
            spacing_back_other_lane = (vehicle_back_other_lane.rect.y - self.player.rect.y) / PIXEL_CONVERSION_FACTOR
            velocity_back_other_lane = vehicle_back_other_lane.v

        feature_list = [spacing_front_same_lane, spacing_front_other_lane, spacing_back_other_lane,
                        velocity_front_same_lane, velocity_front_other_lane, velocity_back_other_lane]
        return feature_list

    def neighbouring_vehicles(self, vehicle):
        same_lane_vehicles = [sprite for sprite in self.all_vehicles if sprite.lane == vehicle.lane]
        same_lane_vehicles_front = [sprite for sprite in same_lane_vehicles if vehicle.rect.top - sprite.rect.top > 0]
        same_lane_vehicles_back = [sprite for sprite in same_lane_vehicles if vehicle.rect.top - sprite.rect.top < 0]
        if len(same_lane_vehicles_front) > 0:
            vehicle_front_same_lane = min([sprite for sprite in same_lane_vehicles_front],
                                          key=lambda sprite: vehicle.rect.top - sprite.rect.top)
        else:
            vehicle_front_same_lane = None
        if len(same_lane_vehicles_back) > 0:
            vehicle_back_same_lane = min([sprite for sprite in same_lane_vehicles_back],
                                         key=lambda sprite: sprite.rect.top - vehicle.rect.top)
        else:
            vehicle_back_same_lane = None
        other_lane_vehicles = [sprite for sprite in self.all_vehicles if not sprite.lane == vehicle.lane]
        other_lane_vehicles_front = [sprite for sprite in other_lane_vehicles if vehicle.rect.top - sprite.rect.top > 0]
        other_lane_vehicles_back = [sprite for sprite in other_lane_vehicles if vehicle.rect.top - sprite.rect.top < 0]
        if len(other_lane_vehicles_front) > 0:
            vehicle_front_other_lane = min([sprite for sprite in other_lane_vehicles_front],
                                           key=lambda sprite: vehicle.rect.top - sprite.rect.top)
        else:
            vehicle_front_other_lane = None
        if len(other_lane_vehicles_back) > 0:
            vehicle_back_other_lane = min([sprite for sprite in other_lane_vehicles_back],
                                          key=lambda sprite: sprite.rect.top - vehicle.rect.top)
        else:
            vehicle_back_other_lane = None
        return vehicle_front_same_lane, vehicle_back_same_lane, vehicle_front_other_lane, vehicle_back_other_lane

    def front_vehicle(self, vehicle):
        same_lane_vehicles = [sprite for sprite in self.all_vehicles if sprite.lane == vehicle.lane]
        same_lane_vehicles_front = [sprite for sprite in same_lane_vehicles if vehicle.rect.top - sprite.rect.top > 0]
        if len(same_lane_vehicles_front) > 0:
            vehicle_front_same_lane = min([sprite for sprite in same_lane_vehicles_front],
                                          key=lambda sprite: vehicle.rect.top - sprite.rect.top)
        else:
            vehicle_front_same_lane = None
        return vehicle_front_same_lane

    def step(self):
        self.check_for_quit()
        self.generate_incoming_vehicles()
        self.insert_veh_leftlane()
        self.insert_veh_rightlane()
        self.update_gap_velocitydifference()
        self.action= self.player.update()
        self.all_vehicles.update()
        self.state = self.extract_state_features()
        self.camera.update(self.player_car)
        self.draw()
        return self.state

    def write_data(self):
        pass

    def update_gap_velocitydifference(self):
        for veh in self.other_vehicles:
            front_veh = self.front_vehicle(veh)
            if front_veh == None:
                veh.delta_v = -10000000000
                veh.s = 10000000000
            else:
                veh.delta_v = veh.v - front_vehicle.v
                veh.s = (veh.rect.top - front_vehicle.rect.top) / PIXEL_CONVERSION_FACTOR

    def main_game(self, num_steps_for_each_trajectory, trajectory_number):
        """Player playing the game.

        This is the main part of the game where the player plays by moving the player car. The player car is inserted
        at 'entry_point_of_player'. The game is played for `num_steps_for_each_trajectory' time steps, in each of which,
        features (state space) of the player car and action taken by player car are written to a text file.

        Args:
            num_steps_for_each_trajectory: int variable indicating number of time steps for the game
            trajectory_number: int variable indicating trajectory in data collection . This is not used in any
            calculations here, but only for writing to the text file.

        Returns:
            None

        Raises:
            None
        """
        # display start of game
        self.player_car = PlayerCar(PLAYER_ENTRY_POINT, random.randint(1, 2))
        # message_font=pygame.font.SysFont("None",50)
        # self.screen.blit(message_font.render("START GAME",0,(0,0,255)),(200,50))
        # pygame.display.flip()
        for step_no in range(num_steps_for_each_trajectory + 1):

            self.insert_player()
            self.generate_incoming_vehicles()
            self.insert_veh_leftlane()
            self.insert_veh_rightlane()
            # self.all_vehicles.update(self)
            self.other_vehicles.update(self)
            feature_list = self.player_car.update(self)
            if self.player_entered:
                row_data = [trajectory_number, step_no]
                for f in feature_list:
                    row_data.append(f)
                    # Data.append(row_data)
                self.f.write(str(row_data))
                self.f.write('/n')
            self.camera.update(self.player_car)
            collided_list = pygame.sprite.spritecollide(self.player_car, self.other_vehicles, False)
            if collided_list:
                print("Collision")
            self.draw()
            self.clock.tick(FPS)
        self.f.close()

    def generate_incoming_vehicles(self):
        """Create vehicles as per input flow value.

        In every time step, float number of vehicles (usually fractional value) are added to `vehicle_created_counter'
        according to input flow. If `vehicle_created_counter' equals or exceeds 1 in any time step, then a vehicle of
        `OtherVeh' instance is  added randomly to `waiting_vehicles_left' or `waiting_vehicles_right' at the entry point
        of road. The vehicle sprites created are stacked up on top of each other.

        Args:
                None

        Returns:
                None

        Raises:
                None
        """
        # TODO: make this function direct by fps conversions, remove seconds_elapsed
        time = pygame.time.get_ticks() / 1000
        self.vehicle_created_counter = self.vehicle_created_counter + ((time - self.seconds_elapsed_checkpoint) * FLOW)
        self.seconds_elapsed_checkpoint = time
        if self.vehicle_created_counter >= 1:
            self.vehicle_created_counter = self.vehicle_created_counter - 1  # will work only till 1veh/sec
            if random.randint(1, 2) == 1:
                self.waiting_vehicles_left.add(OtherVeh(ROAD_IMAGE_RECT.height, 1, self))
            else:
                self.waiting_vehicles_right.add(OtherVeh(ROAD_IMAGE_RECT.height, 2, self))

    def check_for_quit(self):
        """Check if user quits pygame window.

        This function closes the pygame window if event type in any frame is detected as quit.

        Args:
                None

        Returns:
                None

        Raises:
                None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def insert_veh_leftlane(self):
        """Adds vehicle from `waiting_vehicles_left' to `all_vehicles' subject to conditions.

        This functions selects a vehicle sprite randomly from `waiting_vehicles_left' and finds its neighbouring
        vehicles. If there is enough space/gap  for the sprite to enter the left lane, then the sprite is moved onto the road
        by removing it from `waiting_vehicles_left' and adding to `all_vehicles'.

        Args:
                None

        Returns:
                None

        Raises:
                None
        """

        # if self.waiting_vehicles_left:
        if False:
            sprite = random.choice(self.waiting_vehicles_left.sprites())
            # TODO: avoid finding neighbouring fully, just find front veh
            vehicle_in_front_same_lane, vehicle_in_back_same_lane, vehicle_in_front_other_lane, vehicle_in_back_other_lane = neighbouring_vehicles(
                sprite, self.all_vehicles)
            if vehicle_in_front_same_lane == None:
                sprite.rect.y = ROAD_IMAGE_RECT.height - sprite.length
                self.all_vehicles.add(sprite)
                self.waiting_vehicles_left.remove(sprite)
            """elif sprite.rect.top-vehicle_in_front_same_lane.rect.top-vehicle_in_front_same_lane.length >  (sprite.s0*pixel_conversion)+sprite.length:
                sprite.rect.y=road_image_rect.height-sprite.length
                self.all_vehicles.add(sprite)
                self.waiting_vehicles_left.remove(sprite)"""

    def insert_veh_rightlane(self):
        """Adds vehicle from `waiting_vehicles_right' to `all_vehicles' subject to conditions.

        This functions done the same as `insert_veh_leftlane()', but for the right lane case.

        Args:
                None

        Returns:
                None

        Raises:
                None
        """
        # if self.waiting_vehicles_right:
        if False:
            sprite = random.choice(self.waiting_vehicles_right.sprites())
            vehicle_in_front_same_lane, vehicle_in_back_same_lane, vehicle_in_front_other_lane, vehicle_in_back_other_lane = neighbouring_vehicles(
                sprite, self.all_vehicles)
            if vehicle_in_front_same_lane == None:
                sprite.rect.y = ROAD_IMAGE_RECT.height - sprite.length
                self.all_vehicles.add(sprite)
                self.waiting_vehicles_right.remove(sprite)
            """elif sprite.rect.top-vehicle_in_front_same_lane.rect.top-vehicle_in_front_same_lane.length >  (sprite.s0*pixel_conversion)+sprite.length:
                sprite.rect.y=road_image_rect.height-sprite.length
                self.all_vehicles.add(sprite)
                self.waiting_vehicles_right.remove(sprite)"""

    def insert_player(self):
        """Insert player vehicle on road.

        If `player_entered' is 0, this function checks for space/gap for `player_car' to enter the road at
        `entry_point_of_player'. If sufficient space is found, 'player_car' is added to 'all_vehicles'.

        Args:
                None

        Returns:
                None

        Raises:
                None
        """
        if not self.player_entered:
            """vehicle_in_front_same_lane,vehicle_in_back_same_lane,vehicle_in_front_other_lane,vehicle_in_back_other_lane=neighbouring_vehicles(self.player_car,self.all_vehicles)
            if not vehicle_in_front_same_lane==None:
                if self.player_car.rect.top-vehicle_in_front_same_lane.rect.top-vehicle_in_front_same_lane.length >  1.5*(self.player_car.length):
                    if vehicle_in_back_same_lane.rect.top-self.player_car.rect.bottom > self.player_car.length + (vehicle_in_back_same_lane.s0*pixel_conversion):
                        self.player_car.rect.y=entry_point_of_player
                        self.all_vehicles.add(self.player_car)
                        self.player_entered=1
                        print("player_entered")
            elif vehicle_in_back_same_lane.rect.top-self.player_car.rect.bottom > self.player_car.length + (vehicle_in_back_same_lane.s0*pixel_conversion):
                self.player_car.rect.y=entry_point_of_player
                self.all_vehicles.add(self.player_car)
                self.player_entered=1
                print("player_entered")"""
            self.all_vehicles.add(self.player_car)
            self.player_car.rect.y = PLAYER_ENTRY_POINT
            self.player_entered = 1
            print("player_entered")

    def draw(self):
        """Does rendering in pygame window.

        Al sprites in the groups `background_tiles' and `all_vehicles' are drawn on screen and screen is updated.

        Args:
                None

        Returns:
                None

        Raises:
                None
        """
        for sprite in self.background_tiles:
            self.screen.blit(sprite.image, self.camera.apply(sprite))
        # TODO: check camera apply on background sprites
        for sprite in self.all_vehicles:
            self.screen.blit(sprite.image, self.camera.apply(sprite))
        # pygame.display.flip()
        pygame.display.update()



game=Game()
game.reset(1)
while True:
    state=game.step()

