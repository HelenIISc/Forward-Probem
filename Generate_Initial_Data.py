# disable draw() and time.click in simulator.py to run this code fast, writing features in each step disabled

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


# TODO: collect data, IRL
# =====================================================================================================
class Initial_Data_Collector:
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
        self.all_vehicles = pygame.sprite.Group()
        self.input_waiting_vehicles = pygame.sprite.Group()
        # ========declaring pygame display elements========
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.display.set_caption("IDM")
        self.camera = Camera(1000, 1000)  # entering arbitrary values for length and width of camera
        # ==========background sprites================================
        self.background_tiles = pygame.sprite.Group()
        self.create_background_sprites(ROAD_LENGTH_MULTIPLIER)
        # =============files===================================
        self.f = open("vehicle_data.txt", "a")
        # =============== MDP ===================================
        self.state = None
        self.action = None
        # ===============INPUT FLOW AT ENTRY====================================
        self.input_vehicles = 0

        # TODO: get rid of this clock, feels it wastes time
        self.clock = pygame.time.Clock()

        # TODO: how to insert message that lasts a few seconds
        """message_font=pygame.font.SysFont("None",20)
        self.NoLeftShift_message_text=message_font.render("Already in Left lane",0,(255,255,255))
        self.NoRightShift_message_text=message_font.render("Already in Right lane",0,(255,255,255))
        self.MaxSpeed_message_text=message_font.render("Attained maximum speed",0,(255,255,255))"""

    def reset(self, seed, draw=0):
        self.all_vehicles.empty()
        self.input_waiting_vehicles.empty()
        # pygame.quit()
        Vehicle_data = self.read_vehicle_data(seed)  # note this is a list of strings
        pygame.init()
        self.create_initial_vehicles(Vehicle_data)
        self.player = random.choice(self.all_vehicles.sprites())
        self.camera.update(self.player)
        if draw ==1:
            self.draw()

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
            self.all_vehicles.add(
                OtherVeh(int(vehicle_data[1]), PIXEL_ROAD_LENGTH - convert_to_pixels(float(vehicle_data[2]))))

    def front_vehicle(self, vehicle):
        same_lane_vehicles = [sprite for sprite in self.all_vehicles if sprite.lane == vehicle.lane]
        same_lane_vehicles_front = [sprite for sprite in same_lane_vehicles if vehicle.rect.top - sprite.rect.top > 0]
        if len(same_lane_vehicles_front) > 0:
            vehicle_front_same_lane = min([sprite for sprite in same_lane_vehicles_front],
                                          key=lambda sprite: vehicle.rect.top - sprite.rect.top)
        else:
            vehicle_front_same_lane = None
        return vehicle_front_same_lane

    # TODO: check if rendering all vehicles or not, check if way sto reduce that

    def step(self,draw=0):
        self.check_for_quit()
        for veh in self.all_vehicles:
            self.update_gap_velocitydifference(veh)
        self.all_vehicles.update()
        self.camera.update(self.player)
        if draw == 1:
            self.draw()
        self.generate_incoming_vehicles()
        print(self.player.rect.y)

    def update_gap_velocitydifference(self, veh):
        front_vehicle = self.front_vehicle(veh)
        if front_vehicle == None:
            veh.delta_v = -10000000000
            veh.s = 10000000000
        else:
            veh.delta_v = veh.v - front_vehicle.v
            veh.s = (veh.rect.top - front_vehicle.rect.top) / PIXEL_CONVERSION_FACTOR



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
        if self.input_vehicles >= 1:
            for _ in range(int(self.input_vehicles)):
                self.input_vehicles = self.input_vehicles - 1
                self.input_waiting_vehicles.add(OtherVeh(random.randint(1, 2), PIXEL_ROAD_LENGTH))
        for veh in self.input_waiting_vehicles:
            self.update_gap_velocitydifference(veh)
            if veh.s > veh.length + veh.s0:
                self.input_waiting_vehicles.remove(veh)
                self.all_vehicles.add(veh)
        self.input_vehicles = self.input_vehicles + (FLOW * DELTA_T)

    # TODO: merge other_vehicles and all_vehicles as same, dont keep seperate copies-but how to do that??

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


game = Initial_Data_Collector()
game.reset(1, 1)
for _ in range(100):
    game.step(1)
file = open("vehicle_data.txt", "a")
for veh in game.all_vehicles:
    veh_data=[veh.type, veh.lane, veh.rect.y,veh.v,veh.acc,veh.s,veh.delta_v]
    file.write(str(veh_data))
    file.write('/n')
