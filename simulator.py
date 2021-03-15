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
#=====================================================================================================
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
        pygame.init()
        self.screen=pygame.display.set_mode((Width_of_Screen,Length_of_Screen))
        pygame.display.set_caption("IDM" )
        self.clock= pygame.time.Clock()
        self.load_data()
        self.f = open("trajectory_data.txt", "a")
        """message_font=pygame.font.SysFont("None",20)
        self.NoLeftShift_message_text=message_font.render("Already in Left lane",0,(255,255,255))
        self.NoRightShift_message_text=message_font.render("Already in Right lane",0,(255,255,255))
        self.MaxSpeed_message_text=message_font.render("Attained maximum speed",0,(255,255,255))"""
        
    def load_data(self):
        """Sets map object from file.

        This fetches the textfile containing the background map image and set it as `map' variable of the `Game' class.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        game_folder=path.dirname(__file__)
        self.map=Map(path.join(game_folder,'roadmap.txt'))
        
    def create_background_sprites_from_image(self):
        """Creates background sprites.

        This functions creates sprites which are used as background image. These images of the sprites are formed
        by splitting 'road_image' into tiles of  dimension `backgroundtilesize' X `backgroundtilesize'.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        for row in range(0,road_image_rect.width,backgroundtilesize):
            for col in range (0,road_image_rect.height,backgroundtilesize):
                Background_Tile(self,row,col)              
        
        
    """def set_screen_background(self):
        for row,tiles in enumerate(self.map.data):
            for col,tile  in enumerate(tiles):
                if tile=='G':
                    #self.screen.blit(grasstexture_image,(col*background_tilesize,row*background_tilesize))
                    Grass()
                if tile=='R':
                    self.screen.blit(roadtexture_image,(col*background_tilesize,row*background_tilesize))
                if tile=='Y':
                    self.screen.blit(yellowpainttexture_image,(col*background_tilesize,row*background_tilesize))
       """     
    def new(self):
        """Sets up variables for a new game.

        Variables/lists/objects for a new game is created/initialized.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        self.all_vehicles=pygame.sprite.Group()
        self.other_vehicles=pygame.sprite.Group()
        self.waiting_vehicles_left=pygame.sprite.Group()
        self.waiting_vehicles_right=pygame.sprite.Group()
        self.background_tiles=pygame.sprite.Group()
        self.create_background_sprites_from_image()
        self.seconds_elapsed_checkpoint=0
        self.vehicle_created_counter=0
        self.player_entered=0
        self.camera=Camera(self.map.width, self.map.height)
        
        
    def Warm_up(self):
        """Initial run to populate vehicles on road.

        During the period `Warm_up_time', vehicle sprites other than player are generated as per input flow from the
        entry point of the road, and they are updated according to their update policy (IDM).

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # adding a vehicle at entry position each on left and right lanes
        self.all_vehicles.add(OtherVeh(road_image_rect.height-10,1,self))
        self.all_vehicles.add(OtherVeh(road_image_rect.height-10,2,self))

        for wt in range(Warm_up_time*FPS):
            self.check_for_quit()
            self.generate_incoming_vehicles()
            self.insert_veh_leftlane()
            self.insert_veh_rightlane()            
            #self.all_vehicles.update(self)
            self.other_vehicles.update(self)
            self.draw()  # rendering done for checking purpose, though not really required in warmup
            self.clock.tick(FPS)

    def main_game(self,num_steps_for_each_trajectory,trajectory_number):
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
        #display start of game
        self.player_car=PlayerCar(entry_point_of_player,random.randint(1,2))
        #message_font=pygame.font.SysFont("None",50)
        #self.screen.blit(message_font.render("START GAME",0,(0,0,255)),(200,50)) 
        #pygame.display.flip()
        for step_no in range(num_steps_for_each_trajectory+1):
            self.check_for_quit()
            self.insert_player()            
            self.generate_incoming_vehicles()
            self.insert_veh_leftlane()
            self.insert_veh_rightlane()            
            #self.all_vehicles.update(self)
            self.other_vehicles.update(self)
            feature_list=self.player_car.update(self)
            if self.player_entered:
                row_data=[trajectory_number,step_no]
                for f in feature_list:
                    row_data.append(f) 
                #Data.append(row_data)
                self.f.write(str(row_data))
            self.camera.update(self.player_car)
            collided_list=pygame.sprite.spritecollide(self.player_car,self.other_vehicles,False)
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
        time=pygame.time.get_ticks()/1000
        self.vehicle_created_counter=self.vehicle_created_counter + ((time-self.seconds_elapsed_checkpoint)*FLOW)
        self.seconds_elapsed_checkpoint=time
        if self.vehicle_created_counter >= 1:
            self.vehicle_created_counter=self.vehicle_created_counter-1 #will work only till 1veh/sec
            if random.randint(1,2)==1:
                self.waiting_vehicles_left.add(OtherVeh(road_image_rect.height,1,self))            
            else:
                self.waiting_vehicles_right.add(OtherVeh(road_image_rect.height,2,self))
     
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

        #if self.waiting_vehicles_left:
        if False:
            sprite = random.choice(self.waiting_vehicles_left.sprites())
            # TODO: avoid finding neighbouring fully, just find front veh
            vehicle_in_front_same_lane,vehicle_in_back_same_lane,vehicle_in_front_other_lane,vehicle_in_back_other_lane=neighbouring_vehicles(sprite,self.all_vehicles)
            if vehicle_in_front_same_lane==None:
                sprite.rect.y=road_image_rect.height-sprite.length
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
        #if self.waiting_vehicles_right:
        if False:
            sprite = random.choice(self.waiting_vehicles_right.sprites())
            vehicle_in_front_same_lane,vehicle_in_back_same_lane,vehicle_in_front_other_lane,vehicle_in_back_other_lane=neighbouring_vehicles(sprite,self.all_vehicles)
            if vehicle_in_front_same_lane==None:
                sprite.rect.y=road_image_rect.height-sprite.length
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
            self.player_car.rect.y=entry_point_of_player            
            self.player_entered=1
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
        #self.set_screen_background()
        #self.screen.fill((0,0,0))
        for sprite in self.background_tiles:
            self.screen.blit(sprite.image,self.camera.apply(sprite))
        for sprite in self.all_vehicles:
            self.screen.blit(sprite.image,self.camera.apply(sprite))
        #pygame.display.flip() 
        pygame.display.update()
        
    
        
"""        
game=Game()
game.new()
game.Warm_up()  
print("Warm up Over")      
game.main_game()  """     
        
        
        
        
        
        
        
        
        
