from os import path
from VehicleClass import *
#=====================================================================================================
class Game:
    def __init__(self):
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
        game_folder=path.dirname(__file__)
        self.map=Map(path.join(game_folder,'roadmap.txt'))
        
    def create_background_sprites_from_image(self):
        for row in range(0,road_image_rect.width,backgroundtilesize):
            for col in range (0,2*road_image_rect.height,backgroundtilesize):
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
        #initialize all variables and do all the setup for a new game
        self.all_vehicles=pygame.sprite.Group()   #creating a group for all moving sprites
        self.other_vehicles=pygame.sprite.Group()
        self.waiting_vehicles_left=pygame.sprite.Group()  # group for vehicles created according to flow, but not yet entered lane
        self.waiting_vehicles_right=pygame.sprite.Group()
        self.background_tiles=pygame.sprite.Group()
        self.create_background_sprites_from_image()
        self.seconds_elapsed_checkpoint=0
        self.vehicle_created_counter=0 #keeps track of cumulative number of vehicles created in each frame
        self.player_entered=0
        self.camera=Camera(self.map.width, self.map.height)
        
        
    def Warm_up(self):
        self.all_vehicles.add(OtherVeh(road_image_rect.height-10,1,self))  #adding a vehicle in both lanes initially
        self.all_vehicles.add(OtherVeh(road_image_rect.height-10,2,self))
        
        #note: add multiple vehicles to warm up fast
        for wt in range(Warm_up_time*FPS):
            self.check_for_quit()
            self.generate_incoming_vehicles()
            self.insert_veh_leftlane()
            self.insert_veh_rightlane()            
            #self.all_vehicles.update(self)
            self.other_vehicles.update(self)
            self.draw()  #not required in warmup
            self.clock.tick(FPS) 
            
    #def main_game(self,num_steps_for_each_trajectory,trajectory_number,Data):
    def main_game(self,num_steps_for_each_trajectory,trajectory_number):
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
    def insert_veh_leftlane(self):
        #waiting vehicles check for space to enter, if there is space these vehicles enter
        #if self.waiting_vehicles_left:
        if False:
            sprite = random.choice(self.waiting_vehicles_left.sprites())
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
        
        
        
        
        
        
        
        
        
