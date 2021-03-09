
from Assests import *
#=====================================================================================================
class OtherVeh(pygame.sprite.Sprite):    
    def __init__(self,position,lane,game):
        pygame.sprite.Sprite.__init__(self,game.other_vehicles)
        # LIST OUT ALL THE CLASS VARIABLES 
        #init() creates either a truck or a car depending upon 'proportion_of_trucks'
        if rand() < proportion_of_trucks:
            self.image=enemytruck_image
            self.a,self.b,self.s0,self.T,self.v0=random.gauss(a_TRUCK,sigma_a_TRUCK),random.gauss(b_TRUCK,sigma_b_TRUCK),random.gauss(s0_TRUCK,sigma_s0_TRUCK),random.gauss(T_TRUCK,sigma_T_TRUCK),random.gauss(v0_TRUCK,sigma_v0_TRUCK)
            self.length=length_of_truck
            self.width=width_of_truck
            self.type=2
        else:
            # 2 different images are used for IDM cars
            image_number=random.randint(1,2)
            if  image_number == 1:
                self.image=enemycar_image1
            else:
                self.image=enemycar_image2
            self.a,self.b,self.s0,self.T,self.v0=random.gauss(a_CAR,sigma_a_CAR),random.gauss(b_CAR,sigma_b_CAR),random.gauss(s0_CAR,sigma_s0_CAR),random.gauss(T_CAR,sigma_T_CAR),random.gauss(v0_CAR,sigma_v0_CAR)
            self.length=length_of_car
            self.width=width_of_car
            self.type=1
        self.rect=self.image.get_rect()
        self.lane=lane
        if lane==1:        
            self.rect.x=left_lane_mid-(self.width/2)
        else:
            self.rect.x=right_lane_mid-(self.width/2)
        self.rect.y=position
        self.v, self.acc,self.s,self.delta_v=0,0,0,0
    
    
    def update(self,game):
        print(game.player_entered)
        self.vehicle_in_front_same_lane,self.vehicle_in_back_same_lane,self.vehicle_in_front_other_lane,self.vehicle_in_back_other_lane=neighbouring_vehicles(self,game.all_vehicles)
        if self.vehicle_in_front_same_lane==None:
            self.delta_v=-10000000000            
            self.s=10000000000
        else:            
            self.delta_v=self.v-self.vehicle_in_front_same_lane.v
            self.s=(self.rect.top-self.vehicle_in_front_same_lane.rect.top)/pixel_conversion
               
        s_star=self.s0+ max(0,(self.v*self.T)+(self.v*self.delta_v/(2*numpy.power(self.a*self.b,0.5))))
        self.acc=self.a*(1-numpy.power(self.v/self.v0,delta)-numpy.power(s_star/self.s,2))
        
        #Expected gap
        if self.vehicle_in_front_same_lane==None:
            expected_gap=100000000000
        else:
            expected_gap=self.s + (self.vehicle_in_front_same_lane.v*delta_t) +(0.5*self.vehicle_in_front_same_lane.acc*delta_t*delta_t)
        
        acc_upper_cutoff=min(1.5*self.a,2*(expected_gap+self.s0-(self.v*delta_t))/(delta_t*delta_t))
        if self.acc > acc_upper_cutoff:
            self.acc=acc_upper_cutoff
        acc_lower_cutoff=max(-1*self.b,-1*self.v/delta_t)
        if self.acc < acc_lower_cutoff:
            self.acc=acc_lower_cutoff
            
        distance_moved=(self.v*delta_t)+(0.5*self.acc*delta_t*delta_t)        
        self.rect.y=self.rect.y-(distance_moved*pixel_conversion)
        
        self.v=self.v+(self.acc*delta_t)
        """if self.v < 0:
            print("negetive v")
            self.v=0"""
#========================================================================================================================
class PlayerCar(pygame.sprite.Sprite):
    def __init__(self,position,lane):
        pygame.sprite.Sprite.__init__(self)
        self.image=mycar_image
        self.rect=self.image.get_rect()
        self.lane=lane
        self.length=length_of_car
        self.width=width_of_car
        if self.lane==1:        
            self.rect.x=left_lane_mid-(self.width/2)
        else:
            self.rect.x=right_lane_mid-(self.width/2)
        self.rect.y=position  #note that we need to start somewhere midway, so start other vehicles much before        
        self.v, self.acc=player_start_velocity,0
        self.max_velocity=player_max_velocity
        
    def update(self,game):
        x_change=0
        y_change=0
        
        self.vehicle_in_front_same_lane,self.vehicle_in_back_same_lane,self.vehicle_in_front_other_lane,self.vehicle_in_back_other_lane=neighbouring_vehicles(self,game.all_vehicles)
        feature_list=[self.v,self.rect.y,1,2]
        
        
        #Apply decceleration due to friction
        self.v=max(0,self.v-(player_friction_decc*(1/FPS)))
         
        print(self.rect.top)
        action=0
        keys=pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            if self.lane==2:
                x_change=-(right_lane_mid-left_lane_mid)
                self.lane=1
                action=1
            """else:
                game.screen.blit(game.NoLeftShift_message_text,(50,50)) 
                pygame.display.flip()""" 
        if keys[pygame.K_RIGHT]:
            if self.lane==1:
                x_change=(right_lane_mid-left_lane_mid)
                self.lane=2
                action=2
            """else:
                game.screen.blit(game.NoRightShift_message_text,(50,50))
                pygame.display.flip()""" 
        if keys[pygame.K_UP]:            
            self.v=min(player_max_velocity,self.v+(player_acceleration_step*(1/FPS)))
            action=3
        if keys[pygame.K_DOWN]:
            self.v=max(0,self.v-(player_decceleration_step*(1/FPS)))
            action=4
        y_change=-(self.v/FPS)-(0.5*player_acceleration_step/(FPS*FPS))    
            
        self.rect.x = self.rect.x + x_change
        self.rect.y+=(y_change*pixel_conversion)
        
        feature_list.append(action)
        return feature_list    
#==========================================================================================================================            
            
def neighbouring_vehicles(vehicle,all_vehicles):
    same_lane_vehicles=[sprite for sprite in all_vehicles if sprite.lane==vehicle.lane]
    same_lane_vehicles_in_front=[sprite for sprite in same_lane_vehicles if vehicle.rect.top-sprite.rect.top>0]
    same_lane_vehicles_in_back=[sprite for sprite in same_lane_vehicles if vehicle.rect.top-sprite.rect.top<0]
    if len(same_lane_vehicles_in_front) > 0:
        vehicle_in_front_same_lane=min([sprite for sprite in same_lane_vehicles_in_front], key=lambda sprite: vehicle.rect.top-sprite.rect.top)
    else:
        vehicle_in_front_same_lane=None
    if len(same_lane_vehicles_in_back) > 0:
        vehicle_in_back_same_lane=min([sprite for sprite in same_lane_vehicles_in_back], key=lambda sprite: sprite.rect.top-vehicle.rect.top)
    else:
        vehicle_in_back_same_lane=None
    other_lane_vehicles=[sprite for sprite in all_vehicles if not sprite.lane==vehicle.lane]
    other_lane_vehicles_in_front=[sprite for sprite in other_lane_vehicles if vehicle.rect.top-sprite.rect.top>0]
    other_lane_vehicles_in_back=[sprite for sprite in other_lane_vehicles if vehicle.rect.top-sprite.rect.top<0]            
    if len(other_lane_vehicles_in_front) > 0:
        vehicle_in_front_other_lane=min([sprite for sprite in other_lane_vehicles_in_front], key=lambda sprite: vehicle.rect.top-sprite.rect.top)
    else:
        vehicle_in_front_other_lane=None
    if len(other_lane_vehicles_in_back) > 0:    
        vehicle_in_back_other_lane=min([sprite for sprite in other_lane_vehicles_in_back], key=lambda sprite: sprite.rect.top-vehicle.rect.top)
    else:
        vehicle_in_back_other_lane=None
    return vehicle_in_front_same_lane,vehicle_in_back_same_lane,vehicle_in_front_other_lane,vehicle_in_back_other_lane
#=========================================================================================================================================
class Camera:
    def __init__(self,width,height):
        self.camera=pygame.Rect(0,0,width,height)
        self.width=width
        self.height=height
        
    def apply(self,entity):
        return entity.rect.move(self.camera.topleft)
    
    def update(self,target):
        #x = -target.rect.x + int(Width_of_Screen/2)
        x = 0
        y = -target.rect.y + int(Length_of_Screen/2)
        """x=min(0,x) #left
        y=min(0,y) #top
        x=max(-(self.width-Width_of_Screen),x)
        y=max(-(self.height-Length_of_Screen),y)"""
        self.camera=pygame.Rect(x,y,self.width,self.height)
#================================================================================================
class Map:
    def __init__(self,filename):
        self.data=[]
        with open(filename,'rt') as f:
            for line in f:
                self.data.append(line.strip())                
        
        self.width=len(self.data[0])*background_tilesize
        self.height=len(self.data)*background_tilesize
#=======================================================================================================
class Background_Tile(pygame.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.background_tiles
        pygame.sprite.Sprite.__init__(self, self.groups)  #adds this sprite to grass_tiles group
        self.game = game  #not required for now I think
        multiple= int(y/road_image.get_rect().height)
        y= y- (multiple*road_image.get_rect().height)
        self.image = road_image.subsurface((x,y,backgroundtilesize,backgroundtilesize))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y