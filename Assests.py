from Constantsv2 import *


def clean_image(mycar_image,road_image):#try out the transparent option
    """Cleans images
    
    If the moving sprite, for ex: vehicle image, has a border that is 
    not transparent to the background, this function changes the colour
    of border to that of background.
    
    Args:
        mycar_image: Moving sprite ( like vehicle) to be cleaned
        road_image: Stationary background image of road 
    
    Returns:
        Image of moving sprite whose border colours are adjusted to
        the colour of background image. 
    
    Raises:
        None        
    """
    road_color = road_image.get_at((round(road_image.get_width()/2.5), round(road_image.get_height()/2)))  # extract colour of background
    threshold_white = 220  # checking for border colours close to WHITE=(255,255,255)
    threshold_black = 20  # checking for black border colours close to BLACK=(0,0,0)
    for x in range(mycar_image.get_width()):
        for y in range(mycar_image.get_height()):  # scanning through the all pixels of my_car.image 
            color = mycar_image.get_at((x, y))
            if color.r > threshold_white and color.g > threshold_white and color.b > threshold_white:
                mycar_image.set_at((x, y), (road_color.r, road_color.g, road_color.b, 0)) 


"""The assests are imported here """
enemycar_image1=pygame.image.load("enemycar1.jfif")
enemycar_image1=pygame.transform.scale(enemycar_image1,(width_of_car,length_of_car))
enemycar_image2=pygame.image.load("enemycar2.png")
enemycar_image2=pygame.transform.scale(enemycar_image2,(width_of_car,length_of_car))
mycar_image=pygame.image.load("mycar.png")
mycar_image=pygame.transform.scale(mycar_image,(width_of_car,length_of_car))
enemytruck_image=pygame.image.load("truck.jfif")
enemytruck_image=pygame.transform.scale(enemytruck_image,(width_of_truck,length_of_truck))
grasstexture_image=pygame.image.load("grass.jpg")
#grasstexture_image=pygame.transform.scale(grasstexture_image,(background_tilesize,background_tilesize))
roadtexture_image=pygame.image.load("roadtexture.jpg")
#roadtexture_image=pygame.transform.scale(roadtexture_image,(background_tilesize,background_tilesize))
yellowpainttexture_image=pygame.image.load("yellowpainttexture.jpg")
#yellowpainttexture_image=pygame.transform.scale(yellowpainttexture_image,(background_tilesize,background_tilesize))


"""The assests are cleaned here """
clean_image(enemycar_image1,roadtexture_image)
clean_image(enemycar_image2,roadtexture_image)
clean_image(mycar_image,roadtexture_image)
clean_image(enemytruck_image,roadtexture_image)