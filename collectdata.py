#to collect data

num_of_trajectories_required=2
num_steps_for_each_trajectory=500

from simulator import *

#Data=[]        
for trajectory_number in range(1,num_of_trajectories_required+1):
    pygame.quit()
    step_no=1 
    game=Game()
    game.new()
    game.Warm_up()  
    print("Warm up Over")      
    #game.main_game(num_steps_for_each_trajectory,trajectory_number,Data)
    game.main_game(num_steps_for_each_trajectory,trajectory_number)
#numpy.savetxt('data.csv', Data , delimiter=',')  

