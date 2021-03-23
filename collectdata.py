"""Collects data from human driver.

This acts like a main() file to collect data from test subject
from multiple trajectories. It creates instance of Game object for
each trajectory. The data collected is written to `trajectory_data.txt'
in Game instance method 'main_game()'
"""

num_of_trajectories_required = 2  # specify the required number of episode data to be collected
num_steps_for_each_trajectory = 500  # specify the required length of episode in terms of time step in seconds

from simulator import *

for trajectory_number in range(1, num_of_trajectories_required + 1):
    pygame.quit()  # closes the existing pygame window (if any opened)
    game = Game()  # creates an instance of Game object to run the simulator
    game.new()
    game.Warm_up()
    print("Warm up Over")
    game.main_game(num_steps_for_each_trajectory, trajectory_number)
