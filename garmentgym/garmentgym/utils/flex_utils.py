import colorsys
from copy import deepcopy
import random
import pyflex
import numpy as np
def center_object():
    pos = pyflex.get_positions().reshape(-1, 4)
    mid_x = (np.max(pos[:, 0]) + np.min(pos[:, 0]))/2
    mid_y = (np.max(pos[:, 2]) + np.min(pos[:, 2]))/2
    pos[:, [0, 2]] -= np.array([mid_x, mid_y])
    pyflex.set_positions(pos.flatten())
    pyflex.step()
    pyflex.render()

def set_random_cloth_color():
    hsv_color = [
        random.uniform(0.0, 1.0),
        random.uniform(0.0, 1.0),
        random.uniform(0.6, 1.0)
    ]
    rgb_color = colorsys.hsv_to_rgb(*hsv_color)
    pyflex.change_cloth_color(rgb_color)



def set_state(state_dict):
    pyflex.set_positions(state_dict['particle_pos'])
    pyflex.set_velocities(state_dict['particle_vel'])
    pyflex.set_shape_states(state_dict['shape_pos'])
    pyflex.set_phases(state_dict['phase'])

def wait_until_stable(max_steps=300,
                      tolerance=1e-2,
                      gui=False,
                      step_sim_fn=lambda: pyflex.step()):
    for _ in range(max_steps):
        particle_velocity = pyflex.get_velocities()
        if np.abs(particle_velocity).max() < tolerance:
            return True
        step_sim_fn()
        pyflex.render()
    return False


