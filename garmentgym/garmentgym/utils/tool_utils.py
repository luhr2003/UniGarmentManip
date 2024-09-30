import abc
from copy import deepcopy
import numpy as np
import scipy
import pyflex


class ActionToolBase(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self, state):
        """ Reset """

    @abc.abstractmethod
    def step(self, action):
        """ 
        Step funciton to change the action space states.
        Does not call pyflex.step()
        """


class Picker(ActionToolBase):
    def __init__(self, num_picker=1,
                 picker_radius=0.02,
                 init_pos=(0., -0.1, 0.),
                 picker_threshold=0.05,
                 particle_radius=0.05,
                 picker_low=(-0.4, 0., -0.4),
                 picker_high=(0.4, 0.5, 0.4),
                 init_particle_pos=None,
                 spring_coef=1.2, picker_size=0.05,**kwargs):
        """
        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """

        super(Picker).__init__()
        self.picker_radius = picker_radius
        self.picker_threshold = picker_threshold
        self.num_picker = num_picker
        self.picker_size=picker_size
        self.picked_particles = [None] * self.num_picker
        self.picker_low, self.picker_high = np.array(
            list(picker_low)), np.array(list(picker_high))
        self.init_pos = init_pos
        self.particle_radius = particle_radius
        self.init_particle_pos = init_particle_pos
        # Prevent picker to drag two particles too far away
        self.spring_coef = spring_coef

    def _get_centered_picker_pos(self, center):
        r = np.sqrt(self.num_picker - 1) * self.picker_radius * 2.
        pos = []
        for i in range(self.num_picker):
            x = center[0] + np.sin(2 * np.pi * i / self.num_picker) * r
            y = center[1]
            z = center[2] + np.cos(2 * np.pi * i / self.num_picker) * r
            pos.append([x, y, z])
        return np.array(pos)

    def reset(self, center):
        for i in (0, 2):
            offset = center[i] - (self.picker_high[i] +
                                  self.picker_low[i]) / 2.
            self.picker_low[i] += offset
            self.picker_high[i] += offset
        init_picker_poses = self._get_centered_picker_pos(center)

        for picker_pos in init_picker_poses:
            pyflex.add_sphere(self.picker_size, picker_pos, [1, 0, 0, 0])
        # Need to call this to update the shape collision
        pos = pyflex.get_shape_states()
        pyflex.set_shape_states(pos)

        self.picked_particles = [None] * self.num_picker
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        centered_picker_pos = self._get_centered_picker_pos(center)
        for (i, centered_picker_pos) in enumerate(centered_picker_pos):
            shape_state[i] = np.hstack(
                [centered_picker_pos,
                 centered_picker_pos,
                 [1, 0, 0, 0],
                 [1, 0, 0, 0]])
        pyflex.set_shape_states(shape_state)
        # Remove this as having an additional step here
        # may affect the cloth drop env
        self.particle_inv_mass = \
            pyflex.get_positions().reshape(-1, 4)[:, 3]

    def add_pickers(self, picker_poses):
        for picker_pos in picker_poses:
            pyflex.add_sphere(self.picker_radius, picker_pos, [1, 0, 0, 0])

        # Need to call this to update the shape collision
        pos = pyflex.get_shape_states()
        pyflex.set_shape_states(pos)
            
    def remove_pickers(self):
        picker_pos, raw_particle_pos = self._get_pos()
        new_particle_pos = raw_particle_pos.copy()

        # Un-pick the particles
        for i in range(self.num_picker):
            if self.picked_particles[i] is not None:
                # Revert the mass
                new_particle_pos[self.picked_particles[i], 3] = self.particle_inv_mass[self.picked_particles[i]]
                self.picked_particles[i] = None
        self._set_pos(picker_pos, new_particle_pos)
        pyflex.pop_shape(self.num_picker)
        pos = pyflex.get_shape_states()
        pyflex.set_shape_states(pos)

    def _get_picker_pos(self):
        """ 
        Get the current pos of the pickers
        """
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        return picker_pos[:self.num_picker, :3]


    def _get_pos(self):
        """ 
        Get the current pos of the pickers and the particles,
         along with the inverse mass of each particle
        """
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        tmp_picker_pos = deepcopy(picker_pos[:self.num_picker, :3])
        
        return picker_pos[:self.num_picker, :3], particle_pos

    def _set_pos(self,picker_pos, particle_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:self.num_picker, 3:6] = shape_states[:self.num_picker, :3]
        shape_states[:self.num_picker, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)
        pyflex.step()
        pyflex.render()

    def step(self, action):
        """ action = [translation, pick/unpick] * num_pickers.
        1. Determine whether to pick/unpick the particle and which one,
           for each picker
        2. Update picker pos
        3. Update picked particle pos
        """
        action = np.reshape(action, [-1, 4])
        pick_flag = action[:, 3] > 0.5
        picker_pos, particle_pos = self._get_pos()
        new_particle_pos = particle_pos.copy()
        new_picker_pos = picker_pos.copy()

        # Un-pick the particles
        for i in range(self.num_picker):
            if not pick_flag[i] and self.picked_particles[i] is not None:
                # Revert the mass
                new_particle_pos[self.picked_particles[i], 3] = \
                    self.particle_inv_mass[self.picked_particles[i]]
                self.picked_particles[i] = None

        # Pick new particles and update the mass and the positions
        for i in range(self.num_picker):
            new_picker_pos[i, :] = picker_pos[i, :] + action[i, :3]
            if pick_flag[i]:
                # No particle is currently picked and
                # thus need to select a particle to pick
                if self.picked_particles[i] is None:
                    dists = scipy.spatial.distance.cdist(picker_pos[i].reshape(
                        (-1, 3)), particle_pos[:, :3].reshape((-1, 3)))
                    idx_dists = np.hstack(
                        [np.arange(particle_pos.shape[0]).reshape((-1, 1)),
                         dists.reshape((-1, 1))])
                    mask = dists.flatten() <= self.picker_threshold + \
                        self.picker_radius + self.particle_radius
                    idx_dists = idx_dists[mask, :].reshape((-1, 2))
                    if idx_dists.shape[0] > 0:
                        pick_id, pick_dist = None, None
                        for j in range(idx_dists.shape[0]):
                            if idx_dists[j, 0] not in self.picked_particles\
                                    and (pick_id is None or
                                         idx_dists[j, 1] < pick_dist):
                                pick_id = idx_dists[j, 0]
                                pick_dist = idx_dists[j, 1]
                        if pick_id is not None:
                            self.picked_particles[i] = int(pick_id)

                if self.picked_particles[i] is not None:
                    new_particle_pos[self.picked_particles[i], :3] =\
                        particle_pos[self.picked_particles[i], :3]\
                        + new_picker_pos[i, :] - picker_pos[i, :]
                    # Set the mass to infinity
                    new_particle_pos[self.picked_particles[i], 3] = 0

        # check for e.g., rope, the picker is not dragging the particles
        # too far away that violates the actual physicals constraints.
        if self.init_particle_pos is not None:
            picked_particle_idices = []
            active_picker_indices = []
            for i in range(self.num_picker):
                if self.picked_particles[i] is not None:
                    picked_particle_idices.append(self.picked_particles[i])
                    active_picker_indices.append(i)

            for i in range(len(picked_particle_idices)):
                for j in range(i + 1, len(picked_particle_idices)):
                    init_distance = np.linalg.norm(
                        self.init_particle_pos[picked_particle_idices[i], :3] -
                        self.init_particle_pos[picked_particle_idices[j], :3])
                    now_distance = np.linalg.norm(
                        new_particle_pos[picked_particle_idices[i], :3] -
                        new_particle_pos[picked_particle_idices[j], :3])
                    # if dragged too long, make the action has no effect;
                    # revert it
                    if now_distance >= init_distance * self.spring_coef:
                        new_picker_pos[active_picker_indices[i], :] = \
                            picker_pos[active_picker_indices[i], :].copy()
                        new_picker_pos[active_picker_indices[j], :] = \
                            picker_pos[active_picker_indices[j], :].copy()
                        new_particle_pos[picked_particle_idices[i], :3] =\
                            particle_pos[picked_particle_idices[i], :3].copy()
                        new_particle_pos[picked_particle_idices[j], :3] =\
                            particle_pos[picked_particle_idices[j], :3].copy()

        self._set_pos(new_picker_pos, new_particle_pos)

class PickerPickPlace(Picker):
    def __init__(self, num_picker, steps_limit=1, **kwargs):
        super().__init__(num_picker=num_picker, **kwargs)
        self.delta_move = 1.0
        self.steps_limit = steps_limit
        self.num_picker=num_picker

    def _set_picker_pos(self,picker_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:self.num_picker, 3:6] = shape_states[:, :3]
        shape_states[:self.num_picker, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.step()
        pyflex.render()




    def step(self, action, step_sim_fn=lambda: pyflex.step()):
        """
        action: Array of pick_num x 4. For each picker,
         the action should be [x, y, z, pick/drop]. 
        The picker will then first pick/drop, and keep
         the pick/drop state while moving towards x, y, x.
        """
        total_steps = 0
        action = action.reshape(-1, 4)
        curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:self.num_picker, :3]
        end_pos = np.vstack([picker_pos
                             for picker_pos in action[:, :3]])
        dist = np.linalg.norm(curr_pos - end_pos, axis=1)
        num_step = np.max(np.ceil(dist / self.delta_move))
        if num_step < 0.1:
            return
        delta = (end_pos - curr_pos) / num_step
        norm_delta = np.linalg.norm(delta)
        for i in range(int(min(num_step, self.steps_limit))):
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:self.num_picker, :3]
            dist = np.linalg.norm(end_pos - curr_pos, axis=1)
            if np.alltrue(dist < norm_delta):
                delta = end_pos - curr_pos
            super().step(np.hstack([delta, action[:, 3].reshape(-1, 1)]))
            step_sim_fn()
            total_steps += 1
            if np.alltrue(dist < self.delta_move):
                break
            pyflex.render()
        return total_steps
    def shape_move(self,final_points,speed=0.1):
        final_points=final_points.reshape(-1,3)
        init_shape_pos=self._get_picker_pos()
        for i in range(int(1/speed)):
            next_shape_pos=i*(final_points-init_shape_pos)*speed+init_shape_pos
            self._set_picker_pos(next_shape_pos)
        self._set_picker_pos(final_points)
    

class Pickerpoint(ActionToolBase):
    def __init__(self, num_picker=1,
                 picker_radius=0.02,
                 init_pos=(0., -0.1, 0.),
                 picker_threshold=0.05,
                 particle_radius=0.05,
                 picker_low=(-0.4, 0., -0.4),
                 picker_high=(0.4, 0.5, 0.4),
                 init_particle_pos=None,
                 spring_coef=1.2, picker_size=0.05,**kwargs):
        """
        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """

        super(Picker).__init__()
        self.picker_radius = picker_radius
        self.picker_threshold = picker_threshold
        self.num_picker = num_picker
        self.picker_size=picker_size
        self.picked_particles = [None] * self.num_picker
        self.picker_low, self.picker_high = np.array(
            list(picker_low)), np.array(list(picker_high))
        self.init_pos = init_pos
        self.particle_radius = particle_radius
        self.init_particle_pos = init_particle_pos
        # Prevent picker to drag two particles too far away
        self.spring_coef = spring_coef

    def _get_centered_picker_pos(self, center):
        r = np.sqrt(self.num_picker - 1) * self.picker_radius * 2.
        pos = []
        for i in range(self.num_picker):
            x = center[0] + np.sin(2 * np.pi * i / self.num_picker) * r
            y = center[1]
            z = center[2] + np.cos(2 * np.pi * i / self.num_picker) * r
            pos.append([x, y, z])
        return np.array(pos)

    def reset(self, center):
        for i in (0, 2):
            offset = center[i] - (self.picker_high[i] +
                                  self.picker_low[i]) / 2.
            self.picker_low[i] += offset
            self.picker_high[i] += offset
        init_picker_poses = self._get_centered_picker_pos(center)

        for picker_pos in init_picker_poses:
            pyflex.add_sphere(self.picker_size, picker_pos, [1, 0, 0, 0])
        # Need to call this to update the shape collision
        pos = pyflex.get_shape_states()
        pyflex.set_shape_states(pos)

        self.picked_particles = [None] * self.num_picker
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        centered_picker_pos = self._get_centered_picker_pos(center)
        for (i, centered_picker_pos) in enumerate(centered_picker_pos):
            shape_state[i] = np.hstack(
                [centered_picker_pos,
                 centered_picker_pos,
                 [1, 0, 0, 0],
                 [1, 0, 0, 0]])
        pyflex.set_shape_states(shape_state)
        # Remove this as having an additional step here
        # may affect the cloth drop env
        self.particle_inv_mass = \
            pyflex.get_positions().reshape(-1, 4)[:, 3]

    def add_pickers(self, picker_poses):
        for picker_pos in picker_poses:
            pyflex.add_sphere(self.picker_radius, picker_pos, [1, 0, 0, 0])

        # Need to call this to update the shape collision
        pos = pyflex.get_shape_states()
        pyflex.set_shape_states(pos)
            
    def remove_pickers(self):
        picker_pos, raw_particle_pos = self._get_pos()
        new_particle_pos = raw_particle_pos.copy()

        # Un-pick the particles
        for i in range(self.num_picker):
            if self.picked_particles[i] is not None:
                # Revert the mass
                new_particle_pos[self.picked_particles[i], 3] = self.particle_inv_mass[self.picked_particles[i]]
                self.picked_particles[i] = None
        self._set_pos(picker_pos, new_particle_pos)
        pyflex.pop_shape(self.num_picker)
        pos = pyflex.get_shape_states()
        pyflex.set_shape_states(pos)


    @staticmethod
    def _get_picker_pos():
        """ 
        Get the current pos of the pickers
        """
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        return picker_pos[:, :3]


    @staticmethod
    def _get_pos():
        """ 
        Get the current pos of the pickers and the particles,
         along with the inverse mass of each particle
        """
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        tmp_picker_pos = deepcopy(picker_pos[:, :3])
        
        return picker_pos[:, :3], particle_pos

    @staticmethod
    def _set_pos(picker_pos, particle_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)
        pyflex.step()
        pyflex.render()


    def hide_pickers(self):
        curr_pos_shape=self._get_picker_pos()
        curr_pos_shape[:,1]=0
        self._set_picker_pos(curr_pos_shape)


    @staticmethod
    def _set_picker_pos(picker_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.step()
        pyflex.render()


    def shape_move(self,final_points,speed=0.1):
        final_points=final_points.reshape(-1,3)
        init_shape_pos=self._get_picker_pos()
        for i in range(int(1/speed)):
            next_shape_pos=i*(final_points-init_shape_pos)*speed+init_shape_pos
            self._set_picker_pos(next_shape_pos)
        self._set_picker_pos(final_points)


    def move(self,pickpoint, final_place, speed=0.01, cond=lambda : True, step_fn=None,grasp=True):
        curr_pos=pyflex.get_positions().reshape(-1,4)
        mass=curr_pos[pickpoint][3]
        init_point = curr_pos[pickpoint][:3].copy()
        init_shape_point=deepcopy(init_point)
        init_shape_point[1]+=self.picker_size*2
        # self.shape_move(init_shape_point)
        if grasp:
            init_point = curr_pos[pickpoint][:3].copy()
            for j in range(int(1/speed)):
                if not cond():
                    break
                curr_pos = pyflex.get_positions().reshape(-1,4)
                curr_vel = pyflex.get_velocities().reshape(-1,4)
                pickpoint_pos = (final_place-init_point)*(j*speed) + init_point
                curr_pos[pickpoint][:3] = pickpoint_pos
                curr_pos[pickpoint][3] = 0
                curr_vel[pickpoint][:3] = [0, 0, 0]

                pyflex.set_positions(curr_pos)
                pyflex.set_velocities(curr_vel)
                if step_fn is not None:
                    step_fn()
                curr_pos = pyflex.get_positions().reshape(-1,4)
                curr_pos[pickpoint][3]=mass
                pyflex.set_positions(curr_pos)
                if step_fn is not None:
                    step_fn()
                # next_shape_pos=deepcopy(pickpoint_pos)
                # next_shape_pos[1]+=self.picker_size*2
                # self._set_picker_pos(pickpoint_pos)

            curr_pos = pyflex.get_positions().reshape(-1,4)
            curr_vel = pyflex.get_velocities().reshape(-1,4)
            curr_pos[pickpoint][:3]=final_place
            curr_pos[pickpoint][3]=0
            curr_vel[pickpoint][:3] = [0, 0, 0]
            pyflex.set_positions(curr_pos)
            pyflex.set_velocities(curr_vel)
            if step_fn is not None:
                step_fn()
            
            curr_pos = pyflex.get_positions().reshape(-1,4)
            curr_pos[pickpoint][3]=mass
            pyflex.set_positions(curr_pos)
            if step_fn is not None:
                step_fn()

        
    def pick_place(self,pickpoint,placepoint,speed=0.05,height=0.1,step_fn=pyflex.step):
        curr_pos=pyflex.get_positions().reshape(-1,4)
        init_point = curr_pos[pickpoint][:3].copy()
        init_point[1]=height
        self.move(pickpoint,init_point,step_fn=step_fn)
        print("2")
        placepointup=deepcopy(placepoint)
        placepointup[1]=height
        self.move(pickpoint,placepointup,step_fn=step_fn)
        print("3")
        self.move(pickpoint,placepoint,step_fn=step_fn)
        print("4")
    

    def step():
        raise NotImplementedError