import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Pendulum():
    def __init__(self, length: float, x0, linear: bool, dt: float, steps: int):
        """
        simulates the dynamics of a mass hanging on a rope
        :param length: rope length
        :param x0: initial position and velocity of mass 2x1
        :param linear: linear model or nonlinear model
        :param dt: timestep
        :param steps: number of timesteps
        """
        super().__init__()
        ''' logging and simulation settings '''
        self.x_log = np.zeros((steps, 2))
        self.u_log = np.zeros(steps)
        self.x_log[0] = x0.flatten()
        self.dt = dt
        self.log_idx = 0
        self.steps = steps

        ''' system definition '''
        g = 9.81
        self.A = np.array([[0, 1], [-g / length, 0]])
        self.b = np.array([[0], [g / length]])

        self.c = np.array([[1, 0]])
        self.l = length
        self.linear = linear

        '''initial state'''
        self.x = x0
        self.y = self.c @ self.x

    def step(self, cntrl_u: float):
        """
        increment the system state
        :param cntrl_u: control input u
        :return: state
        """
        self.x = self.x + np.array([[0.1],[0.1]])*np.sin(self.log_idx*0.1)
        # TODO: update the state according to linear system dynamics dot x = Ax + bu
        # the above is just some random update so you get a video and plots

        if not self.linear:
            pass  # TODO: Bonus: update the state with nonlinear dynamics

        self.y = self.c @ self.x

        ''' logging '''
        self.log_idx += 1
        self.x_log[self.log_idx] = self.x.flatten()
        self.u_log[self.log_idx] = u

        return self.x

    def plot(self, ax_t=None, ax_x=None, suffix=''):
        """ plot the results """
        # nothing to do here, plots should be fine

        t_range = np.arange(0., self.dt * self.steps, self.dt)

        if ax_t is None:
            fig, ax_t = plt.subplots(1, figsize=(10, 5))
        ax_t.plot(t_range, self.x_log[:, 0], label=r'$x$' + suffix)
        ax_t.plot(t_range, self.x_log[:, 1], label=r'$\dot x$' + suffix)
        if suffix != 'obs':
            ax_t.plot(t_range, self.u_log, label=r'$u$')
        ax_t.legend()
        ax_t.set_xlabel('time in s')
        ax_t.set_ylabel('m or m/s')
        ax_t.grid()

        if ax_x is None:
            fig, ax_x = plt.subplots(1, figsize=(10, 5))
        ax_x.plot(self.x_log[:, 0], self.x_log[:, 1], label=r'${x}$' + suffix)
        ax_x.legend()
        ax_x.set_xlabel('$x$ in m')
        ax_x.set_ylabel(r'$\dot x$ in m/s')
        ax_x.grid()

        return ax_t, ax_x

    def create_video(self):
        """ plot the results """
        # nothing to do here, video should be fine
        t_range = np.arange(0., self.dt * self.steps, self.dt)
        fig, ax = plt.subplots(1, figsize=(10, 5))
        ax.set_xlim([-self.l, self.l])
        mass, = ax.plot(self.x_log[0, 0], -self.l, 'o')
        rope, = ax.plot([0, self.x_log[0, 0]], [0, -self.l])

        def init():
            return [mass, rope]

        def animate(i):
            mass_x = self.x_log[i, 0]
            phi_i = np.arcsin(mass_x / self.l)
            mass_y = -self.l * np.cos(phi_i)
            mass.set_data(mass_x, mass_y)
            rope.set_data([self.u_log[i], mass_x], [0, mass_y])

            return [mass, rope]

        ax.set_aspect('equal', adjustable='box')
        anim = FuncAnimation(fig, animate, len(t_range), init_func=init, interval=self.dt, blit=False)
        anim.save('pendulum.mp4', writer='ffmpeg', fps=1 / self.dt)


def compute_feedback_gain(A, b, eigenvalues_f):
    from controllable_canonical_form import controlability_matrix, characteristic_polynomial

    # TODO: Exercise 2 compute the feedback gain using the controllable canonical form

    return k


if __name__ == '__main__':  # this code is only used when we directly execute this file (not when importing it)

    ''' simulation settings'''
    dt = 0.05
    steps = 200
    t_range = np.arange(dt, dt * steps, dt)

    pendulum = Pendulum(length=3, x0=np.array([[1.], [0.]]), linear=False, dt=dt,
                        steps=steps)  # create an instance of the pendulum

    l = np.array([[-2], [-2]])  # define the observer feedback

    # TODO: Exercise 2 define desired eigenvalues
    # k = compute_feedback_gain(pendulum.A, pendulum.b, eigenvalues_f=[complex(-1.5, 0.1), complex(-1.5, -0.1)])


    ''' simulation '''
    for t in t_range:
        u = 0. # TODO: compute the control feedback using pendulum.x
        pendulum.step(u)

    ''' visualization '''
    ax_t, ax_x = pendulum.plot()
    pendulum.create_video()
    plt.show()
