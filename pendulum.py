import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Pendulum():
    def __init__(self, length, x0, linear, dt, steps):
        super().__init__()
        ''' logging and simulation settings '''
        self.x_log = np.zeros((steps, 2))
        self.u_log = np.zeros(steps)
        self.x_log[0] = x0.flatten()
        self.dt = dt
        self.log_idx = 0
        self.steps = steps

        ''' system definition '''
        self.x = x0
        g = 9.81
        self.A = np.array([[0, 1], [-g / length, 0]])
        self.b = np.array([[0], [g / length]])
        self.l = length
        self.linear = linear

    def step(self, cntrl_u):
        """ increment the system state """
        if self.linear:  # the pendulum is modeled by linear dynamics
            dx = self.A @ self.x + self.b * cntrl_u
            self.x += dx * self.dt  # this is not the nice way! there are discretization techniques for linear systems
        else:  # the pendulum is modeled by nonlinear dynamics
            phi = (self.x - cntrl_u)[0, 0] / self.l
            ddx = - 9.81 * np.tan(phi)
            dx = self.x[1, 0]
            self.x += np.array([[dx], [ddx]]) * self.dt

        ''' logging '''
        self.log_idx += 1
        self.x_log[self.log_idx] = self.x.flatten()
        self.u_log[self.log_idx] = u

        return self.x

    def plot(self):
        t_range = np.arange(0., self.dt * self.steps, self.dt)
        fig, ax = plt.subplots(1, figsize=(10, 5))
        ax.plot(t_range, self.x_log[:, 0], label=r'$x$')
        ax.plot(t_range, self.x_log[:, 1], label=r'$\dot x$')

        ax.plot(t_range, self.u_log, label=r'$u$')
        ax.legend()

    def create_video(self):
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


if __name__ == '__main__':  # this code is only used when we directly execute this file (not when importing it)

    ''' simulation settings'''
    dt = 0.05
    steps = 100
    t_range = np.arange(dt, dt * steps, dt)

    pendulum = Pendulum(length=3, x0=np.array([[1.], [0.]]), linear=False, dt=dt,
                        steps=steps)  # create an instance of the pendulum
    k = np.array([[-1., -1.]])  # define the control feedback

    ''' simulation '''
    for t in t_range:
        u = k @ pendulum.x
        pendulum.step(u)

    ''' visualization '''
    pendulum.plot()
    pendulum.create_video()
    plt.show()
