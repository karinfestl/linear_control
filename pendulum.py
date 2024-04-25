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
        self.A = np.array([[0, 1],
                           [-g / length, 0]])
        self.b = np.array([[0],
                           [g / length]])
        self.c = np.array([[1, 0]])
        self.l = length
        self.linear = linear

        '''initial state'''
        self.x = x0
        self.y = self.c @ self.x

    def step(self, cntrl_u, obs_e=0):
        """ increment the system state """
        if self.linear:  # the pendulum is modeled by linear dynamics
            dx = self.A @ self.x + self.b * cntrl_u + obs_e
            self.x += dx * self.dt  # this is not the nice way! there are discretization techniques for linear systems
        else:  # the pendulum is modeled by nonlinear dynamics
            phi = (self.x - cntrl_u)[0, 0] / self.l
            ddx = - 9.81 * np.tan(phi)
            dx = self.x[1, 0]
            self.x += np.array([[dx], [ddx]]) * self.dt

        self.y = (self.c @ self.x)

        ''' logging '''
        self.log_idx += 1
        self.x_log[self.log_idx] = self.x.flatten()
        self.u_log[self.log_idx] = u

        return self.x

    def plot(self, ax_t=None, ax_x=None, suffix=''):
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
    # eigenvalues_f = [complex(-1.5, 0.5), complex(-1.5, -0.5), complex(-1, 1), complex(-1, -1)]  # desired eigenvalues

    ''' compute transformation '''
    C = controlability_matrix(A, b)
    alpha = characteristic_polynomial(A)

    Cbarinv = np.eye(2)
    Cbarinv[0, 1] = alpha[1]

    Pinv = C @ Cbarinv
    P = np.linalg.inv(Pinv)

    ''' compute feedback gain '''
    alphabar = np.poly(eigenvalues_f)

    kbar = alphabar[1:] - alpha[1:]
    kbar = np.expand_dims(kbar, 0)
    k = kbar @ P

    return k


if __name__ == '__main__':  # this code is only used when we directly execute this file (not when importing it)

    ''' simulation settings'''
    dt = 0.05
    steps = 200
    t_range = np.arange(dt, dt * steps, dt)

    pendulum = Pendulum(length=3, x0=np.array([[1.], [0.]]), linear=False, dt=dt,
                        steps=steps)  # create an instance of the pendulum
    pendulum_obs = Pendulum(length=3, x0=np.array([[0.], [0.]]), linear=True, dt=dt,
                            steps=steps)  # create an instance of the pendulum

    k = np.array([[-2., -1.]]) * 0  # define the control feedback
    l = np.array([[-2], [-2]])  # define the observer feedback
    r = 0.

    k = -compute_feedback_gain(pendulum.A, pendulum.b, eigenvalues_f=[complex(-1.5, 0.1), complex(-1.5, -0.1)])
    print("k {0}".format(k))

    ''' simulation '''
    for t in t_range:
        u = k @ np.array([[pendulum.y[0,0]], [pendulum_obs.x[1,0]]]) + r * (1 - k[0, 0])
        obs_e = l @ (pendulum_obs.y - pendulum.y)
        pendulum.step(u)
        pendulum_obs.step(u, obs_e)

    ''' visualization '''
    ax_t, ax_x = pendulum.plot()
    pendulum_obs.plot(ax_t, ax_x, suffix='obs')
    pendulum.create_video()
    plt.show()
