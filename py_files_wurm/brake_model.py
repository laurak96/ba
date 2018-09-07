import pickle
import numpy as np
import pandas as pd
import rule_generator
from scipy.interpolate import InterpolatedUnivariateSpline


class BrakeTest:
    """
    Simulates a brake test from v_init to 0.
    """

    def __init__(self, v_init, t_s, mu_f, weight):
        """
        Constructor
        :param v_init: Velocity at the begining of the test
        :param t_s: Time (in seconds) passed till the maximal deceleration is reached (if it's not between 0.14 and 0.18
         the brake test is considered as  faulty).
        :param mu_f: Friction coefficient to compute the brake force.
        :param weight: Weight of the car used for computing the brake force and maximal deceleration.
        """
        # The time till max. deceleration is reached
        self.t_s = t_s
        # friction coefficient
        self.mu_f = mu_f
        # Weight of the car
        self.weight = weight
        # brake force
        self.f_a = weight * 9.81 * self.mu_f
        self.v_init = v_init
        # Maximal deceleration
        self.a = -self.f_a / weight
        # Time passed till the car reaches zero velocity
        self.t_v = round(self.time_till_halt(), 2)
        # Range of time points till max. deceleration is reached. Stepping in milliseconds.
        self.range_s = np.arange(0.0, t_s + 0.001, 0.001)
        # Range of time points starting when the car reaches max. deceleration till car reaches zero velocity.
        # Stepping in milliseconds.
        self.range_v = np.arange(0.0, self.t_v + 0.001, 0.001)
        # The over all time range of the test. stepping in milliseconds
        self.range_comp = np.arange(0.0, t_s + self.t_v + 0.002, 0.001)
        # Deceleration values while time till max deceleration is passed
        self.a_ts = self.brake_acceleraction(self.range_s)
        # Complete acceleration values till car reaches zero velocity. Computed every millisecond.
        self.a_comp = np.concatenate((self.a_ts, np.full(self.range_comp.size - self.a_ts.size, self.a)))
        # Interpolated function of the deceleration
        self.f_of_a = InterpolatedUnivariateSpline(self.range_comp, self.a_comp)
        # Velocity till max. deceleration is reached. Computed every millisecond
        self.v2 = self.velocity_threshold(self.range_s)
        # Velocity while max. deceleration till it reaches (nearly) zero
        self.v3 = self.velocity_norm(self.range_v)
        # Complete velocitiy values till car reaches zero velocity. Computed every millisecond.
        self.v_comp = np.concatenate((self.v2, self.v3))
        # Interpolated function of velocity
        self.f_of_v = InterpolatedUnivariateSpline(self.range_comp, self.v_comp)
        # Distance passed while the decceleration is finished. Last values equals the brake distance.
        self.s_comp = self.compute_distances()
        # Interpolated function of the distance
        self.f_of_s = InterpolatedUnivariateSpline(self.range_comp, self.s_comp)
        # DataFrame containing the computed deceleration, velocity and distance values
        self.report = pd.DataFrame(columns=['a', 'v', 's'], index=self.range_comp)
        self.report['a'] = self.a_comp
        self.report['v'] = self.v_comp
        self.report['s'] = self.s_comp

    def brake_distance(self):
        """
        Computes the brake distance without integral.
        :return: brake distance.
        """
        return self.v_init * (self.t_s / 2) - ((self.v_init ** 2) / (2 * self.a) + (self.a / 24) * self.t_s ** 2)

    def compute_distances(self):
        """
        Computes the distances every millisecond with help of the integral of the velocity.
        :return: 1d arry of distances.
        """
        dist = []
        for i in self.range_comp:
            dist.append(self.f_of_v.integral(0, i))
        return np.asarray(dist)

    def brake_acceleraction(self, t):
        """
        Computes the acceleration during the time for reaching max. deceleration.
        :param t: Time point for which the deceleration shoud be computed.
        :return: Acceleration
        """
        return (self.a / self.t_s) * t

    def velocity_threshold(self, t):
        """
        Computes the veleocity during the time of reaching the max. deceleration
        :param t: Time point for which the deceleration shoud be computed.
        :return: Velocity.
        """
        return self.v_init + (self.a / (2 * self.t_s)) * t ** 2

    def time_till_halt(self):
        """
        Computes the time till the car reaches zero velocity.
        :return: Time
        """
        return (self.v_init / (-self.a)) - (self.t_s / 2)

    def velocity_norm(self, t):
        """
        Computes velocity at time t after max. acceleration is reached.
        :param t: Time point
        :return: Velocity.
        """
        return self.v2[-1] + self.a * t

    @staticmethod
    def linear_dep_mu_ts(t_s):
        """
        Computes the synthetic linear dependence between t_s and the friction coefficient.
        :param t_s: Threshold time.
        :return: Values of linear dependence.
        """
        return -(20 / 23) * t_s + (106 / 115)

    @staticmethod
    def generate_test_specimens(n=100):
        """
        Generates muliple correct and faulty brake tests.
        :param n: Number of brake test to compute for each case.
        :return: List of positive example, negative examples.
        """
        pos_ts = np.random.uniform(0.14, 0.19, n)
        # pos_mu = BrakeTest.linear_dep_mu_ts(pos_ts)
        pos_mu = np.random.uniform(0.75, 0.8, n)

        neg_ts = np.random.uniform(0.19, 0.30, n)
        # neg_mu = BrakeTest.linear_dep_mu_ts(neg_ts)
        neg_mu = np.random.uniform(0.66, 0.75, n)

        pos_brk = []
        for i in range(0, pos_ts.size):
            bm = BrakeTest(13, pos_ts[i], pos_mu[i], 1000)
            pos_brk.append(bm)

        neg_brk = []
        for i in range(0, neg_ts.size):
            bm = BrakeTest(13, neg_ts[i], neg_mu[i], 1000)
            neg_brk.append(bm)

        return pos_brk, neg_brk

    @staticmethod
    def generate_rules(n=100, verbose=True, length=140, delta=0.1, lag=80, alpha=-1, beta=-1):
        """
        Generates the binary rules for each test case. And pickles the result in file sample.pkl at the same directory.
        :param n: Number of test cases.
        :param verbose: Enables verbosity.
        :param length: Length at which the time series is cut.
        :param delta: The delta value.
        :param lag: The lag.
        :param alpha: The alpha value.
        :param beta: The beta value.
        :return: list of positive, negative brake test and the corresponding BinaryRuleGenerator objects.
        """
        # List of BinaryRuleGenerator objects for the positive samples
        pos_brs = []
        # List of BinaryRuleGenerator objects for the negative samples
        neg_brs = []
        # generate the samples
        pos_brk, neg_brk = BrakeTest.generate_test_specimens(n=n)
        # For each sample generate the rules
        for i in range(0, len(pos_brk)):
            x = pos_brk[i].range_comp
            a = pos_brk[i].a_comp
            f_a = InterpolatedUnivariateSpline(x, a)
            dxa = f_a.derivative()
            slope_a = dxa(x)
            a = a[0:length:1]
            slope_a = slope_a[0:length:1]

            v = pos_brk[i].v_comp
            f_v = InterpolatedUnivariateSpline(x, v)
            dxv = f_v.derivative()
            slope_v = dxv(x)
            v = v[0:length:1]
            slope_v = slope_v[0:length:1]

            s = pos_brk[i].s_comp
            f_s = InterpolatedUnivariateSpline(x, s)
            dxs = f_s.derivative()
            slope_s = dxs(x)
            s = s[0:length:1]
            slope_s = slope_s[0:length:1]

            dataset = pd.DataFrame()
            dataset['a'] = a
            dataset['v'] = v
            dataset['s'] = s

            slopeset = pd.DataFrame()

            slopeset['a'] = slope_a
            slopeset['v'] = slope_v
            slopeset['s'] = slope_s

            brs = rule_generator.RuleGenerator()
            brs.data_from_frame(dataset.T, slopeset.T)
            brs.fit(delta, alpha1=alpha, lag=lag, beta=beta)
            if verbose:
                print('Done pos ' + str(i))

            pos_brs.append(brs)

            x = neg_brk[i].range_comp
            a = neg_brk[i].a_comp
            f_a = InterpolatedUnivariateSpline(x, a)
            dxa = f_a.derivative()
            slope_a = dxa(x)
            a = a[0:length:1]
            slope_a = slope_a[0:length:1]

            v = neg_brk[i].v_comp
            f_v = InterpolatedUnivariateSpline(x, v)
            dxv = f_v.derivative()
            slope_v = dxv(x)
            v = v[0:length:1]
            slope_v = slope_v[0:length:1]

            s = neg_brk[i].s_comp
            f_s = InterpolatedUnivariateSpline(x, s)
            dxs = f_s.derivative()
            slope_s = dxs(x)
            s = s[0:length:1]
            slope_s = slope_s[0:length:1]

            dataset = pd.DataFrame()
            dataset['a'] = a
            dataset['v'] = v
            dataset['s'] = s

            slopeset = pd.DataFrame()

            slopeset['a'] = slope_a
            slopeset['v'] = slope_v
            slopeset['s'] = slope_s

            brs = rule_generator.RuleGenerator()
            brs.data_from_frame(dataset.T, slopeset.T)
            brs.fit(delta, alpha1=alpha, lag=lag, beta=beta)

            neg_brs.append(brs)
            if verbose:
                print('Done neg ' + str(i))
            pickle.dump([pos_brk, neg_brk, pos_brs, neg_brs], open('pickle/sample.pkl', 'wb'))
        return pos_brk, neg_brk, pos_brs, neg_brs


if __name__ == "__main__":
    print(BrakeTest.linear_dep_mu_ts(0.30))
