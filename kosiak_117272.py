#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math as m

# const
M = 20
A_br = 3 # A borrow estimate
A_rr = 3 # A return estimate
B_br = 4
B_rr = 2


def poisson(pi, n): return m.pow(pi, n) / m.factorial(n) * m.pow(m.e, -pi)


def get_rewards_table():
    car_value_1 = 1
    car_value_2 = 2

    rewards = [
        [j * car_value_1 + i * car_value_2 for j in range(M + 1)] for i in range(M + 1)
    ]

    return rewards


def get_a_props_table():

    a_br_probs = [poisson(A_br, i) for i in range(M + 1)]
    a_rr_probs = [poisson(A_rr, i) for i in range(M + 1)]

    b_br_probs = [poisson(B_br, i) for i in range(M + 1)]
    b_rr_probs = [poisson(B_rr, i) for i in range(M + 1)]

    a_c_probs = [0 for r in range(2 * M + 1)]

    for i in range(M + 1):
        for j in range(M + 1):
            a_c_probs[j - i + M] += a_br_probs[i] * a_rr_probs[j]

    print('\n'.join([str([i - M, str(a_c_probs[i])]) for i in range(2 * M + 1)]))


    # return get_a_props


def get_policy(rewards):
    policy = np.random.randint(-5, 5, (M + 1, M + 1))

    return policy

    for i in range(M + 1):
        for j in range(M + 1):
            print("{:2d} ".format(policy[i][j]), end='')
        print()

    return policy


def main():
    rewards = get_rewards_table()
    # print(np.matrix(rewards))
    print(np.matrix(get_a_props_table()))
    get_policy(rewards)


if __name__ == '__main__':
    main()
