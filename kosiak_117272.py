#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math as m
import copy as cp
import math as math

# conste3
M = 20
A_br = 3  # A borrow estimate
A_rr = 3  # A return estimate
B_br = 4
B_rr = 2
car_value_1 = 0
car_value_2 = 0
car_lost_penalty = 5
car_borrow_reward = 5
discount_rate = 0.9
cars_move_penalty = 1


def poisson(pi, n): return m.pow(pi, n) / m.factorial(n) * m.pow(m.e, -pi)


def print_number_2d(ar):
    print(('\n'.join([' '.join(['{0:2d}'.format(el) for el in line]) for line in ar])))


def print_float_2d(ar):
    print(('\n'.join([', '.join(['{0:.3f}'.format(el) for el in line]) for line in ar])))


def get_rewards_table():
    rewards = [
        [j * car_value_2 + i * car_value_1 for j in range(M + 1)] for i in range(M + 1)
        # [0.1 for j in range(M + 1)] for i in range(M + 1)
    ]

    return rewards


def get_transitions_probability(borrow_rate, return_rate):
    #  probability distribution for borrowing
    br_probs = [poisson(borrow_rate, i) for i in range(M + 1)]
    #  probability distribution for returning
    rr_probs = [poisson(return_rate, i) for i in range(M + 1)]

    c_probs = [0 for r in range(2 * M + 1)]

    for i in range(M + 1):
        for j in range(M + 1):
            c_probs[j - i + M] += br_probs[i] * rr_probs[j]

    return c_probs


def get_a_props_table():

    a_c_probs = get_transitions_probability(A_br, A_rr)
    b_c_probs = get_transitions_probability(B_br, B_rr)

    c_global_transition_probs = [[0 for i in range(2 * M + 1)] for j in range(2 * M + 1)]

    for i in range(2 * M + 1):
        for j in range(2 * M + 1):
            c_global_transition_probs[i][j] = a_c_probs[i] * b_c_probs[j]

    return c_global_transition_probs


def get_initial_policy():
    return [[1 for i in range(M + 1)] for j in range(M + 1)]


def calculate_move_result(cars_to_move, cars_in_a, cars_in_b, fixed_rewards, u, transition_probs):
    next_i = cars_in_a - cars_to_move
    next_j = cars_in_b + cars_to_move

    assert next_i >= 0
    assert next_j >= 0

    acc = 0

    for i in range(next_i - M, next_i + M + 1):
        for j in range(next_j - M, next_j + M + 1):

            current_val = -car_lost_penalty

            if 0 <= i < M + 1 and 0 <= j < M + 1:
                current_val = u[i][j]

            acc += current_val * transition_probs[i - (next_i - M)][j - (next_j - M)]

    # if 0 <= i < M + 1 and 0 <= j < M + 1:
    #   acc += discount_rate * u[cars_in_a][cars_in_b]

    return acc


def iterate_policy(policy, fixed_state_rewards, u, trans_props_table):
    next_u = [[0 for i in range(M + 1)] for j in range(M + 1)]
    next_policy = cp.deepcopy(policy)

    for i in range(M + 1):
        for j in range(M + 1):
            max_cu = calculate_move_result(0, j, i, fixed_state_rewards, u, trans_props_table)
            p = 0

            for k in range(1, min([i, 5 + 1])):
                temp_u = calculate_move_result(k, i, j, fixed_state_rewards, u, trans_props_table)

                if temp_u > max_cu:
                    max_cu = temp_u
                    p = k

            for k in range(max([-5, -j]), 0):
                temp_u = calculate_move_result(k, i, j, fixed_state_rewards, u, trans_props_table)

                if temp_u > max_cu:
                    max_cu = temp_u
                    p = k

            next_policy[i][j] = p
            next_u[i][j] = max_cu * discount_rate

    return [next_policy, next_u]


def get_policy(fixed_state_rewards, trans_props_table):

    print_u = False

    policy = get_initial_policy()

    p, u = iterate_policy(policy, fixed_state_rewards, fixed_state_rewards, trans_props_table)

    if print_u:
        print_float_2d(u)
    else:
        print_number_2d(p)

    print(''.join(['-' for i in range(70)]))

    for i in range(10):
        p, u = iterate_policy(p, fixed_state_rewards, u, trans_props_table)

        if print_u:
            print_float_2d(u)
        else:
            print_number_2d(p)

        print(''.join(['-' for i in range(70)]))

    return policy


def main():
    rewards = get_rewards_table()
    #  print(np.matrix(rewards))
    trans_props_table = get_a_props_table()
    #  print_2d(trans_props_table)
    get_policy(rewards, trans_props_table)


if __name__ == '__main__':
    main()
