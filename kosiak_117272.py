#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math as m
import copy as cp
import math as math

# params
M = 20
A_br = 3  # A borrow count estimate
A_rr = 3  # A return count estimate
B_br = 4
B_rr = 2
car_value_1 = 0
car_value_2 = 0
car_lost_penalty = 0
car_borrow_reward = 100
discount_rate = 0.9
cars_move_penalty = 20


def poisson(pi, n): return m.pow(pi, n) / m.factorial(n) * m.pow(m.e, -pi)


def print_number_2d(ar):
    print(('\n'.join([' '.join(['{0:2d}'.format(el) for el in line]) for line in ar])))


def print_float_2d(ar):
    print(('\n'.join([', '.join(['{0:.3f}'.format(el) for el in line]) for line in ar])))


def get_initial_u():
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
            composite_prob = br_probs[i] * rr_probs[j]
            c_probs[j - i + M] += composite_prob

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


def calculate_move_result(cars_to_move, cars_in_a, cars_in_b, u, transition_probs):

    # calculating position after moving cars
    next_i = cars_in_a - cars_to_move
    next_j = cars_in_b + cars_to_move

    assert next_i >= 0
    assert next_j >= 0

    acc = math.fabs(cars_to_move) * -cars_move_penalty

    for i in range(next_i - M, next_i + M + 1):
        for j in range(next_j - M, next_j + M + 1):

            current_val = -car_lost_penalty

            reward_val = 0

            temp_i = min([max([0, i]), M])
            temp_j = min([max([0, j]), M])

            current_val += u[temp_i][temp_j]

            # value smaller than current is equal to amount of borrowed cars
            if temp_i < next_i:
                reward_val += car_borrow_reward * (next_i - temp_i)

            if temp_j < next_j:
                reward_val += car_borrow_reward * (next_j - temp_j)

            # Belman equation R(s,a) {without adding current state reward u[i,j]}
            acc += discount_rate * (current_val + reward_val) * transition_probs[i - (next_i - M)][j - (next_j - M)]

    return acc


def iterate_policy(policy, u, trans_props_table):
    next_u = [[0 for i in range(M + 1)] for j in range(M + 1)]
    next_policy = cp.deepcopy(policy)

    for i in range(M + 1):
        for j in range(M + 1):

            # we can borrow only as mny cars as we have
            # and moving more cars that
            from_a_to_b = min([i, M - j, 5])
            from_b_to_a = max([-5, -j, -(M - i)])

            # calculate first value
            max_cu = calculate_move_result(from_b_to_a, i, j, u, trans_props_table)
            p = from_b_to_a

            for k in range(from_b_to_a + 1, from_a_to_b + 1):
                temp_u = calculate_move_result(k, i, j, u, trans_props_table)

                if temp_u > max_cu:
                    max_cu = temp_u
                    p = k

            next_policy[i][j] = p
            next_u[i][j] = max_cu

    return [next_policy, next_u]


def get_policy(initial_u, trans_props):

    # if True prints 'u' matrix
    print_u = False

    initial_policy = get_initial_policy()

    p, u = iterate_policy(initial_policy, initial_u, trans_props)

    if print_u:
        print_float_2d(u)
    else:
        print_number_2d(p)

    print(''.join(['-' for i in range(70)]))

    # count of iterations
    iterations = 50

    for i in range(iterations):
        p, u = iterate_policy(p, u, trans_props)

        if print_u:
            print_float_2d(u)
        else:
            print_number_2d(p)

        print(''.join(['-' for i in range(70)]))

    return initial_policy


def main():
    initial_u = get_initial_u()

    #  print(np.matrix(initial_u))

    trans_props = get_a_props_table()

    # print_float_2d(trans_props)
    # return

    # print_float_2d(trans_rewards)
    # return

    get_policy(initial_u, trans_props)


if __name__ == '__main__':
    main()
