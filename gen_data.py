# -*- coding: utf-8 -*-

# --------------------------------------------------------
# File Name: gen_data.py
#
# Written by Jiaming Qiu
# --------------------------------------------------------

import os
import os.path
import argparse
from itertools import product
from multiprocessing import Pool

import numpy as np
from numpy.random import normal, randint
import scipy as S
import pylab

from util import log_print

# sample interval
step_in_second = 15

# step in the unit of day
step = step_in_second / 86400

# maximum amplitude of background signal
sin_amp_threshold = 0.25

# continuity of samples(dataset-dependent)
continuous = True

# sample seconds in one day(dataset-dependent)
sample_seconds_in_one_day = 28800

# sample days(dataset-dependent)
num_days = 24

# sample points in one day(dataset-dependent)
sample_points_in_one_day = sample_seconds_in_one_day // step_in_second

# length of history part of the sample(dataset-dependent)
history_len = sample_points_in_one_day * num_days

# directory to save data(dataset-dependent)
data_path = ('./data/full')

# directory to save pictures(dataset-dependent)
pic_path = ('./data_pic/full')

#-------------------------- templates -------------------------

# for full dataset and gwac dataset：
# composed by 12 different tLs and 5 different u0s, 60 in total
temp_path = ('./temp/full')

# for threshold dataset：
# composed by 12 different tLs and 3 different u0s, 36 in total
# temp_path = ('./temp/threshold')

temp_list = []

#------------------------------- σ --------------------------------

# for full dataset：
# range of σ：[0.005, 0.021, 0.037, 0.05, 0.068, 0.084, 0.1, 0.15, 0.2], 9
# in total
sigma_range = np.concatenate((S.linspace(0.005, 0.1, 7), [0.15, 0.2]))

# for threshold dataset：
# range of σ: [0.005, 0.01, 0.05, 0.1], 4 in total
# sigma_range = S.array([0.005, 0.01, 0.05, 0.1])

# for gwac dataset：
# range of σ: [0.068], 4 in total
# sigma_range = S.array([0.068])

#----------------------------- relative saliency ------------------------------

# for full and threshold dataset:
# range of relative saliency: [1.0/5.0, 1/4.0, 1/3.0, 1/2.0, 1.0, 2.0, 3.0, 4.0, 5.0]
# 9 in total
part_b = S.linspace(1.0, 5.0, 5)
part_a = [1 / elem for elem in part_b]
part_a = part_a[1:]
part_a.reverse()
relative_amp_range = np.concatenate([part_a, part_b])

#------------------------------- A ---------------------------------

# for gwac dataset:
# range of A: [0.25]
# 1 in total
A_range = [0.25]

#------------------------------- T ---------------------------------

# range of T: [0.2 ,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# 9 in total
T_range = [float(elem) for elem in list(S.linspace(0.2, 1.0, 9))]

#------------------------------- φ ---------------------------------

# range of φ: [0, 1/6, 2/6, 3/6, 4/6, 5/6] * 2π
phi_range = [float(elem) for elem in list(S.linspace(0, 2 * np.pi, 7)[:-1])]


def set_global_variables(dataset):
    global continuous
    global sample_seconds_in_one_day
    global num_days
    if dataset == 'full' or dataset == 'threshold':
        continuous = True
        sample_seconds_in_one_day = 86400
        num_days = 3
    elif dataset == 'gwac':
        continuous = False
        sample_seconds_in_one_day = 28800
        num_days = 24

    global sample_points_in_one_day
    global history_len
    sample_points_in_one_day = sample_seconds_in_one_day // step_in_second
    history_len = sample_points_in_one_day * num_days

    global data_path
    global pic_path
    data_path = ('./data/{}'.format(dataset))
    pic_path = ('./data_pic/{}'.format(dataset))

    global temp_path
    global temp_list
    if dataset == 'full':
        temp_path = ('./temp/full')
    elif dataset == 'gwac':
        temp_path = ('./temp/gwac')
    else:
        temp_path = ('./temp/threshold')
    temp_list = os.listdir(temp_path)

    global sigma_range
    if dataset == 'full':
        sigma_range = np.concatenate((S.linspace(0.005, 0.1, 7), [0.15, 0.2]))
    elif dataset == 'threshold':
        sigma_range = S.array([0.005, 0.01, 0.05, 0.1])
    else:
        sigma_range = S.array([0.068])

    global phi_range
    if dataset == 'full' or dataset == 'gwac':
        phi_range = [float(elem) for elem in list(S.linspace(0, 2 * np.pi, 7)[:-1])]
    else:
        phi_range = [0]

def gen(sigma, dataset, step_in_second, step, sin_amp_threshold, continuous,
        sample_seconds_in_one_day, num_days, sample_points_in_one_day,
        history_len, data_path, pic_path, temp_path, temp_list, relative_amp_range,
        A_range, T_range, phi_range):
    
    gen_num = 0

    for temp_file in temp_list:
        with open(os.path.join(temp_path, temp_file)) as file_handle:
            data_frame = np.loadtxt(file_handle)
        temp_y = data_frame[:, 1]
        temp_amp = abs(temp_y.min())

        if dataset == 'threshold' or dataset == 'full':
            param_combinations = product(relative_amp_range, T_range, phi_range)
        elif dataset == 'gwac':
            param_combinations = product(A_range, T_range, phi_range)
        else:
            raise Exception('dataset type not supported')
        
        for A_or_relative_amp, T, phi in param_combinations:
            if dataset == 'threshold' or dataset == 'full':
                A = temp_amp / A_or_relative_amp
                if A <= 1.5:
                    gen_num += 1
            else:
                gen_num += 1

    log_print("Number of samples to be generated: %d" % gen_num)

    ind = 0

    for temp_file in temp_list:
        # 1.read gravitational microlensing signal template
        with open(os.path.join(temp_path, temp_file)) as file_handle:
            data_frame = np.loadtxt(file_handle)
        temp_x = data_frame[:, 0]
        temp_y = data_frame[:, 1]
        tE = float(temp_file.split('_')[0])

        # 2.cutting of template
        i = 0
        while True:
            if temp_y[i] != temp_y[i + 1]:
                break
            i += 1
        cutted_temp_x = temp_x[i: -(i + 1)]
        cutted_temp_y = temp_y[i: -(i + 1)]

        temp_len = len(cutted_temp_x)  # template length T'
        temp_amp = abs(cutted_temp_y.min())  # template amplitude A'
        
        if dataset == 'threshold' or dataset == 'full':
            param_combinations = product(relative_amp_range, T_range, phi_range)
        elif dataset == 'gwac':
            param_combinations = product(A_range, T_range, phi_range)
        else:
            raise Exception('dataset type not supported')

        for A_or_relative_amp, T, phi in param_combinations:
            # num of outliers
            num_outliers = 3

            # sample point numbers in a period
            sin_period = int(T * sample_points_in_one_day)

            if dataset == 'gwac':
                # relative saliency
                A = A_or_relative_amp
                relative_amp = temp_amp / A
            else:
                # amplitude of sine curve
                relative_amp = A_or_relative_amp
                A = temp_amp / A_or_relative_amp
                if A > 1.5:
                    continue

            # generate a single period of the background signal
            single_period_sin = A * np.sin(
                S.linspace(0 + phi, 2 * np.pi + phi, sin_period)
            )

            # final length
            if dataset == 'full' or dataset == 'threshold':
                final_len = history_len + temp_len
            else:
                final_len = history_len

            dup_time = final_len // sin_period
            res_len = final_len % sin_period
            basic_sin = np.tile(single_period_sin, dup_time)
            res_sin = single_period_sin[:res_len]
            basic_sin = np.concatenate([basic_sin, res_sin], axis=0)

            if continuous:
                final_x = [i * step for i in range(final_len)]
                noised_sin = basic_sin + normal(0, sigma, len(basic_sin))
            else:
                uncontinuous_basic_sin = np.array([])

                final_x = []
                for day in range(num_days):
                    # The phase of first section is fixed for injecting ML signal
                    if day == 0:
                        random_start_ind = 0
                    else:
                        random_start_ind = randint(0, final_len // 2)

                    uncontinuous_basic_sin = np.concatenate(
                        [uncontinuous_basic_sin,
                        basic_sin[random_start_ind: random_start_ind + sample_points_in_one_day]],
                        axis=0
                    )

                    day_x = [day + i * step
                         for i in range(sample_points_in_one_day)]
                    final_x.extend(day_x)

                assert len(basic_sin) == len(uncontinuous_basic_sin)
                noised_sin = uncontinuous_basic_sin \
                             + normal(0, sigma, len(basic_sin))

            # We inject the ML signal into the first section of background signal
            # and then horizontallyreverse the signal to ensure that the ML signal
            # is overlapped on the desired phase of the background signal
            final_y = noised_sin
            final_y[:temp_len] = final_y[:temp_len] + cutted_temp_y
            final_y = final_y[::-1]

            assert len(final_y) == final_len
            assert len(final_x) == len(final_y)

            tE_col = np.tile(tE, final_len)
            temp_start_col = np.tile(final_x[-temp_len:][0], final_len)
            temp_end_col = np.tile(final_x[-temp_len:][-1], final_len)

            A_prime = temp_amp
            T_prime = cutted_temp_x[-1] - cutted_temp_x[0]

            # write data
            data_name = "%.3f_%.3f_%.3f_%.3f_%.3f_%.3f.dat" % (sigma,
                                                               T_prime,
                                                               T,
                                                               A_prime,
                                                               relative_amp,
                                                               phi)

            with open(os.path.join(data_path, data_name), 'w') as data_file:
                for i in range(final_len):
                    # injecting outlier points
                    if randint(10000) == 0:
                        if num_outliers > 0:
                            final_y[i] = -A_prime + normal(0, sigma)
                            num_outliers -= 1

                    print(
                        "%5.8f    %5.2f    %5.3f    %5.8f    %5.8f" %
                        (final_x[i], final_y[i], tE_col[i],
                         temp_start_col[i], temp_end_col[i]),
                        file=data_file
                    )

            # write image
            pic_name = "%.3f_%.3f_%.3f_%.3f_%.3f_%.3f.png" % (sigma,
                                                              T_prime,
                                                              T,
                                                              A_prime,
                                                              relative_amp,
                                                              phi)

            pic_file = os.path.join(pic_path, pic_name)

            pylab.figure()
            pylab.plot(final_x, final_y, '.')

            pylab.title("%.3f_%.3f_%.3f_%.3f_%.3f_%.3f.png" % (sigma,
                                                               T_prime,
                                                               T,
                                                               A_prime,
                                                               relative_amp,
                                                               phi))

            pylab.ylim(final_y.max() + 0.1, final_y.min() - 0.1)
            pylab.ylabel('mag')
            pylab.xlabel('t-t0(day)')
            pylab.savefig(pic_file)
            pylab.close()

            ind += 1
            log_print("finished generating: %s/%s, %d samples generated"
                  % (data_path, data_name, ind))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for generating datasets')
    parser.add_argument('--dataset', type=str, help='name of dataset')
    args = parser.parse_args()

    if args.dataset != 'threshold' and args.dataset != 'full' and args.dataset != 'gwac':
        raise Exception('dataset type not supported')

    set_global_variables(args.dataset)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
 
    p = Pool()
    for sigma in sigma_range:
        p.apply_async(gen, args=(sigma, args.dataset, step_in_second, step, sin_amp_threshold, continuous,
                                 sample_seconds_in_one_day, num_days, sample_points_in_one_day, history_len,
                                 data_path, pic_path, temp_path, temp_list, relative_amp_range, A_range, T_range,
                                 phi_range))
    p.close()
    p.join()
    log_print("All subprocesses done.")

    #gen(0.0, args.dataset, step_in_second, step, sin_amp_threshold, 
    #    continuous, sample_seconds_in_one_day, num_days, 
    #    sample_points_in_one_day, history_len, data_path, pic_path, 
    #    temp_path, temp_list, relative_amp_range, A_range, T_range,
    #    phi_range)
