# -*- coding: utf-8 -*-

# --------------------------------------------------------
# File Name: gen_data.py
#
# Written by Chao Wu and Jiaming Qiu
#
# Reference: 'Extracting Parameters from a Planetary 
#             Microlensing Event Without a Computer'
# --------------------------------------------------------

import argparse
import os.path

import scipy as S
from pylab import *

temp_dir = ''
temp_pic_dir = ''

def simuA(t, u0, tE, t0=0):
    """
    A~ 1/u
    u = u0 + S.absolute((t-t0)/TE)
    A= (u**2 +2)/(sqrt(u**2 + 4)*u)
    """

    u = u0 + S.absolute((t - t0) / tE)
    # A= S.absolute((u**2 +2)/(S.sqrt(u**2 + 4)*u))
    A = (u ** 2 + 2) / (S.sqrt(u ** 2 + 4) * u)
    return A


def generate_template(dataset):
    print("Generating templates: {}".format(dataset))

    step = 0.00017361 # 15 sec = 0.00017361 day
    tL = 1.0 / 24.0 * S.linspace(1.0, 23.0, 12.0)
    tE = tL / 7.7

    if dataset == 'threshold':
        u0 = S.array([0.9, 0.1, 0.01])
    elif dataset == 'full' or dataset == 'gwac':
        u0 = S.linspace(0.001, 1.0, 5)
    else:
        raise Exception('dataset type not supported')

    # tE= [(10 ** (n * S.log10(2.)) + S.log10(1.)) / 24. for n in range(0, 7)]
    # u0 = S.logspace(-0.0969, -3)

    oo = []
    for tE0, tL0 in zip(tE, tL):
        t = S.linspace(0.0, tL0, (tL0 - 0.0) / step + 1.0)
        t0 = 0.5 * tL0
        for u00 in u0:
            A = simuA(t, u00, tE0, t0=t0)
            m = -2.5 * S.log10(A)
            m = m - m.max()
            # print t,m
            file0 = os.path.join(temp_dir, '%5.3f_%5.4f.dat' % (tE0, u00))

            with open(file0, 'w') as ff:
                # print >>ff, "#-- tE=%5.3f,u0=%5.4f,t0=%5.2f " %(tE0,u00,t0)
                print("#-- tE=%5.3f,u0=%5.4f" % (tE0, u00), file=ff)
                print("#- col1: t - t0 (day), col2: mag ", file=ff)
                for i, tt0 in enumerate(t - t0):
                    print("%5.8f   %5.2f" % (tt0, m[i]), file=ff)

            figure()
            plot(t - t0, m, '.')
            png0 = os.path.join(temp_pic_dir, '%5.3f_%5.4f.png' % (tE0, u00))
            title('tE=%5.3f,u0=%5.4f' % (tE0, u00))
            ylim(m.max() + 0.1, m.min() - 0.1)
            ylabel('mag')
            xlabel('t - t0 (day)')
            savefig(png0)
            close()
            oo.append([t - t0, m, u00, tE0, t0])
    return oo

def main():
    parser = argparse.ArgumentParser(description='Script for generating datasets')
    parser.add_argument('--dataset', type=str, help='name of dataset')
    args = parser.parse_args()

    global temp_dir
    global temp_pic_dir
    temp_dir = 'temp/{}'.format(args.dataset)
    temp_pic_dir = 'temp_pic/{}'.format(args.dataset)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not os.path.exists(temp_pic_dir):
        os.makedirs(temp_pic_dir)

    generate_template(args.dataset)

if __name__ == '__main__':
    main()
    
