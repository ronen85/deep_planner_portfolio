#!/usr/bin/env python

if __name__ == '__main__':
    model_keys_list = ['net_1', 'net_1_double', 'net_1_triple', 'net_2', 'net_2_double',
            'net_2_triple', 'net_3', 'net_3_double', 'net_3_triple']
    exp_cmd_list = []

    for model in model_keys_list:
        for use_ft in [0,1]:
            for optim in ['Adam', 'SGD']:
                cmd = 'python3 main.py ' + model + ' ' + str(use_ft) + ' -optimizer ' + optim
                exp_cmd_list.append(cmd)

    with open('test.txt', 'w') as f:
        for l in exp_cmd_list:
            f.write("%s\n" % l)