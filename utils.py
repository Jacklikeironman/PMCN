import torch
import numpy as np
import time

def print_info(info_list):
    for info in info_list:
        for key, value in info.items():
            print(key, ':', value)

def record_info(info_list, file_path):
    with open(file_path, 'a') as file:
        file.write('\n\n*** log_time ' + get_time_str() + ' ***\n')
        for info in info_list:
            for key, value in info.items():
                file.write(str(key) + ' : ' + str(value) + '\n')

def progress_bar(done_num, total_num, length=40):
    if total_num == 0:
        print('Error: Total number is 0.')
        return
    percent = done_num / total_num
    show = int(length * percent)
    bar = '|' + '=' * show + ' ' * (length - show) + '| (' + \
          str(int(percent * 100)) + '%) ' + str(done_num) + ' of ' + str(total_num) + ' ' * 10
    print(bar, end='\r')

def get_time_str():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())

def calculate_variables(model, print_vars=False):
    params = list(model.named_parameters())
    k = 0
    all_size = 0
    for parm in params:
        (name, value) = parm
        l = 1
        for j in value.size():
            l *= j
        if print_vars:
            print("{}：".format(name) + str(list(value.size())) + '    ' + "该层参数和：" + str(l))
        k = k + l
        if value.dtype == torch.float32:
            all_size += (4 * l)
        elif value.dtype == torch.float64:
            all_size += (8 * l)

    print("总参数数量和：" + str(k))
    print('Variables  size  : %d B' % (all_size))
    print('Variables  size  : %d b' % (all_size * 8))
    print('Variables  size  : %f KB' % (all_size / 1024))
    print('Variables  size  : %f Kb' % (all_size * 8 / 1024))
    print('Variables  size  : %f MB' % (all_size / 1024 / 1024))
    print('Variables  size  : %f Mb' % (all_size * 8 / 1024 / 1024))