# -*- coding: utf-8 -*-

# --------------------------------------------------------
# File Name: util.py
#
# Written by Jiaming Qiu
# --------------------------------------------------------

import os
import inspect
import functools
import time

def log_print(in_str):
    caller_frame = inspect.currentframe().f_back
    filename, line_number, function_name, _, _ = inspect.getframeinfo(caller_frame)
    log_head = '<%s Process:%s %s %s():Line %d>' % (time.strftime(
                                                        '%Y-%m-%d %H:%M:%S', \
                                                        time.localtime(time.time()) \
                                                    ), \
                                                    os.getpid(),
                                                    filename,
                                                    function_name,
                                                    line_number)
    print('%s %s' % (log_head, in_str))

def counter(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = time.time()
        result_ = func(*args, **kw)
        end_time = time.time()
        log_print("Time cost of function %s(): %.3fs" \
                  % (func.__name__, end_time - start_time))
        return result_
    return wrapper
