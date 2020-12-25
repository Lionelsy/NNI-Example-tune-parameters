# -*- coding: utf-8 -*-
# @Time    : 2020/12/25
# @Author  : Shuyu ZHANG
# @FileName: main.py
# @Software: PyCharm
# @Description: A template for NNI AutoML


import argparse
import logging
import nni
from nni.utils import merge_parameter

logger = logging.getLogger('Main')
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

def train(args):
    pass

def val(args):
    return 0


def test(args):
    return 0


def main(args):
    train(args)

    intermediate_result = val(args)

    # report intermediate result
    nni.report_intermediate_result(intermediate_result)

    test_result = test(args)

    logger.debug('Final result is %g', test_result)
    # report final result
    nni.report_final_result(test_result)



def get_params():
    # Parameters Setting
    parser = argparse.ArgumentParser(description='Example')

    parser.add_argument("--demo", type=str,
                        default='demo', help="Demo")
                        
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise