import contextlib

@contextlib.contextmanager

def config_test():
    print('start')
    try:
        ## 전처리
        yield
        ## 후처리 
    finally:
        print('done')

with config_test():
    print('process...')