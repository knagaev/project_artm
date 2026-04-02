def test():
    """Stupid test function"""
    L = [i for i in range(1000)]

if __name__ == '__main__':
    import timeit
    print(timeit.timeit("test()", setup="from __main__ import test"))
