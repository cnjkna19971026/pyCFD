import time

def timeit(func , *args, **kwargs):

    start   = time.perf_counter()
    result  = func(*args,**kwargs)
    end     = time.perf_counter()

    elapsed = end - start
    print(f"[timeit] {func.__name__}() execute time : {elapsed:.6f} second")

    return result , elapsed
