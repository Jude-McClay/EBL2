import numpy as np
from multiprocessing import Pool, cpu_count
import time

a = 4

def my_task(x):
    # result = sum(range(1000000))
    # time.sleep(1)
    print("hey")
    array = np.zeros(3)
    array[0] = x * a
    array[1] = 1
    return array

if __name__ == "__main__":
    print("hi")
    num_processes = cpu_count()
    pool = Pool(processes= 3)
    
    inputs = [1, 2, 3]
    # Record the start time
    start_time = time.time()
    results = pool.map(my_task, inputs)
    
    pool.close()
    pool.join()
        # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print("Elapsed time:", elapsed_time, "seconds")
    
    print(results)

