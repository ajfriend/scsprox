"""
This script runs the proxer many times and queries the
process memory to see if memory is leaking at all.
"""

import os
import psutil
from proximal import Prox
from proximal.examples import example_rand

def get_mem_MB():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss/1e6

print("Before example data: ", get_mem_MB())
prob, x_vars, true_sol = example_rand(100, 50)


print("Before Prox obj: ", get_mem_MB())
prox = Prox(prob, x_vars, verbose=False)

print("Before prox.prox: ", get_mem_MB())
x0 = prox.do(verbose=False)

print("Before loop: ", get_mem_MB())

steps_to_check = 3
for i in range(steps_to_check*10):
    x0 = prox.do(x0, verbose=False, max_iters=100)
    if i % steps_to_check == 0:
        print("i: ", get_mem_MB())
        print(prox.info)


# todo: check the runtimes on big problems. py vs C
# periodically within this run
        