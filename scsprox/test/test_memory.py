"""
This script runs the proxer many times and queries the
process memory to see if memory is leaking.
"""

import os
import psutil

import numpy as np

from scsprox import Prox
from scsprox.examples import example_rand

# only run the memory test (which is slow), if this flag is on
TEST_ON = False


def get_mem_MB():
    """ Get the memory in MB used by the current python process.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss/float(2**20)

if TEST_ON:
    def test_memory():
        prob, x_vars, _ = example_rand(100, 50)
        prox = Prox(prob, x_vars, verbose=False, max_iters=20, eps=1e-7)
        x0 = prox()


        num_checks = 100
        check_iters = 100

        mem = np.zeros(num_checks)

        # a few iterations allows the memory variation to settle down
        for _ in range(check_iters*3):
            x0 = prox(x0)

        # see if memory grows with iterations
        for i in range(num_checks*check_iters):
            x0 = prox(x0)

            if (i+1) % check_iters == 0:
                m = get_mem_MB()
                mem[i//check_iters] = m

        assert np.var(mem)/np.mean(mem) <= 1e-6
