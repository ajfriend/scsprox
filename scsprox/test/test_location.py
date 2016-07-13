from scsprox.examples import print_file
import os


def test1():
    # show the location of the file in the package being run
    print_file()

    # show the location of this test file
    print('Running test file from:', os.path.abspath(__file__))
