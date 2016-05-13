.PHONY: test clean

all: test

# can run -vs, where s makes it not capture output
# the -l flag will print out a list of local variables with their corresponding values when a test fails
test:
	py.test scs_prox -vs

clean:
	-find . -name "*.cache" -exec rm -rf {} \;
	-find . -name "__pycache__" -exec rm -rf {} \;
	-rm -rf .ipynb_checkpoints/