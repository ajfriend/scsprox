.PHONY: test clean install

all: test

# can run -vs, where s makes it not capture output
# the -l flag will print out a list of local variables with their corresponding values when a test fails
test:
	py.test scs_prox -vs

clean:
	-pip uninstall scs_prox
	-rm -rf build/ dist/ scs_prox.egg-info/
	#-find . -name "*.cache" -exec rm -rf {} \;
	#-find . -name "__pycache__" -exec rm -rf {} \;
	-rm -rf __pycache__ scs_prox/__pycache__ scs_prox/tests/__pycache__
	-rm -rf .ipynb_checkpoints/

install:
	python setup.py install