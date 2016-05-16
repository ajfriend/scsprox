.PHONY: test clean install

all: test

# can run -vs, where s makes it not capture output
# the -l flag will print out a list of local variables with their corresponding values when a test fails
test:
	py.test scsprox -vs

clean:
	-pip uninstall scsprox
	-rm -rf build/ dist/ scsprox.egg-info/
	#-find . -name "*.cache" -exec rm -rf {} \;
	#-find . -name "__pycache__" -exec rm -rf {} \;
	-rm -rf __pycache__ scsprox/__pycache__ scsprox/tests/__pycache__
	-rm -rf .ipynb_checkpoints/

install:
	python setup.py install