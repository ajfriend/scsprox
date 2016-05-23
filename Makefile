.PHONY: test clean install

#all: test

# can run -vs, where s makes it not capture output
# the -l flag will print out a list of local variables with their corresponding values when a test fails
test:
	py.test proximal -vs

testslow:
	py.test proximal --runslow

clean:
	-pip uninstall proximal
	-rm -rf build/ dist/ proximal.egg-info/
	#-find . -name "*.cache" -exec rm -rf {} \;
	#-find . -name "__pycache__" -exec rm -rf {} \;
	-rm -rf __pycache__ proximal/__pycache__ proximal/tests/__pycache__ .cache
	-rm -rf .ipynb_checkpoints/

install:
	python setup.py install