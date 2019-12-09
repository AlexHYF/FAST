.DEFAULT_GOAL = target

.PHONY : target clean

%.npy : %.avi
	python3 farneback.py -c $< -o $@

target : $(patsubst %.avi, %.npy, $(shell find -name '*.avi'))

clean : 
	rm *.npy
