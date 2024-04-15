cc=mpicxx
cflags=-g
include_flags=-I/mnt/home/dalfaver/eigen-3.4.0/
link_flags=-lhdf5

sources=$(wildcard src/*.cpp)
objects=$(patsubst %.cpp,%.o,$(sources))

all: wigner

wigner: $(objects)
	$(cc) $(link_flags) $^ -o $@

src/%.o: src/%.cpp
	$(cc) $(cflags) -c $< -o $@ $(include_flags)

clean:
	rm wigner src/*.o *.png *.csv *.h5
