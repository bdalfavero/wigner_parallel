cc=mpicxx
cflags=-g
include_flags=-I/Users/benjamindalfavero/include/eigen-3.4.0/

sources=$(wildcard src/*.cpp)
objects=$(patsubst %.cpp,%.o,$(sources))

all: wigner

wigner: $(objects)
	$(cc) $^ -o $@

src/%.o: src/%.cpp
	$(cc) $(cflags) -c $< -o $@ $(include_flags)