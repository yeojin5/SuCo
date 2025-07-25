SETTING_RELEASE=-O3 -std=c++17 -larmadillo -lmlpack -lboost_serialization -fopenmp -fpic -march=native -mavx -mavx2 -msse3
SETTING_DEBUG=-g -std=c++17 -larmadillo -lmlpack -lboost_serialization -fopenmp -fpic -march=native -mavx -mavx2 -msse3


SOURCE=$(wildcard ./src/*.cpp)

suco:$(SOURCE)
	rm -rf suco
	g++ $(SOURCE) -o suco $(SETTING_RELEASE)
debug:$(SOURCE)
	rm -rf suco
	g++ $(SOURCE) -o suco $(SETTING_DEBUG)

clean:
	rm -rf suco
