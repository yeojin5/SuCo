SETTING_RELEASE=-O3 -std=c++17 -larmadillo -lmlpack -lboost_serialization -fopenmp -fpic -march=native -mavx -mavx2 -msse3
SETTING_DEBUG=-g -std=c++17 -larmadillo -lmlpack -lboost_serialization -fopenmp -fpic -march=native -mavx -mavx2 -msse3


SOURCE=$(wildcard ./src/*.cpp)

suco:$(SOURCE)
	rm -rf suco
	g++ $(filter-out ./src/ground_truth.cpp, $(SOURCE)) -o suco $(SETTING_RELEASE)

ground_truth: ./src/ground_truth.cpp ./src/preprocess.cpp ./src/dist_calculation.cpp
	rm -rf ground_truth
	g++ ./src/ground_truth.cpp ./src/preprocess.cpp ./src/dist_calculation.cpp -o ground_truth $(SETTING_RELEASE)

debug:$(SOURCE)
	rm -rf suco
	g++ $(filter-out ./src/ground_truth.cpp, $(SOURCE)) -o suco $(SETTING_DEBUG)

clean:
	rm -rf suco ground_truth
