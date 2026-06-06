HEADERS=probability.h problem_def.h bruteforce.h knapsack.h
CPP=probability.cpp bruteforce.cpp knapsack.cpp main.cpp

third_party:
	mkdir -p third_party/

	wget https://github.com/p-ranav/argparse/archive/refs/tags/v3.2.tar.gz --directory-prefix third_party/
	tar -xvzf third_party/v3.2.tar.gz --directory third_party/
	rm third_party/v3.2.tar.gz


build: third_party
	rm *.gch || true
	g++ -O3 -I third_party/argparse-3.2/include/ $(HEADERS) $(CPP) -o gerrymandering

