CC=g++
#FLAGS=-g -Wall --std=c++20 -march=native
FLAGS=-O3 -Wall --std=c++20 -march=native -pthread

SRC=src
BIN=bin

GURINC = -I /opt/gurobi1100/linux64/include/
GURARGS = -std=c++20 -m64
GURLIB = -L/opt/gurobi1100/linux64/lib -lgurobi_c++ -lgurobi110

#openmp stuff. Not currently used. Using std::thread instead.
#on my Mac use:
#OMPCOM = -I/opt/homebrew/opt/libomp/include -Xclang -fopenmp
#OMPLIN = -L/opt/homebrew/opt/libomp/lib -Xclang -fopenmp -lomp
#on the linux cluster use:
#OMPCOM = -fopenmp
#OMPLIN = -fopenmp -lomp

#SJ: on my mac this is what I need to make chebsilon compile and link
GMP_HDR_PATH=/opt/homebrew/Cellar/gmp/6.3.0/include
GMP_LIB_PATH=/opt/homebrew/Cellar/gmp/6.3.0/lib

all: sawxor brute load_decode epsilon chebsilon evalxor gaussheur random_regular random_tanner lrandreg ldgmsat erdos_renyi evalist salist feldman_lp greedy degrees bigpanneal rsinstance randgauss partial_anneal full_anneal measure_bias partial_anneal_mp adversarize baby_irreg advrand advrand_mp wscan cperpscan

check: build
	$(CC) $(FLAGS) $(BIN)/test_xortools.o $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/pmatrix.o $(BIN)/listsat.o -o $(BIN)/test_xortools
	$(BIN)/test_xortools

distcheck:
	@echo "Not implemented yet"

build:  $(BIN)/test_xortools.o $(BIN)/bitmatrix.o $(BIN)/pmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/sawxor.o $(BIN)/brute.o $(BIN)/ldpc.o $(BIN)/load_decode.o $(BIN)/evalxor.o $(BIN)/gaussheur.o $(BIN)/random_regular.o $(BIN)/random_tanner.o $(BIN)/ldgmsat.o $(BIN)/listsat.o $(BIN)/evalist.o $(BIN)/salist.o $(BIN)/lrandreg.o
	@echo "Build Complete"

feldman_gurobi:	$(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/feldman_gurobi.o
	$(CC) $(GURARGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/feldman_gurobi.o -o $(BIN)/feldman_gurobi $(GURLIB) -lm

polytopeR:	$(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/polytopeR.o
	$(CC) $(GURARGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/polytopeR.o -o $(BIN)/polytopeR $(GURLIB) -lm

load_decode_lp:	$(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/load_decode_lp.o
	$(CC) $(GURARGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/load_decode_lp.o -o $(BIN)/load_decode_lp $(GURLIB) -lm

load_decode_R:	$(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/load_decode_R.o
	$(CC) $(GURARGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/load_decode_R.o -o $(BIN)/load_decode_R $(GURLIB) -lm

load_decode_ip:	$(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/load_decode_ip.o
	$(CC) $(GURARGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/load_decode_ip.o -o $(BIN)/load_decode_ip $(GURLIB) -lm

tanatmis:	$(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/tanatmis.o
	$(CC) $(GURARGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/tanatmis.o -o $(BIN)/tanatmis $(GURLIB) -lm

wscan: $(BIN)/wscan.o $(BIN)/utils.o
	$(CC) -Wall -O3 --std=c++20 $(BIN)/wscan.o $(BIN)/utils.o -o $(BIN)/wscan

cperpscan: $(BIN)/cperpscan.o $(BIN)/utils.o $(BIN)/bitmatrix.o $(BIN)/ldpc.o $(BIN)/xorsat.o
	$(CC) -Wall -O3 --std=c++20 $(BIN)/cperpscan.o $(BIN)/utils.o $(BIN)/bitmatrix.o $(BIN)/ldpc.o $(BIN)/xorsat.o -o $(BIN)/cperpscan

sawxor:  $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/sawxor.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/sawxor.o -o $(BIN)/sawxor

partial_anneal_mp:  $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/partial_anneal_mp.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/partial_anneal_mp.o -o $(BIN)/partial_anneal_mp

partial_anneal:  $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/partial_anneal.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/partial_anneal.o -o $(BIN)/partial_anneal

full_anneal:  $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/full_anneal.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/full_anneal.o -o $(BIN)/full_anneal

advrand:  $(BIN)/utils.o $(BIN)/advrand.o
	$(CC) $(FLAGS) $(BIN)/utils.o $(BIN)/advrand.o -o $(BIN)/advrand

advrand_mp:  $(BIN)/utils.o $(BIN)/advrand_mp.o
	$(CC) $(FLAGS) $(BIN)/utils.o $(BIN)/advrand_mp.o -o $(BIN)/advrand_mp

measure_bias:  $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/measure_bias.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/measure_bias.o -o $(BIN)/measure_bias

adversarize:  $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/adversarize.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/adversarize.o -o $(BIN)/adversarize

baby_irreg:  $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/baby_irreg.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xoropt.o $(BIN)/baby_irreg.o -o $(BIN)/baby_irreg

bigpanneal:  $(BIN)/utils.o $(BIN)/bigpanneal.o
	$(CC) $(FLAGS) $(BIN)/utils.o $(BIN)/bigpanneal.o -o $(BIN)/bigpanneal

rsinstance:  $(BIN)/utils.o $(BIN)/rsinstance.o
	$(CC) $(FLAGS) $(BIN)/utils.o $(BIN)/rsinstance.o -o $(BIN)/rsinstance

degrees:  $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/degrees.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/degrees.o -o $(BIN)/degrees

greedy:  $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/greedy.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/greedy.o -o $(BIN)/greedy

brute:  $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/brute.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/brute.o -o $(BIN)/brute

feldman_lp:  $(BIN)/bitmatrix.o $(BIN)/feldman_lp.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/feldman_lp.o $(BIN)/xorsat.o $(BIN)/ldpc.o -o $(BIN)/feldman_lp

salist:  $(BIN)/pmatrix.o $(BIN)/utils.o $(BIN)/listsat.o $(BIN)/salist.o
	$(CC) $(FLAGS) $(BIN)/pmatrix.o $(BIN)/utils.o $(BIN)/listsat.o $(BIN)/salist.o -o $(BIN)/salist

load_decode:  $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/load_decode.o
	$(CC) $(FLAGS) $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/ldpc.o $(BIN)/load_decode.o -o $(BIN)/load_decode

erdos_renyi:  $(BIN)/utils.o $(BIN)/bitmatrix.o $(BIN)/ldpc.o $(BIN)/xorsat.o $(BIN)/erdos_renyi.o
	$(CC) $(FLAGS) $(BIN)/utils.o $(BIN)/bitmatrix.o $(BIN)/ldpc.o $(BIN)/xorsat.o $(BIN)/erdos_renyi.o -o $(BIN)/erdos_renyi

epsilon:  $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/bitmatrix.o $(BIN)/epsilon.o
	$(CC) $(FLAGS) $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/bitmatrix.o $(BIN)/epsilon.o -o $(BIN)/epsilon

chebsilon:  $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/bitmatrix.o $(BIN)/chebsilon.o
	$(CC) $(FLAGS) -L $(GMP_LIB_PATH) $(BIN)/utils.o $(BIN)/xorsat.o $(BIN)/bitmatrix.o $(BIN)/chebsilon.o -o $(BIN)/chebsilon -lgmp

evalxor:  $(BIN)/xorsat.o $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/evalxor.o
	$(CC) $(FLAGS) $(BIN)/xorsat.o $(BIN)/bitmatrix.o $(BIN)/utils.o $(BIN)/evalxor.o -o $(BIN)/evalxor

evalist:  $(BIN)/listsat.o $(BIN)/pmatrix.o $(BIN)/utils.o $(BIN)/evalist.o
	$(CC) $(FLAGS) $(BIN)/listsat.o $(BIN)/pmatrix.o $(BIN)/utils.o $(BIN)/evalist.o -o $(BIN)/evalist

gaussheur:  $(BIN)/xorsat.o $(BIN)/bitmatrix.o $(BIN)/ldpc.o $(BIN)/utils.o $(BIN)/gaussheur.o
	$(CC) $(FLAGS) $(BIN)/xorsat.o $(BIN)/bitmatrix.o $(BIN)/ldpc.o $(BIN)/utils.o $(BIN)/gaussheur.o -o $(BIN)/gaussheur

randgauss:  $(BIN)/xorsat.o $(BIN)/bitmatrix.o $(BIN)/ldpc.o $(BIN)/utils.o $(BIN)/randgauss.o
	$(CC) $(FLAGS) $(BIN)/xorsat.o $(BIN)/bitmatrix.o $(BIN)/ldpc.o $(BIN)/utils.o $(BIN)/randgauss.o -o $(BIN)/randgauss

random_regular:  $(BIN)/random_regular.o $(BIN)/utils.o
	$(CC) $(FLAGS) $(BIN)/utils.o $(BIN)/random_regular.o -o $(BIN)/random_regular

random_tanner:  $(BIN)/random_tanner.o $(BIN)/utils.o
	$(CC) $(FLAGS) $(BIN)/utils.o $(BIN)/random_tanner.o -o $(BIN)/random_tanner

lrandreg:  $(BIN)/lrandreg.o $(BIN)/utils.o
	$(CC) $(FLAGS) $(BIN)/utils.o $(BIN)/lrandreg.o -o $(BIN)/lrandreg

youngpoly: $(SRC)/youngpoly.cpp $(SRC)/polytools.hpp
	$(CC) -Wall -O3 --std=c++11 $(SRC)/youngpoly.cpp -o $(BIN)/youngpoly -lntl -lgmp -pthread

polyscratch: $(SRC)/polyscratch.cpp $(SRC)/polytools.hpp
	$(CC) -Wall -O3 --std=c++11 $(SRC)/polyscratch.cpp -o $(BIN)/polyscratch -lntl -lgmp -pthread

ansatz_test: $(SRC)/ansatz_test.cpp $(SRC)/polytools.hpp
	$(CC) -Wall -O3 --std=c++11 $(SRC)/ansatz_test.cpp -o $(BIN)/ansatz_test -lntl -lgmp -pthread

polyvec: $(SRC)/polyvec.cpp $(SRC)/polytools.hpp
	$(CC) -Wall -O3 --std=c++11 $(SRC)/polyvec.cpp -o $(BIN)/polyvec -lntl -lgmp -pthread

ldgmsat:  $(BIN)/utils.o $(BIN)/bitmatrix.o $(BIN)/ldpc.o $(BIN)/ldgmsat.o $(BIN)/xorsat.o
	$(CC) $(FLAGS) $(BIN)/utils.o $(BIN)/bitmatrix.o $(BIN)/ldpc.o $(BIN)/ldgmsat.o $(BIN)/xorsat.o -o $(BIN)/ldgmsat

$(BIN)/bitmatrix.o: $(SRC)/bitmatrix.cpp $(SRC)/bitmatrix.hpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/bitmatrix.cpp -o $(BIN)/bitmatrix.o

$(BIN)/test_xortools.o: $(SRC)/test_xortools.cpp $(SRC)/bitmatrix.hpp $(SRC)/utils.hpp $(SRC)/ldpc.hpp $(SRC)/pmatrix.hpp
	$(CC) $(FLAGS) -c $(SRC)/test_xortools.cpp -o $(BIN)/test_xortools.o

$(BIN)/xorsat.o: $(SRC)/xorsat.cpp $(SRC)/xorsat.hpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/xorsat.cpp -o $(BIN)/xorsat.o

$(BIN)/utils.o: $(SRC)/utils.cpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/utils.cpp -o $(BIN)/utils.o

$(BIN)/sawxor.o: $(SRC)/sawxor.cpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp
	$(CC) $(FLAGS) -c $(SRC)/sawxor.cpp -o $(BIN)/sawxor.o

$(BIN)/xoropt.o: $(SRC)/xoropt.cpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp
	$(CC) $(FLAGS) -c $(SRC)/xoropt.cpp -o $(BIN)/xoropt.o

$(BIN)/partial_anneal_mp.o: $(SRC)/partial_anneal_mp.cpp $(SRC)/xoropt.hpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp
	$(CC) $(FLAGS) -c $(SRC)/partial_anneal_mp.cpp -o $(BIN)/partial_anneal_mp.o

$(BIN)/partial_anneal.o: $(SRC)/partial_anneal.cpp $(SRC)/xoropt.hpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp
	$(CC) $(FLAGS) -c $(SRC)/partial_anneal.cpp -o $(BIN)/partial_anneal.o

$(BIN)/full_anneal.o: $(SRC)/full_anneal.cpp $(SRC)/xoropt.hpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp
	$(CC) $(FLAGS) -c $(SRC)/full_anneal.cpp -o $(BIN)/full_anneal.o

$(BIN)/advrand.o: $(SRC)/advrand.cpp $(SRC)/utils.hpp $(SRC)/advrand.hpp
	$(CC) $(FLAGS) -c $(SRC)/advrand.cpp -o $(BIN)/advrand.o

$(BIN)/advrand_mp.o: $(SRC)/advrand_mp.cpp $(SRC)/utils.hpp $(SRC)/advrand.hpp
	$(CC) $(FLAGS) -c $(SRC)/advrand_mp.cpp -o $(BIN)/advrand_mp.o

$(BIN)/measure_bias.o: $(SRC)/measure_bias.cpp $(SRC)/xoropt.hpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp
	$(CC) $(FLAGS) -c $(SRC)/measure_bias.cpp -o $(BIN)/measure_bias.o

$(BIN)/adversarize.o: $(SRC)/adversarize.cpp $(SRC)/xoropt.hpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp
	$(CC) $(FLAGS) -c $(SRC)/adversarize.cpp -o $(BIN)/adversarize.o

$(BIN)/baby_irreg.o: $(SRC)/baby_irreg.cpp $(SRC)/xoropt.hpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp
	$(CC) $(FLAGS) -c $(SRC)/baby_irreg.cpp -o $(BIN)/baby_irreg.o

$(BIN)/bigpanneal.o: $(SRC)/bigpanneal.cpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/bigpanneal.cpp -o $(BIN)/bigpanneal.o

$(BIN)/rsinstance.o: $(SRC)/rsinstance.cpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/rsinstance.cpp -o $(BIN)/rsinstance.o

$(BIN)/degrees.o: $(SRC)/degrees.cpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp
	$(CC) $(FLAGS) -c $(SRC)/degrees.cpp -o $(BIN)/degrees.o

$(BIN)/greedy.o: $(SRC)/greedy.cpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp
	$(CC) $(FLAGS) -c $(SRC)/greedy.cpp -o $(BIN)/greedy.o

$(BIN)/brute.o: $(SRC)/brute.cpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp
	$(CC) $(FLAGS) -c $(SRC)/brute.cpp -o $(BIN)/brute.o

$(BIN)/feldman_lp.o: $(SRC)/feldman_lp.cpp $(SRC)/bitmatrix.hpp
	$(CC) $(FLAGS) -c $(SRC)/feldman_lp.cpp -o $(BIN)/feldman_lp.o

$(BIN)/wscan.o: $(SRC)/wscan.cpp $(SRC)/utils.hpp
	$(CC) -Wall -O3 --std=c++20 -c $(SRC)/wscan.cpp -o $(BIN)/wscan.o

$(BIN)/cperpscan.o: $(SRC)/cperpscan.cpp $(SRC)/utils.hpp $(SRC)/ldpc.hpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp
	$(CC) -Wall -O3 --std=c++20 -c $(SRC)/cperpscan.cpp -o $(BIN)/cperpscan.o

$(BIN)/salist.o: $(SRC)/salist.cpp $(SRC)/utils.hpp $(SRC)/pmatrix.hpp $(SRC)/listsat.hpp $(SRC)/panneal.hpp
	$(CC) $(FLAGS) -c $(SRC)/salist.cpp -o $(BIN)/salist.o

$(BIN)/load_decode.o: $(SRC)/load_decode.cpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp $(SRC)/ldpc.hpp
	$(CC) $(FLAGS) -c $(SRC)/load_decode.cpp -o $(BIN)/load_decode.o

$(BIN)/erdos_renyi.o: $(SRC)/erdos_renyi.cpp $(SRC)/bitmatrix.hpp $(SRC)/ldpc.hpp $(SRC)/xorsat.hpp
	$(CC) $(FLAGS) -c $(SRC)/erdos_renyi.cpp -o $(BIN)/erdos_renyi.o

$(BIN)/epsilon.o: $(SRC)/epsilon.cpp $(SRC)/utils.hpp $(SRC)/xorsat.hpp
	$(CC) $(FLAGS) -c $(SRC)/epsilon.cpp -o $(BIN)/epsilon.o

$(BIN)/chebsilon.o: $(SRC)/chebsilon.cpp $(SRC)/utils.hpp $(SRC)/xorsat.hpp
	$(CC) $(FLAGS) -I $(GMP_HDR_PATH) -c $(SRC)/chebsilon.cpp -o $(BIN)/chebsilon.o

$(BIN)/evalxor.o: $(SRC)/evalxor.cpp $(SRC)/xorsat.hpp $(SRC)/bitmatrix.hpp
	$(CC) $(FLAGS) -c $(SRC)/evalxor.cpp -o $(BIN)/evalxor.o

$(BIN)/evalist.o: $(SRC)/evalist.cpp $(SRC)/listsat.hpp $(SRC)/pmatrix.hpp
	$(CC) $(FLAGS) -c $(SRC)/evalist.cpp -o $(BIN)/evalist.o

$(BIN)/gaussheur.o: $(SRC)/gaussheur.cpp $(SRC)/xorsat.hpp $(SRC)/bitmatrix.hpp $(SRC)/ldpc.hpp
	$(CC) $(FLAGS) -c $(SRC)/gaussheur.cpp -o $(BIN)/gaussheur.o

$(BIN)/randgauss.o: $(SRC)/randgauss.cpp $(SRC)/xorsat.hpp $(SRC)/bitmatrix.hpp $(SRC)/ldpc.hpp
	$(CC) $(FLAGS) -c $(SRC)/randgauss.cpp -o $(BIN)/randgauss.o

$(BIN)/random_regular.o: $(SRC)/random_regular.cpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/random_regular.cpp -o $(BIN)/random_regular.o

$(BIN)/random_tanner.o: $(SRC)/random_tanner.cpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/random_tanner.cpp -o $(BIN)/random_tanner.o

$(BIN)/lrandreg.o: $(SRC)/lrandreg.cpp $(SRC)/randreg.hpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/lrandreg.cpp -o $(BIN)/lrandreg.o

$(BIN)/ldpc.o: $(SRC)/ldpc.cpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp $(SRC)/ldpc.hpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/ldpc.cpp -o $(BIN)/ldpc.o

$(BIN)/ldgmsat.o: $(SRC)/ldgmsat.cpp $(SRC)/bitmatrix.hpp $(SRC)/ldpc.hpp $(SRC)/xorsat.hpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/ldgmsat.cpp -o $(BIN)/ldgmsat.o

$(BIN)/pmatrix.o: $(SRC)/pmatrix.cpp $(SRC)/pmatrix.hpp $(SRC)/modpinverse.hpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/pmatrix.cpp -o $(BIN)/pmatrix.o

$(BIN)/listsat.o: $(SRC)/listsat.cpp $(SRC)/listsat.hpp $(SRC)/pmatrix.hpp $(SRC)/modpinverse.hpp $(SRC)/utils.hpp
	$(CC) $(FLAGS) -c $(SRC)/listsat.cpp -o $(BIN)/listsat.o

$(BIN)/feldman_gurobi.o: $(SRC)/feldman_gurobi.cpp $(SRC)/utils.hpp $(SRC)/ldpc.hpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp
	$(CC) $(GURARGS) $(GURINC) $(GURLIB) -c $(SRC)/feldman_gurobi.cpp -o $(BIN)/feldman_gurobi.o

$(BIN)/load_decode_lp.o: $(SRC)/load_decode_lp.cpp $(SRC)/utils.hpp $(SRC)/ldpc.hpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp
	$(CC) $(GURARGS) $(GURINC) $(GURLIB) -c $(SRC)/load_decode_lp.cpp -o $(BIN)/load_decode_lp.o

$(BIN)/load_decode_R.o: $(SRC)/load_decode_R.cpp $(SRC)/utils.hpp $(SRC)/ldpc.hpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp
	$(CC) $(GURARGS) $(GURINC) $(GURLIB) -c $(SRC)/load_decode_R.cpp -o $(BIN)/load_decode_R.o

$(BIN)/load_decode_ip.o: $(SRC)/load_decode_ip.cpp $(SRC)/utils.hpp $(SRC)/ldpc.hpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp
	$(CC) $(GURARGS) $(GURINC) $(GURLIB) -c $(SRC)/load_decode_ip.cpp -o $(BIN)/load_decode_ip.o

$(BIN)/tanatmis.o: $(SRC)/tanatmis.cpp $(SRC)/utils.hpp $(SRC)/ldpc.hpp $(SRC)/bitmatrix.hpp $(SRC)/xorsat.hpp
	$(CC) $(GURARGS) $(GURINC) $(GURLIB) -c $(SRC)/tanatmis.cpp -o $(BIN)/tanatmis.o

$(BIN)/polytopeR.o: $(SRC)/polytopeR.cpp $(SRC)/utils.hpp $(SRC)/bitmatrix.hpp
	$(CC) $(GURARGS) $(GURINC) $(GURLIB) -c $(SRC)/polytopeR.cpp -o $(BIN)/polytopeR.o

clean:
	rm -f $(BIN)/*.o $(BIN)/test_xortools $(BIN)/sawxor $(BIN)/brute $(BIN)/load_decode $(BIN)/epsilon $(BIN)/evalxor $(BIN)/gaussheur $(BIN)/random_regular $(BIN)/random_tanner $(BIN)/ldgmsat $(BIN)/erdos_renyi $(BIN)/evalist $(BIN)/lrandreg $(BIN)/salist $(BIN)/youngpoly $(BIN)/polyscratch $(BIN)/ansatz_test $(BIN)/polyvec $(BIN)/feldman_lp $(BIN)/greedy $(BIN)/degrees $(BIN)/bigpanneal $(BIN)/rsinstance $(BIN)/randgauss $(BIN)/partial_anneal $(BIN)/partial_anneal_mp $(BIN)/full_anneal $(BIN)/measure_bias $(BIN)/adversarize $(BIN)/baby_irreg $(BIN)/advrand $(BIN)/advrand_mt $(BIN)/wscan $(BIN)/cperpscan
