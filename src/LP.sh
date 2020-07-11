#!/bin/bash

python main_nc.py -data citeseer -e 3000 -tc 3000 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 3327

python main_nc.py -data cora -e 3000 -tc 3000 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 2708

python main_nc.py -data pubmed -e 200 -tc 200 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 19717

python main.py -data Celegans -e 200 -tc 200 -w 1e-6 -l 1e-2 -regterm 1e-6 -it True

python main.py -data USAir -e 100 -tc 100 -w 1e-6 -l 1e-2 -regterm 1e-6 -it True

python main.py -data Power -e 200 -tc 200 -w 1e-8 -l 1e-1 -regterm 1e-8 -it True

python main.py -data road -e 100 -tc 100 -w 1e-6 -l 1e-1 -regterm 1e-6 -it True

python main.py -data road-minnesota -e 300 -tc 300 -w 1e-8 -l 1e-1 -regterm 1e-9 -it True

python main.py -data bio-SC-GT -e 100 -tc 100 -w 1e-5 -l 1e-2 -regterm 1e-7 -it True

python main.py -data infect-hyper -e 200 -tc 200 -w 1e-5 -l 1e-2 -regterm 1e-6 -it True

python main.py -data ppi -e 100 -tc 100 -w 1e-8 -l 1e-1 -regterm 1e-7 -it True

python main.py -data HepTh -e 1000 -tc 1000 -w 1e-8 -l 1e-1 -regterm 1e-6 -it True

var = "$(python main.py -data facebook -e 100 -tc 100 -w 1e-8 -l 1e-1 -regterm 1e-9 -it True)"

echo $(var)
