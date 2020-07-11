#!/bin/bash

python main_nc.py -data citeseer -e 3000 -tc 3000 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 3327

python main_nc.py -data cora -e 3000 -tc 3000 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 2708

python main_nc.py -data pubmed -e 200 -tc 200 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 19717
