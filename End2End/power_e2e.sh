#!/bin/bash

# 100 is the walk length aka iterations, 2 is the hop size
python Gen_RW_Opt_hop.py Power 100 3
python par2vec.py Power
python main.py -data Power -e 200 -tc 200 -w 1e-8 -l 1e-1 -regterm 1e-8 -it True
