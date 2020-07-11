#!/bin/bash

# 100 is the walk length aka iterations, 2 is the hop size
python Gen_RW_Opt_hop.py Celegans 100 2
python par2vec.py Celegans
python main.py -data Celegans -e 200 -tc 200 -w 1e-6 -l 1e-2 -regterm 1e-6
