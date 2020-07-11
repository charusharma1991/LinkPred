import subprocess
import sys
import os
auc = {"Celegans": 0.0, "USAir": 0.0, "Power": 0.0, "road": 0.0, "road-minnesota": 0.0, 
		"bio-SC-GT": 0.0, "infect-hyper": 0.0, "ppi": 0.0, "HepTh": 0.0}

command = "python main_nc.py -data citeseer -e 3000 -tc 3000 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 3327"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1]
	auc["citeseer"] = float(output)

command = "python main_nc.py -data cora -e 3000 -tc 3000 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 2708"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1]
	auc["cora"] = float(output)

command = "python main_nc.py -data pubmed -e 200 -tc 200 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 19717"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1]
	auc["pubmed"] = float(output)

command = "python main.py -data Celegans -e 200 -tc 200 -w 1e-6 -l 1e-2 -regterm 1e-6 -it True"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1].split(" ")[-1]
	auc["Celegans"] = float(output)

command = "python main.py -data USAir -e 100 -tc 100 -w 1e-6 -l 1e-2 -regterm 1e-6 -it True"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1].split(" ")[-1]
	auc["USAir"] = float(output)

command = "python main.py -data Power -e 200 -tc 200 -w 1e-8 -l 1e-1 -regterm 1e-8 -it True"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1].split(" ")[-1]
	auc["Power"] = float(output)

command = "python main.py -data road -e 100 -tc 100 -w 1e-6 -l 1e-1 -regterm 1e-6 -it True"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1].split(" ")[-1]
	auc["road"] = float(output)

command = "python main.py -data road-minnesota -e 300 -tc 300 -w 1e-8 -l 1e-1 -regterm 1e-9 -it True"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1].split(" ")[-1]
	auc["road-minnesota"] = float(output)

command = "python main.py -data bio-SC-GT -e 100 -tc 100 -w 1e-5 -l 1e-2 -regterm 1e-7 -it True"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1].split(" ")[-1]
	auc["bio-SC-GT"] = float(output)

command = "python main.py -data infect-hyper -e 200 -tc 200 -w 1e-5 -l 1e-2 -regterm 1e-6 -it True"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1].split(" ")[-1]
	auc["infect-hyper"] = float(output)

command = "python main.py -data ppi -e 100 -tc 100 -w 1e-8 -l 1e-1 -regterm 1e-7 -it True"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1].split(" ")[-1]
	auc["ppi"] = float(output)

command = "python main.py -data HepTh -e 1000 -tc 1000 -w 1e-8 -l 1e-1 -regterm 1e-6 -it True"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1].split(" ")[-1]
	auc["HepTh"] = float(output)

print(auc)

if not os.path.exists("result_logs"):
	os.mkdir("result_logs")
	if not os.path.exists("result_logs/LP"):
		os.mkdir("result_logs/LP")

if not os.path.exists("result_logs/LP/results.txt"):
	with open("result_logs/LP/results.txt", "w") as f:
		f.write("Dataset Name" + "\t\t\t" + "AUC values" + '\n')
		for dataset, acc in auc.items():
			f.write(dataset + "\t\t\t\t\t" + str(acc) + '\n')

else:
	with open("result_logs/LP/results.txt", "a+") as f:
		for dataset, acc in auc.items():
			f.write(dataset + "\t\t\t\t\t" + str(acc) + '\n')
