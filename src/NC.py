import subprocess
import sys
import os

accuracies = {"citeseer": 0.0, "cora": 0.0, "pubmed": 0.0}

command = "python main_nc.py -data citeseer -e 3000 -tc 3000 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 3327"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1]
	accuracies["citeseer"] = float(output)

command = "python main_nc.py -data cora -e 3000 -tc 3000 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 2708"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1]
	accuracies["cora"] = float(output)

command = "python main_nc.py -data pubmed -e 200 -tc 200 -w 1e-6 -l 1e-3 -regterm 1e-6 -numnodes 19717"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
for line in process.stdout:
	output = line.decode("utf-8")[:-1]
	accuracies["pubmed"] = float(output)

print(accuracies)

if not os.path.exists("result_logs"):
	os.mkdir("result_logs")
	if not os.path.exists("result_logs/NC"):
		os.mkdir("result_logs/NC")

if not os.path.exists("result_logs/NC/results.txt"):
	with open("result_logs/NC/results.txt", "w") as f:
		f.write("Dataset Name" + "\t\t\t" + "Classification Accuracy" + '\n')
		for dataset, acc in accuracies.items():
			f.write(dataset + "\t\t\t\t\t" + str(acc) + '\n')

else:
	with open("result_logs/NC/results.txt", "a+") as f:
		for dataset, acc in accuracies.items():
			f.write(dataset + "\t\t\t\t\t" + str(acc) + '\n')