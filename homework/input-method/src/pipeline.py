import os


def pipeline():
	print("refactor start")
	os.system("python3 refactor_data.py > refactor_log.txt")
	print("train start")
	os.system("python3 train.py -s Large > training_log.txt")
	print("judge start")
	os.system("./complete.sh > complete_log.txt")


if __name__ == "__main__":
	pipeline()

