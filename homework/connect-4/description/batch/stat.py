import os

class time:
	def __init__(self, mine, his):
		self.mine = mine
		self.his = his

class result:
	def __init__(self):
		self.res = {	'wina'		:	0,	#self's win as offensive
						'winb'		:	0,	#self's win as deffensive
						'losea'		:	0,	#self's lose as offensive
						'loseb'		:	0,	#selfs' lose as deffensive
						'tie'		:	0,	#tie
						'bug'		:	0,	#self's bug
						'debug'		:	0,	#opponent's bug
						'illegal'	:	0,	#self's illegal point
						'deillegal'	:	0,	#opponent's illegal point
						'timeout'	:	0,	#self's timeout
						'detimeout'	:	0,	#opponent's timeout
						'loaderr'	:	0,	#self's load file err
						'deloaderr'	:	0,	#opponent's load file err
						'noentry'	:	0,	#self's find function entry err
						'denoentry'	:	0	#opponent's find function entry err
					}
		self.tieTimeList = []
	
idList = {}


def process(line, A, lineNo):
	line = line.split('\t')
	line[2] = line[2][:(len(line[2])-2)]
	
	if not A in idList:
		idList[A] = result()
	
	if line[0] == '0':
		idList[A].res['tie'] += 1
		idList[A].tieTimeList.append(time(line[1], line[2]))
	
	elif line[0] == '1':
		if lineNo == 1:
			idList[A].res['wina'] += 1
		elif lineNo == 2:
			idList[A].res['winb'] += 1
			
	elif line[0] == '2':
		if lineNo == 1:
			idList[A].res['losea'] += 1
		elif lineNo == 2:
			idList[A].res['loseb'] += 1
	
	elif line[0] == '3':
		idList[A].res['bug'] += 1
	
	elif line[0] == '4':
		idList[A].res['illegal'] += 1
	
	elif line[0] == '5':
		idList[A].res['debug'] += 1
	
	elif line[0] == '6':
		idList[A].res['deillegal'] += 1
	
	elif line[0] == '7':
		idList[A].res['timeout'] += 1
	
	elif line[0] == '8':
		idList[A].res['detimeout'] += 1
	
	elif line[0] == '-1':
		idList[A].res['loaderr'] += 1
	
	elif line[0] == '-2':
		idList[A].res['deloaderr'] += 1
	
	elif line[0] == '-3':
		idList[A].res['noentry'] += 1
	
	elif line[0] == '-4':
		idList[A].res['denoentry'] += 1
	
	return

def process_block(file, A):
	line = file.readline().strip()
	if not line.endswith(':') or not line[:-1].decode('utf-8').isnumeric():
		return False

	line = file.readline()
	if not len(line):
		line = '-3\t0\t0'
	process(line, A, 1)

	line = file.readline()
	if not len(line):
		line = '-3\t0\t0'
	process(line, A, 2)
	line = file.readline()
	return True

def main():	
	resdir = "./compete_result/"
	namelist = os.listdir(resdir)
	for name in namelist:
		A = name[0:10]
		f = open(resdir + name, 'r')
		flag = True
		while flag:
			flag = process_block(f, A)
		f.close()

	resfile = open("./stat.txt", 'w')

	resfile.write("id\twina\twinb\tlosea\tloseb\ttie\tbug\tdebug\tillegal\tdeillegal\ttimeout\tdetimeout\tloaderr\tdeloaderr\tnoentry\tdenoentry\n")
	for i in idList:
		res = idList[i].res
		line = str(i) + "\t" + str(res['wina']) + "\t" + str(res['winb']) + "\t" + str(res['losea']) + "\t" + str(res['loseb']) + "\t" + str(res['tie']) + "\t" + str(res['bug']) + "\t" + str(res['debug']) + "\t" + str(res['illegal']) + "\t" + str(res['deillegal']) + "\t" + str(res['timeout']) + "\t" + str(res['detimeout']) + "\t" + str(res['loaderr']) + "\t" + str(res['deloaderr']) + "\t" + str(res['noentry']) + "\t" + str(res['denoentry'])
		resfile.write(line + "\n")
	resfile.close()


if __name__ == '__main__':
	main()

