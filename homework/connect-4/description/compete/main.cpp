#include <iostream>
#include <string.h>
#include <fstream>
#include "Compete.h"

using namespace std;

unsigned long timeA;
unsigned long timeB;

void determineResult(int res, int &aWin, int &bWin, int &tie)
{
	switch (res)
	{
	case 0:
		tie++; //平局
		cout << "A and B tied" << endl;
		break;
	case 1:
		aWin++; //A赢
		cout << "A - won" << endl;
		break;
	case 2:
		bWin++; //B赢
		cout << "B - won" << endl;
		break;
	case 3:
		bWin++; //A出错,算B赢
		cout << "A - bug occurred" << endl;
		break;
	case 4:
		bWin++; //A给出非法落子,算B赢
		cout << "A - made illegal step" << endl;
		break;
	case 5:
		aWin++; //B出错,算A赢
		cout << "B - bug occurred" << endl;
		break;
	case 6:
		aWin++; //B给出非法落子,算A赢
		cout << "B - made illegal step" << endl;
		break;
	case 7:
		bWin++; //A超时,算B赢
		cout << "A - timed out" << endl;
		break;
	case 8:
		aWin++; //B超时,算A赢
		cout << "B - timed out" << endl;
		break;
	case -1:
		cout << "Load File A Error" << endl;
		break;
	case -2:
		cout << "Load File B Error" << endl;
		break;
	case -3:
		cout << "Function Interface Error in A" << endl;
		break;
	case -4:
		cout << "Function Interface Error in B" << endl;
		break;
	default:
		cout << "Platform Error" << endl;
		break;
	}
}

int main(int argc, char *argv[])
{
	if (argc != 5)
	{
		cout << "Usage:" << endl;
		cout << argv[0] << " <StrategyA.so> <StrategyB.so> <result file name> <times to compete>" << endl;
		return 0;
	}
	int res;
	ofstream out(argv[3]);

	int aWin = 0, bWin = 0, tie = 0;
	int numRounds = atoi(argv[4]);

	for (int i = 0; i < numRounds; i++)
	{
		auto seed = time(0);
		srand(seed);
		cout << "Round " << i << " seed: " << seed << " :" << endl;
		out << i << ":" << endl;

		Data *data = new Data();

		cout << "A first:" << endl;
		timeA = 0;
		timeB = 0;
		//data->reset();
		res = compete(argv[1], argv[2], true, data);
		determineResult(res, aWin, bWin, tie);
		out << res << "\t" << timeA << "\t" << timeB << endl;

		cout << "B first:" << endl;
		timeA = 0;
		timeB = 0;
		data->reset();
		res = compete(argv[1], argv[2], false, data);
		determineResult(res, aWin, bWin, tie);
		out << res << "\t" << timeA << "\t" << timeB << endl;
		out << endl;
		cout << endl;

		delete data;
	}

	double rioAWin = (1.0 * aWin) / (2.0 * numRounds);
	double rioBWin = (1.0 * bWin) / (2.0 * numRounds);
	double rioTie = (1.0 * tie) / (2.0 * numRounds);

	out << "Stat:" << endl;
	out << "ratio of A wins : " << rioAWin << endl;
	out << "ratio of B wins : " << rioBWin << endl;
	out << "ratio of Tie : " << rioTie << endl;
	out << endl;
	out << "ratio of (A wins + tie) : " << rioAWin + rioTie << endl;
	out << "ratio of (B wins + tie) : " << rioBWin + rioTie << endl;

	out.close();

	cout << "Stat:" << endl;
	cout << "ratio of A wins : " << rioAWin << endl;
	cout << "ratio of B wins : " << rioBWin << endl;
	cout << "ratio of Tie : " << rioTie << endl;
	cout << endl;
	cout << "ratio of (A wins + tie) : " << rioAWin + rioTie << endl;
	cout << "ratio of (B wins + tie) : " << rioBWin + rioTie << endl;

	return 0;
}
