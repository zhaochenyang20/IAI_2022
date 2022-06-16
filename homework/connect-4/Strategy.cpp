
#include "Strategy.h"

#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

#include "Judge.h"
#include "Point.h"
#include "enhencement.h"
using namespace std;

/*
	策略函数接口,该函数被对抗平台调用,每次传入当前状态,要求输出你的落子点,该落子点必须是一个符合游戏规则的落子点,不然对抗平台会直接认为你的程序有误
	input:
		为了防止对对抗平台维护的数据造成更改，所有传入的参数均为const属性
		M, N : 棋盘大小 M - 行数 N - 列数 均从 0开始计， 左上角为坐标原点，行用 x 标记，列用 y 标记
*/
//!		top : 当前棋盘每一列列顶的实际位置. e.g. 第i列为空,则 top[i] == M, 第i列已满,则 top[i] == 0
//* 	这实际上取决于棋盘是从左上角开始记录 x，y 坐标，然而下棋是从底部到头顶
/*
		_board : 棋盘的一维数组表示, 为了方便使用，在该函数刚开始处，我们已经将其转化为了二维数组board
				你只需直接使用board即可，左上角为坐标原点，数组从[0][0]开始计(不是[1][1])
				board[x][y]表示第x行、第y列的点(从0开始计)
				board[x][y] == 0/1/2 分别对应(x,y)处 无落子/有用户的子/有程序的子,不可落子点处的值也为 0
		lastX, lastY : 对方上一次落子的位置
		noX, noY : 棋盘上的不可落子点(注:框架已经处理过不可落子点，如果某一步所落的子的上面恰是不可落子点，那么框架已经将该列的 top 值又减了一次 1，
				所以在你的代码中也可以根本不使用noX和noY这两个参数，完全认为top数组就是当前每列的顶部即可
		以上参数实际上包含了当前状态(M N _top _board)以及历史信息(lastX lastY),你要做的就是在这些信息下给出尽可能明智的落子点
	output:
		你的落子点 Point
*/

extern "C" Point* getPoint(const int M, const int N, const int* top,
                           const int* _board, const int lastX, const int lastY,
                           const int ban_x, const int ban_y) {
  startTime = clock();

  /*
          不要更改这段代码
  */
  int x = -1, y = -1;
  int** board = new int*[M];
  for (int i = 0; i < M; i++) {
    board[i] = new int[N];
    for (int j = 0; j < N; j++) {
      board[i][j] = _board[i * N + j];
    }
  }

  /*
          根据你自己的策略来返回落子点,也就是根据你的策略完成对x,y的赋值
          该部分对参数使用没有限制，为了方便实现，你可以定义自己新的类、.h文件、.cpp文件
  */
  // Add your own code below

  UCT* tree = new UCT(N, M, ban_x, ban_y, board, top, true);

  bool mark = false;
  for (int i = 0; i < N; i++) {
    if (top[i] > 0) {
      board[top[i] - 1][i] = 2;
      if (machineWin(top[i] - 1, i, M, N, board)) {
        mark = true;
        board[top[i] - 1][i] = 0;
        x = top[i] - 1;
        y = i;
        break;
      }
      board[top[i] - 1][i] = 0;
    }
  }
  if (!mark) {
    for (int i = 0; i < N; i++) {
      if (top[i] > 0) {
        board[top[i] - 1][i] = 1;
        if (userWin(top[i] - 1, i, M, N, board)) {
          mark = true;
          board[top[i] - 1][i] = 0;
          x = top[i] - 1;
          y = i;
          break;
        }
        board[top[i] - 1][i] = 0;
      }
    }
    if (!mark) {
      Node* finalNode = tree->search();
      y = finalNode->get_position_y();
      x = finalNode->get_position_x();
      delete tree;
    }
  }

  /*
          不要更改这段代码
  */
  clearArray(M, N, board);
  return new Point(x, y);
}

/*
        getPoint函数返回的Point指针是在本so模块中声明的，为避免产生堆错误，应在外部调用本so中的
        函数来释放空间，而不应该在外部直接delete
*/
extern "C" void clearPoint(Point* p) {
  delete p;
  return;
}

/*
        清除top和board数组
*/
void clearArray(int M, int N, int** board) {
  for (int i = 0; i < M; i++) {
    delete[] board[i];
  }
  delete[] board;
}

/*
        添加你自己的辅助函数，你可以声明自己的类、函数，添加新的.h
   .cpp文件来辅助实现你的想法
*/
