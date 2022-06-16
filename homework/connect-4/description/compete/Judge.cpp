#include "Judge.h"

//board传入boardA
bool AWin(const int x, const int y, const int M, const int N, const int* board){
	//横向检测
    int i, j;
    int count = 0;
    for (i = y; i >= 0; i--)
        if (!(board[x * N + i] == 2))
            break;
    count += (y - i);
    for (i = y; i < N; i++)
        if (!(board[x * N + i] == 2))
            break;
    count += (i - y - 1);
    if (count >= 4) return true;
    
    //纵向检测
    count = 0;
    for (i = x; i < M; i++)
        if (!(board[i * N + y] == 2))
            break;
    count += (i - x);
    if (count >= 4) return true;
    
    //左下-右上
    count = 0;
    for (i = x, j = y; i < M && j >= 0; i++, j--)
        if (!(board[i * N + j] == 2))
            break;
    count += (y - j);
    for (i = x, j = y; i >= 0 && j < N; i--, j++)
        if (!(board[i * N + j] == 2))
            break;
    count += (j - y - 1);
    if (count >= 4) return true;
    
    //左上-右下
    count = 0;
    for (i = x, j = y; i >= 0 && j >= 0; i--, j--)
        if (!(board[i * N + j] == 2))
            break;
    count += (y - j);
    for (i = x, j = y; i < M && j < N; i++, j++)
        if (!(board[i * N + j] == 2))
            break;
    count += (j - y - 1);
    if (count >= 4) return true;
    
    return false;
}

//board传入boardB
bool BWin(const int x, const int y, const int M, const int N, const int* board){
	//横向检测
    int i, j;
    int count = 0;
    for (i = y; i >= 0; i--)
        if (!(board[x * N + i] == 2))
            break;
    count += (y - i);
    for (i = y; i < N; i++)
        if (!(board[x * N + i] == 2))
            break;
    count += (i - y - 1);
    if (count >= 4) return true;
    
    //纵向检测
    count = 0;
    for (i = x; i < M; i++)
        if (!(board[i * N + y] == 2))
            break;
    count += (i - x);
    if (count >= 4) return true;
    
    //左下-右上
    count = 0;
    for (i = x, j = y; i < M && j >= 0; i++, j--)
        if (!(board[i * N + j] == 2))
            break;
    count += (y - j);
    for (i = x, j = y; i >= 0 && j < N; i--, j++)
        if (!(board[i * N + j] == 2))
            break;
    count += (j - y - 1);
    if (count >= 4) return true;
    
    //左上-右下
    count = 0;
    for (i = x, j = y; i >= 0 && j >= 0; i--, j--)
        if (!(board[i * N + j] == 2))
            break;
    count += (y - j);
    for (i = x, j = y; i < M && j < N; i++, j++)
        if (!(board[i * N + j] == 2))
            break;
    count += (j - y - 1);
    if (count >= 4) return true;
    
    return false;
}

bool isTie(const int N, const int* top){
	bool tie = true;
    for (int i = 0; i < N; i++)
    {
        if (top[i] > 0)
        {
            tie = false;
            break;
        }
    }
    return tie;
}

bool isLegal(int x, int y, Data* data){
	if(x < 0 || x >= data->M || y < 0 || y >= data->N){
		return false;
	}
	if(data->top[y] != x + 1){
		return false;
	}
	//不可落子点的情况
	if(x == data->noX && y == data->noY){
		return false;
	}
	return true;
}