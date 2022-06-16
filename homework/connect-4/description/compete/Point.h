/********************************************************
 *	Point.h : 棋盘点类                                  *
 *	张永锋                                              *
 *	zhangyf07@gmail.com                                 *
 *	2010.8                                              *
 *********************************************************/

#ifndef POINT_H_
#define POINT_H_

class Point{
public:
	int x;
	int y;
    
	Point(int x, int y){
		this->x = x;
		this->y = y;
	}
};

#endif
