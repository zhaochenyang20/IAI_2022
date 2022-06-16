#ifndef COMPETE_H_
#define COMPETE_H_

#include "Data.h"
#include "Point.h"

#define MAX_TIME_SECOND 3

extern unsigned long timeA;
extern unsigned long timeB;

int compete(char strategyA[], char strategyB[], bool Afirst, Data* data);

#endif