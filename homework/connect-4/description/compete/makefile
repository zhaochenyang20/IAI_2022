objects = Compete
all:
	g++ -std=c++11 *.cpp -o Compete -ldl -lpthread -g -fnon-call-exceptions -Wall

debug:
	g++ -std=c++11 *.cpp -o Compete -ldl -lpthread -g -fnon-call-exceptions -Wall -DDEBUG 

clean:
	rm -f $(objects)