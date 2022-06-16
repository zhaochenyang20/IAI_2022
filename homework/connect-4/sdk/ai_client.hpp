#pragma once

#include <time.h>
#include <string>
#include <iostream>
#include "../Point.h"
#include "../Strategy.h"

#include "json/json/json.h"

typedef unsigned int ui;

class AI_Client
{
    private:
    Point* point;
    Point* Banned_point;
    Point* Last_move;
    int row, col;
    int* board; //chess board
    int* top; //available position
    Json::Value root;
    Json::Reader reader;

    //receive messages from judger and parse them
    bool listen()
    {
        char header[4];
        scanf("%4c", header);
        ui info_len = (header[0] - '0') * 1000 + (header[1] - '0') * 100 + (header[2] - '0') * 10 + header[3] - '0';
        char* info = new char[info_len + 2];
        fgets(info, info_len + 1, stdin);
        if(!reader.parse(info, root))
        {
            fprintf(stderr, "Parse Error\nError Info:\n%s\n", info);
            return false;
        }
        return true;
    }
    //send messages to judger
    void sendLen(std::string s) {
        int len = s.length();
        unsigned char lenb[4];
        lenb[0] = (unsigned char)(len);
        lenb[1] = (unsigned char)(len >> 8);
        lenb[2] = (unsigned char)(len >> 16);
        lenb[3] = (unsigned char)(len >> 24);
        for (int i = 0; i < 4; i++)
            printf("%c", lenb[3 - i]);
    }
    void send()
    {
        Json::Value operation;
        operation.clear();
        operation["X"] = point->x;
        operation["Y"] = point->y;
        Json::FastWriter writer;
        std::string msg = writer.write(operation);
        sendLen(msg);
        std::cout << msg;
        std::cout.flush();
    }
    //parse the received information
    int parse()
    {
        //the contents in json "root" are the followings
        //"type": 0 refers to init message. 1 refers to player message. 2 refers to end info
        //"row", "col": appears only when "type" == 1
        //"nox", "noy": appears only when "type" == 1
        //"top": appears only when "type" == 2. It provides available next move.
        //"board": appears only when "type" == 2. It provides current chess board state.
        //"last": appears only when "type" == 2. It provides your opponent with the last move.
        try
        {
            int type = root["type"].asInt();
            switch (type)
            {
            case 0:
            {
                row = root["row"].asInt();
                col = root["col"].asInt();
                Banned_point->x = root["nox"].asInt();
                Banned_point->y = root["noy"].asInt();
                board = new int[row * col];
                top = new int[col];
                break;
            }
            case 1:
            {
                int top_size = root["top"].size();
                for(int i = 0; i < top_size; i ++) top[i] = root["top"][i].asInt();
                int board_size = root["board"].size();
                for(int i = 0; i < board_size; i ++) board[i] = root["board"][i].asInt();
                Last_move->x = root["lastx"].asInt();
                Last_move->y = root["lasty"].asInt();
                break;
            }
            case 2:
            {
                break;
            }
            
            default:
                fprintf(stderr, "Undefined type value %d\n", type);
                break;
            }
            return type;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        return false;
    }
    void Player_AI()
    {
        point = getPoint(row, col, top, board, Last_move->x, Last_move->y, Banned_point->x, Banned_point->y);
    }

    public:
    void run()
    {
        while(listen())
        {
            int type = parse();
            if(type == 0) continue;
            if(type == 1)
            {
                Player_AI();
                if(point == nullptr)
                {
                    point = new Point(-2, -2);
                    send();
                    delete point;
                    break;
                }
                send();
                delete point;
            }
            if(type >= 2) break;
        }
    }
    AI_Client()
    {
        point = nullptr;
        Banned_point = new Point(-1, -1);
        Last_move = new Point(-1, -1);
    }
    ~AI_Client()
    {
        if(board) delete[] board;
        if(top) delete[] top;
        if(point) delete point;
        if(Banned_point) delete Banned_point;
        if(Last_move) delete Last_move;
    }
};
