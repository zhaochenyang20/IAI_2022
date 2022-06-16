#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

#include "Judge.h"
#include "Point.h"
using namespace std;
#pragma once

const double timelimit{1.7000 * CLOCKS_PER_SEC};

clock_t startTime;
class Node {
 private:
  double profit{0.0};                  // 当前节点的胜率
  int visit{0};                        // 总访问次数
  int ban_x, ban_y;                    // 被去除的点位
  int height;                          // 棋盘高度
  int width;                           // 棋盘宽度
  int expandableNodeNum{0};            // 可扩展节点数
  int position_x{-1}, position_y{-1};  // 落子位置
  bool expanded;                       // 是否已经扩展
  bool chance{false};                  // 是否为己方棋子

 public:
  bool& get_expanded() { return expanded; }
  const int& get_width() { return width; }
  const int& get_height() { return height; }
  const int& get_ban_x() { return ban_x; }
  const int& get_ban_y() { return ban_y; }
  int& get_visit() { return visit; }
  int& get_expandableNodeNum() { return expandableNodeNum; }
  double& get_profit() { return profit; }
  int& get_position_x() { return position_x; }
  int& get_position_y() { return position_y; }
  bool& get_chance() { return chance; }
  int** board;  // 当前局面状况
  int* top;     // 当前每一列顶部状况

  Node* parent{NULL};  // 父节点
  Node** children;     // 子节点

  int* expandable_nodes{NULL};  // 从当前节点开始可扩展节点的行号

  Node(int height, int width, int ban_x, int ban_y, int** board, int* top,
       int position_x = -1, int position_y = -1, bool chance = false,
       Node* parent = NULL)
      : height(height),
        width(width),
        ban_x(ban_x),
        ban_y(ban_y),
        position_x(position_x),
        position_y(position_y),
        chance(chance),
        parent(parent) {
    this->top = new int[width];
    for (int i = 0; i < width; ++i) {
      this->top[i] = top[i];
    }
    this->board = new int*[height];
    for (int i = 0; i < height; ++i) {
      this->board[i] = new int[width];
      for (int j = 0; j < width; ++j) {
        this->board[i][j] = board[i][j];
      }
    }
    children = new Node*[width];
    expandable_nodes = new int[width];
    for (int i = 0; i < width; ++i) {
      if (top[i]) {
        expandable_nodes[expandableNodeNum++] = i;
      }
      children[i] = NULL;
    }
  }

  ~Node() {
    for (int i = 0; i < width; ++i)
      if (children[i]) delete children[i];
    for (int i = 0; i < height; ++i) delete[] board[i];
    delete[] children;
    delete[] board;
    delete[] top;
    delete[] expandable_nodes;
  }

  int must(bool chance) {
    int x{0}, y{0}, player{int(chance)}, n{get_width()};
    if (player == 1) {
      for (y = 0; y < n; ++y) {
        if (top[y] > 0) {
          x = top[y] - 1;
          board[x][y] = 1;
          if (userWin(x, y, get_height(), get_width(), board)) {
            board[x][y] = 0;
            return y;
          }
          board[x][y] = 0;
        }
      }
      for (y = 0; y < n; ++y) {
        if (top[y] > 0) {
          x = top[y] - 1;
          board[x][y] = 2;
          if (machineWin(x, y, get_height(), get_width(), board)) {
            board[x][y] = 0;
            return y;
          }
          board[x][y] = 0;
        }
      }
    } else {
      for (y = 0; y < n; ++y) {
        if (top[y] > 0) {
          x = top[y] - 1;
          board[x][y] = 2;
          if (machineWin(x, y, get_height(), get_width(), board)) {
            board[x][y] = 0;
            return y;
          }
          board[x][y] = 0;
        }
      }
      for (y = 0; y < n; ++y) {
        if (top[y] > 0) {
          x = top[y] - 1;
          board[x][y] = 1;
          if (userWin(x, y, get_height(), get_width(), board)) {
            board[x][y] = 0;
            return y;
          }
          board[x][y] = 0;
        }
      }
    }
    return -1;
  }

  int connection(bool chance) {
    //! 先判断横竖，再判断斜着

    bool connect_three{false};
    int i, k;
    int counter = 0;
    int left_x, left_y, right_x, right_y;
    for (int f = 0; f < expandableNodeNum; f++) {
      int y{expandable_nodes[f]};
      int x = top[y] - 1;
      board[x][y] = int(chance) + 1;
      left_y = right_y = y;
      for (k = y - 1; k >= 0; k--) {
        if (board[x][k] == int(chance) + 1)
          left_y--;
        else
          break;
      }
      for (k = y + 1; k < this->get_width(); k++) {
        if (board[x][k] == int(chance) + 1)
          right_y++;
        else
          break;
      }
      counter = right_y - left_y + 1;
      if (counter >= 3)
        if (left_y != 0 && top[left_y - 1] - 1 == x &&
            board[x][left_y - 1] == 0) {
          if (right_y + 1 != this->get_width() && top[right_y + 1] - 1 == x &&
              board[x][right_y + 1] == 0) {
            board[x][y] = 0;
            connect_three = true;
            break;
          }
        }
      left_x = right_x = x;
      left_y = right_y = y;
      for (i = x + 1, k = y - 1; i < this->get_height() && k >= 0; i++, k--) {
        if (board[i][k] == int(chance) + 1)
          left_x++, left_y--;
        else
          break;
      }
      for (i = x - 1, k = y + 1; i >= 0 && k < this->get_width(); i--, k++) {
        if (board[i][k] == int(chance) + 1)
          right_x--, right_y++;
        else
          break;
      }
      counter = right_y - left_y + 1;
      if (counter >= 3) {
        if (left_x + 1 != this->get_height() && left_y != 0 &&
            top[left_y - 1] - 1 == left_x + 1 &&
            board[left_x + 1][left_y - 1] == 0) {
          if (right_x != 0 && right_y + 1 != this->get_width() &&
              top[right_y + 1] - 1 == right_x - 1 &&
              board[right_x - 1][right_y + 1] == 0) {
            board[x][y] = 0;
            connect_three = true;
            break;
          }
        }
      }

      left_x = right_x = x;
      left_y = right_y = y;

      for (i = x - 1, k = y - 1; i >= 0 && y >= 0; i--, k--) {
        if (board[i][k] == int(chance) + 1)
          left_x--, left_y--;
        else
          break;
      }
      for (i = x + 1, k = y + 1;
           i < this->get_height() && k < this->get_width(); i++, k++) {
        if (board[i][k] == int(chance) + 1)
          right_x++, right_y++;
        else
          break;
      }
      counter = right_y - left_y + 1;
      if (counter >= 3) {
        if (left_x != 0 && left_y != 0 && top[left_y - 1] - 1 == left_x - 1 &&
            board[left_x - 1][left_y - 1] == 0) {
          if (right_x + 1 != this->get_height() &&
              right_y + 1 != this->get_width() &&
              top[right_y + 1] - 1 == right_x + 1 &&
              board[right_x + 1][right_y + 1] == 0) {
            board[x][y] = 0;
            connect_three = true;
          }
        }
      }
      board[x][y] = 0;
      if (connect_three) {
        board[x][y] = 0;
        return y;
      }
      board[x][y] = 0;
    }
    return -1;
  }
};

class UCT {
 private:
  int width, height;    // 棋盘规格
  int ban_x, ban_y;     // 被去除的点位
  bool expanded{true};  // 是否已经展开
  Node* root;

 public:
  const Node* get_root() const { return root; }
  const int& get_width() { return width; }
  const int& get_height() { return height; }
  const int& get_ban_x() { return ban_x; }
  const int& get_ban_y() { return ban_y; }
  bool& get_expanded() { return expanded; }
  int* weight{NULL};
  int full_weight{0};

  UCT(int width, int height, int ban_x, int ban_y, int** board, const int* top,
      bool expanded = true)
      : width(width), height(height), ban_x(ban_x), ban_y(ban_y) {
    int* root_topStatus = new int[this->get_width()];
    for (int i = 0; i < this->width; i++) {
      root_topStatus[i] = top[i];
    }
    this->root =
        new Node(this->get_height(), this->get_width(), this->get_ban_x(),
                 this->get_ban_y(), board, root_topStatus);
    delete[] root_topStatus;
    int middle = width / 2 - (width + 1) % 2;
    this->weight = new int[width];
    for (int i = 0; i < width; i++) {
      if (i <= middle)
        this->weight[i] = i + 1;
      else
        this->weight[i] = weight[width - i - 1];
    }
    this->full_weight = (width / 2 + width % 2 + 1) * (width / 2 + width % 2) -
                        (width % 2) * (width / 2 + 1);
  }

  ~UCT() { delete this->root; delete[] this->weight; }

  Node* search() {
    int count{0}, index{0};
    if ((index = root->must(root->get_chance())) != -1) {
      int new_x{--root->top[index]};
      root->board[new_x][index] = int(root->get_chance()) + 1;
      if (ban_x == new_x - 1 && ban_y == index) root->top[index]--;
      return root->children[index] =
                 new Node(root->get_height(), root->get_width(),
                          root->get_ban_x(), root->get_ban_y(), root->board,
                          root->top, new_x, index, !(root->get_chance()), root);
    }

    while (count++ < 1000000) {
      if (clock() - startTime > timelimit) break;
      Node* next_node = treePolicy(root);
      double profit = defaultPolicy(next_node);
      while (next_node) {
        ++(next_node->get_visit());
        next_node->get_profit() += profit;
        next_node = next_node->parent;
      }
    }
    return defaultChild();
  }

  Node* treePolicy(Node* node) {
    bool terminate{false};
    while (true) {
      if (!node->parent)
        terminate = false;
      else if (isTie(node->get_width(), node->top) ||
               (node->get_chance() &&
                machineWin(node->get_position_x(), node->get_position_y(),
                           node->get_height(), node->get_width(),
                           node->board)) ||
               (!node->get_chance() &&
                userWin(node->get_position_x(), node->get_position_y(),
                        node->get_height(), node->get_width(), node->board)))
        terminate = true;
      if (terminate) break;
      if (node->get_expandableNodeNum() > 0) {
        return expand(node);
      } else {
        node = bestChild(node);
      }
    }
    return node;
  }

  Node* defaultChild() {
    double recorder{-30000000.0};
    Node* default_child{NULL};
    for (int i = 0; i < this->root->get_width(); ++i) {
      if (this->root->children[i]) {
        double current_score = (this->root->get_chance() ? -1 : 1) *
                               double(this->root->children[i]->get_profit()) /
                               this->root->children[i]->get_visit();
        if (current_score > recorder) {
          default_child = this->root->children[i];
          recorder = current_score;
        }
      }
    }
    return default_child;
  }

  Node* bestChild(Node* node) {
    double recorder{-30000000.0};
    Node* best_node{NULL};
    for (int i = 0; i < node->get_width(); ++i) {
      if (node->children[i]) {
        //! 0.707, 0.55
        double current_score = (node->get_chance() ? -1 : 1) *
                                   double(node->children[i]->get_profit()) /
                                   node->children[i]->get_visit() +
                               0.707 * sqrt(2 * log(double(node->get_visit())) /
                                            node->children[i]->get_visit());
        if (current_score > recorder) {
          best_node = node->children[i];
          recorder = current_score;
        }
      }
    }
    return best_node;
  }

  Node* expand(Node* node) {
    int random_number{rand() % (node->get_expandableNodeNum())};
    int** board = new int*[height];
    for (int i = 0; i < height; ++i) {
      board[i] = new int[this->get_width()];
      for (int j = 0; j < this->get_width(); ++j) {
        board[i][j] = node->board[i][j];
      }
    }
    int index{-1};
    int* top = new int[this->get_width()];
    for (int i = 0; i < this->get_width(); ++i) {
      top[i] = node->top[i];
    }
    int position_y = node->expandable_nodes[random_number];
    int position_x = --(top[position_y]);
    board[position_x][position_y] = node->get_chance() ? 1 : 2;
    if (position_x - 1 == this->get_ban_x() && position_y == this->get_ban_y())
      --(top[position_y]);
    node->children[position_y] =
        new Node(this->get_height(), this->get_width(), this->get_ban_x(),
                 this->get_ban_y(), board, top, position_x, position_y,
                 !node->get_chance(), node);
    delete[] top;
    for (int i = 0; i < this->get_height(); ++i) {
      delete[] board[i];
    }
    delete[] board;
    std::swap(node->expandable_nodes[random_number],
              node->expandable_nodes[--(node->get_expandableNodeNum())]);
    return node->children[position_y];
  }

  int middle_the_best(Node* node) {
    int* top = new int[width];
    for (int i = 0; i < width; ++i) {
      top[i] = node->top[i];
    }
    while (true) {
      int index = rand() % this->full_weight;
      int count{0};
      for (int i = 0; i < width; ++i) {
        count += this->weight[i];
        if (count >= index) {
          delete[] top;
          return i + int(count % index == 0);
        }
      }
    }
  }

  double defaultPolicy(Node* node) {
    int** board = new int*[height];
    int* top = new int[width];
    for (int i = 0; i < width; ++i) {
      top[i] = node->top[i];
    }
    bool chance{node->get_chance()};
    int position_x = node->get_position_x();
    int position_y = node->get_position_y();
    for (int i = 0; i < height; ++i) {
      board[i] = new int[width];
      for (int j = 0; j < width; ++j) {
        board[i][j] = node->board[i][j];
      }
    }
    double profit{0};
    if (chance && machineWin(position_x, position_y, height, width, board))
      profit = double(1);
    else if (!chance && userWin(position_x, position_y, height, width, board))
      profit = double(-1);
    else if (isTie(width, top))
      profit = double(0);
    else
      profit = double(-2);
    while (profit == double(-2)) {
      chance = !chance;
      position_y = 0;
      //!
      // while (top[position_y] == 0) position_y = rand() % width;
      //!
      while (true) {
          int index = rand() % this->full_weight;
          int count{0};
          for (int i = 0; i < width; ++i) {
            count += this->weight[i];
            if (count > index) {
              position_y = i;
              break;
            }
          }
          if (top[position_y] != 0) break;
        }
        //!
      position_x = --top[position_y];
      board[position_x][position_y] = chance ? 2 : 1;
      if (position_y == ban_y)
        if (position_x - 1 == ban_x) --top[position_y];
      //* board[position_x][position_y]
      if (chance && machineWin(position_x, position_y, height, width, board))
        profit = double(1);
      else if (!chance && userWin(position_x, position_y, height, width, board))
        profit = double(-1);
      else if (isTie(width, top))
        profit = double(0);
      else
        profit = double(-2);
    }
    for (int i = 0; i < height; ++i) {
      delete[] board[i];
    }
    delete[] top;
    delete[] board;
    return profit;
  }
};
