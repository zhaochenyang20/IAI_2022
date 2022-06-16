import os
import json
import math
import argparse
import numpy as np
from tqdm import tqdm
from p_tqdm import p_map
from pathlib import Path
from IPython import embed
from typing import List
from refactor_data import metric, get_list
from collections import Counter, OrderedDict


dictionary = np.load(Path.cwd() / "dictionary.npz", allow_pickle=True)["dic"].item()
one, two, three = [Counter(each[1].item()) for each in np.load(Path.cwd() / "full_set.npz", allow_pickle=True).items()]


class Point:
    """
    this character
    last is a index point to the last character of last layer
    the cost till now
    """

    def __init__(self, character):
        self.now = character
        self.cost = -math.inf
        self.last = -1


def evaluate_sentence(file_one, file_two):
    """
    evaluate the accuracy
    """
    with open(file_one, encoding="utf-8", errors="ignore") as f:
        lines_one = f.readlines()
    with open(file_two, encoding="utf-8", errors="ignore") as f:
        lines_two = f.readlines()
    index = 0
    for (one, two) in zip(lines_one, lines_two):
        if one != two:
            index += 1
    sentence = f"sentence accuracy is {1 - index / len(lines_one)}"
    with open("./log.txt", "a", encoding="utf-8", errors="ignore") as t:
        t.write(sentence + "\r")


def evaluate_word(file_one, file_two):
    """
    evaluate the accuracy
    """
    with open(file_one, encoding="utf-8", errors="ignore") as f:
        lines_one = f.readlines()
    with open(file_two, encoding="utf-8", errors="ignore") as f:
        lines_two = f.readlines()
    count = 0
    total = 0
    for (one, two) in zip(lines_one, lines_two):
        total += len(one)
        for (a, b) in zip(one, two):
            if a != b:
                count += 1
    sentence = f"word accuracy is {1 - count / total}"
    with open("./log.txt", "a", encoding="utf-8", errors="ignore") as t:
        t.write(sentence + "\r")


def cost(characters: str, coefficient: List[float]) -> float:
    """
    Since the three unit cost is determined by three characters, we need three characters to get the cost.
    """
    x = coefficient[0]
    y = coefficient[1]
    model = len(characters)
    try:
        if model == 1:
            try:
                return math.log(one[f"{characters}"] / 10_000_000)
            except:
                return -10000
        elif model == 2:
            try:
                return math.log((1 - x) * one[f"{characters[1]}"] / 10_000_000 + x * (two[f"{characters}"] / one[f"{characters[0]}"]))
            except:
                return -10000
        else:
            try:
                return math.log(y * three[characters] / two[characters[0:2]] + (1 - y) * ((1 - x) * one[f"{characters[2]}"] / 10_000_000 + x * (two[f"{characters[1:3]}"] / one[f"{characters[1]}"])))
            except:
                return -10000
    except Exception as e:
        embed()
        exit()


def viterbi(line:str, coefficient = [0.4, 0.5]) -> str:
    """
    use a line of chinese pining, return a str of chinese characters.
    """
    # build a graph for viterbi
    line = line.split()
    try:
        graph = [dictionary[each.lower()] for each in line]
    except:
        embed()
        exit()
    layer_list = []
    # algorithm
    for index in range(len(graph)):
        layer = []
        if index == 0:
            for each_character in graph[index]:
                point = Point(each_character)
                point.cost = cost(point.now, coefficient)
                layer.append(point)
            layer_list.append(layer)
        elif index == 1:
            for each_character in graph[index]:
                point = Point(each_character)
                for last_point in layer_list[index - 1]:
                    tmp_cost = last_point.cost + cost(last_point.now + point.now, coefficient)
                    if tmp_cost > point.cost:
                        point.cost = tmp_cost
                        point.last = layer_list[index - 1].index(last_point)
                layer.append(point)
            layer_list.append(layer)
        else:
            for each_character in graph[index]:
                point = Point(each_character)
                for last_point in layer_list[index - 1]:
                    tmp_cost = last_point.cost + cost(layer_list[index - 2][last_point.last].now + last_point.now + point.now, coefficient)
                    if tmp_cost > point.cost:
                        point.cost = tmp_cost
                        point.last = layer_list[index - 1].index(last_point)
                layer.append(point)
            layer_list.append(layer)
    choice, max_cost, print_list = layer_list[-1][-1], -math.inf, []
    try:
        for each in layer_list[-1]:
            if each.cost > max_cost:
                choice = each
                max_cost = each.cost
        print_list.append(choice.now)
        for i in range(len(layer_list) - 2, -1, -1):
            choice = layer_list[i][choice.last]
            print_list.append(choice.now)
        output_str = ""
        for each in print_list[::-1]:
            output_str += each
        return output_str
    except:
        embed()


def parser_data():
    parser = argparse.ArgumentParser(
        prog='Pinyin Input Method',
        description='Pinyin to Chinese.',
        allow_abbrev=True,
    )
    parser.add_argument('-i', '--input-file', dest='input_file_path', type=str, help="Input file")
    parser.add_argument('-o', '--output-file', dest='output_file_path', type=str, help="Output file")
    parser.add_argument('-c', '--coefficient', dest='coefficient', type=float, nargs=2, default=[0.4, 0.5], help="coefficient")
    input_file_path = parser.parse_args().input_file_path
    output_file_path = parser.parse_args().output_file_path
    coefficient = parser.parse_args().coefficient
    try:
        assert os.path.exists(input_file_path) == True
    except:
        print(f"You may use an existing file. But you have use an unexisting file: {input_file_path}")
        print("Thus, the progress would exit right now.")
        exit(1)
    try:
        assert len(coefficient) == 2 and coefficient[0] <= 1 and coefficient[1] <= 1
    except:
        print(f"You may input two coefficient. And theyshould be less than 1. But you have input: {coefficient}")
        print("Thus, the progress would exit right now.")
        exit(1)
    return input_file_path, output_file_path, coefficient


if __name__ == '__main__':
    input_file_path, output_file_path, coefficient = parser_data()
    with open(input_file_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        output_str_list = [viterbi(each, coefficient) for each in lines]
    with open(output_file_path, "w+", encoding="utf-8", errors="ignore") as f:
        for line in output_str_list:
            f.write(line + "\r")
    evaluate_sentence(output_file_path, "../data/std_output.txt")
    evaluate_word(output_file_path, "../data/std_output.txt")
