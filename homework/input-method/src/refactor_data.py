from tqdm import tqdm
from typing import List
from collections import Counter, OrderedDict
from IPython import embed
import csv
import functools
import argparse
import time
import json
from pathlib import Path
import os
import numpy as np


def metric(fn):
    """running time for each main function"""

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        print('start executing %s' % (fn.__name__))
        start_time = time.time()
        result = fn(*args, **kw)
        end_time = time.time()
        t = 1000 * (end_time - start_time)
        print('%s executed in %s ms' % (fn.__name__, t))
        return result
    return wrapper


def is_chinese(char)-> bool:
    """ check chinese"""
    return '\u4e00' <= char <= '\u9fa5'

def get_list(target: str) -> List[str]:
    """get a list of working documents."""

    dirs = os.listdir(target)
    file_list = []
    for file in dirs:
         file_list.append(f"{target}/{file}")
    return file_list


@metric
def refactor() -> None:
    """refactor json to utf-8 and typical json"""
    orginal_list = get_list(Path.cwd()/"origin_data")
    index = 0
    for each in tqdm(orginal_list):
        order = []
        if "wiki" in each:
            with open(each, "r", encoding="utf-8", errors="ignore") as f:
                context = "text"
                for line in f.readlines():
                    dic = json.loads(line)
                    order.append(dic)
        else:
            try:
                with open(each, "r", encoding="gbk", errors="ignore") as f:
                    context = "html"
                    for line in f.readlines():
                        dic = json.loads(line)
                        order.append(dic)
            except Exception as e:
                pass
        store_path = Path.cwd() / "training_data"
        if not store_path.is_dir():
            os.makedirs(store_path)
        string = ""
        new_string = ""
        store_string = ""
        for each_text in order:
            string += each_text[f"{context}"].strip()\
                .replace(u'\u3000', u'').replace('\n', '').replace('\r', '').replace(" ", "")
        for each_word in string:
            if is_chinese(each_word) or each_word == "，" or each_word == "。":
                new_string += each_word
        trans_list = new_string.split("，")
        end_list = []
        for every in trans_list:
            end_list += every.split("。")
        for every in end_list:
            if len(every) <= 5:
                continue
            store_string += every + "|"
        with open(f"{store_path}/{index}.json", "w+", encoding="utf-8", errors="ignore") as f:
            json.dump(store_string, f, ensure_ascii=False, indent=2)
            index += 1


@metric
def deduplicate_text()-> None:
    orginal_list = get_list(Path.cwd() / "training_data")

    for each in tqdm(orginal_list):
        try:
            with open(each, "r", encoding="utf-8", errors="ignore") as f:
                contents = json.loads(f.read())
            string_list = contents.split("|")
            store_list = []
            [store_list.append(each) for each in string_list if (not each in store_list and each != "")]
            store_string = ""
            for every in store_list:
                store_string += every + "|"
            with open(each, "w", encoding="utf-8", errors="ignore") as t:
                json.dump(store_string, t, ensure_ascii=False, indent=2)
        except Exception as e:
            print(e)
            print(each)
            pass

@metric
def refactor_dictionary():
    """
    refactor the original dictionary into a true dictionary stored in dictionary.npz
    """
    working_file = Path.cwd() / "拼音汉字表/拼音汉字表.txt"
    with open(working_file, encoding="gbk", errors="ignore") as f:
        lines = f.readlines()
        dic = {}
        for line in lines:
            dictionary = line.split()
            key, value = dictionary[0], dictionary[1:]
            dic[key] = value
    np.savez(Path.cwd()/"dictionary.npz", dic=dic)


@metric
def parser_data():
    parser = argparse.ArgumentParser(description='Choose A Reasonable Training Set. You can choose Large or Small', allow_abbrev=True)
    parser.add_argument('-size', '--training set size', dest='size', type=str, default="Small", help="Choose A Reasonable\
     Training Set. You can choose Large or Small")
    size = parser.parse_args().size
    print(size)
    try:
        assert size == ("Large" or "Small" )
    except:
        print(f"You may use Large or Small. But you have input {size}")
        print("Thus, the progress would exit right now.")
        exit(1)
    return parser.parse_args().size


if __name__ == '__main__':
    #refactor()
    deduplicate_text()
    # refactor_dictionary()
