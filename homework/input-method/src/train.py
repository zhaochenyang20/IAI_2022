import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from p_tqdm import p_map
from pathlib import Path
from IPython import embed
from refactor_data import metric, get_list
from collections import Counter, OrderedDict


class Neuron:
    """
    each training unit
    from each sentence, we train a neuron
    then collect thse neuron to get a neuron for a document
    finally, thse neurons trained from each document will cllapse into one total neuron
    """

    def __init__(self):
        self.one = Counter()
        self.two = Counter()
        self.three = Counter()

    def train_each_sentence(self, sentence):
        """
        :param sentence: a single chinese sentence like "苟利国家生死以"
        :return: the neuron itself
        """
        for i in range(0, len(sentence)):
            self.one[sentence[i]] += 1
            pair = sentence[i: i + 2]
            if len(pair) == 2:
                self.two[pair] += 1
            triple = sentence[i: i + 3]
            if len(triple) == 3:
                self.three[triple] += 1
        return

    def add_neuron(self, neuron):
        """
        add two neuron together
        """
        for key in neuron.one:
            self.one[key] += neuron.one[key]
        for key in neuron.two:
            self.two[key] += neuron.two[key]
        for key in neuron.three:
            self.three[key] += neuron.three[key]
        return

    @metric
    def train_total_dir(self, director):
        """
        use a director of document to train the neuron
        """
        try:
            training_list = get_list(director)
            results = p_map(train_each_document, training_list, range(1, len(training_list) + 1))
            for each in results:
                self.add_neuron(each)
            neo_one = {}
            neo_two = {}
            neo_three = {}
            print("filter 1 unit")
            for each in tqdm(self.one):
                if self.one[each] == 1:
                    continue
                else:
                    neo_one[each] = self.one[each]
            print("filter 2 unit")
            for each in tqdm(self.two):
                if self.two[each] == 1:
                    continue
                else:
                    neo_two[each] = self.two[each]
            print("filter 3 unit")
            for each in tqdm(self.three):
                if self.three[each] == 1:
                    continue
                else:
                    neo_three[each] = self.three[each]
            store_path = Path.cwd() / "final_training_result"
            if not store_path.is_dir():
                os.makedirs(store_path)
            store_name = store_path / "refactor.npz"
            np.savez(store_name, one=neo_one, two=neo_two, three=neo_three)
        except Exception as e:
            print(e)
            embed()


def train_each_document(document, process_id):
    """
    :param document: a json file path, containing a long string like "苟利国家生死以|美国的华莱士比你们不知道高到哪里去了|没这个能力|"
    :return: the neuron itself
    """
    neuron = Neuron()
    try:
        with open(document, "r", encoding="utf-8", errors="ignore") as f:
            contents = json.loads(f.read())
    except:
        return neuron
    string_list = contents.split("|")
    for sentence in string_list:
        neuron.train_each_sentence(sentence)
    store_path = Path.cwd() / "trans_training_result"
    if not store_path.is_dir():
        os.makedirs(store_path)
    store_name = store_path / f"{process_id}.npz"
    np.savez(store_name, neuron.one, neuron.two, neuron.three)
    return neuron


@metric
def refine_npz(name):
    """
    npz is too large, we need a small one
    """
    A = np.load(name, allow_pickle=True)
    neo_one = {}
    neo_two = {}
    neo_three = {}
    print("refactor 1 unit")
    for each in tqdm(A["arr_0"].item()):
        if A["arr_0"].item()[each] == 1:
            continue
        else:
            neo_one[each] = A["arr_0"].item()[each]
    print("refactor 2 unit")
    for each in tqdm(A["arr_1"].item()):
        if A["arr_1"].item()[each] == 1:
            continue
        else:
            neo_two[each] = A["arr_1"].item()[each]
    print("refactor 3 unit")
    for each in tqdm(A["arr_2"].item()):
        if A["arr_2"].item()[each] == 1:
            continue
        else:
            neo_three[each] = A["arr_2"].item()[each]
    np.savez(name, one=neo_one, two=neo_two, three=neo_three)


@metric
def parser_data():
    parser = argparse.ArgumentParser(description='Choose A Reasonable Training Set. You can choose Large or Small',
                                     allow_abbrev=True)
    parser.add_argument('-size', '--training set size', dest='size', type=str, default="Small", help="Choose A Reasonable\
     Training Set. You can choose Large or Small")
    size = parser.parse_args().size
    print(size)
    try:
        assert size == ("Small" or "Large")
    except:
        print(f"You may use Large or Small. But you have input {size}")
        print("Thus, the progress would exit right now.")
        exit(1)
    return parser.parse_args().size


if __name__ == '__main__':
    size = parser_data()
    train_set = Path.cwd() / f"{size}"
    neuron = Neuron()
    neuron.train_total_dir(train_set)

