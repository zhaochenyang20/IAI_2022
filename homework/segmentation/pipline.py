import os

def pipeline():
        print("RNN_LSTM")
        os.system("python3 main.py -n RNN_LSTM")
        print("RNN_GRU")
        os.system("python3 main.py -n RNN_GRU")
        print("TextCNN")
        os.system("python3 main.py -n TextCNN")
        print("MLP")
        os.system("python3 main.py -n MLP")


def pipeline2():
        learningList = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        for each in learningList:
                os.system(f"python3 main.py -n TextCNN -l {each}")

if __name__ == "__main__":
        pipeline()
