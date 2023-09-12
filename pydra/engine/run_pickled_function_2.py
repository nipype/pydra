import pickle
import pydra
import sys

def run_pickled():
    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]
    file_path_3 = sys.argv[3]
    with open(file_path_1, 'rb') as file:
        loaded_function = pickle.load(file)
    with open(file_path_2, 'rb') as file:
        taskmain = pickle.load(file)
    with open(file_path_3, 'rb') as file:
        ind = pickle.load(file)

    result = loaded_function(taskmain, ind, rerun=False)

    print(f'Result: {result}')    

if __name__ == '__main__':
    run_pickled()
