import pickle
import pydra
import sys

def run_pickled():
    with open('/pydra/pydra/engine/my_function.pkl', 'rb') as file:
        loaded_function = pickle.load(file)

    result = loaded_function(rerun=False)
    
    print(f'Result: {result}')

if __name__ == '__main__':
    run_pickled()
