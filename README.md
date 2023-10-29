# llama2.npy

This repository is a Python+[Numpy](https://numpy.org/doc/stable/index.html) port of [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy.

This repo implements a baby LLama2 architecture using only pure python and Numpy, as well implementing an accompanying byte pair tokenizer in python. The original code, model weights and tokenizer scores are also from Andrej Karpathy.

## Usage

#### 1. Clone the repository.
```
git clone https://github.com/jayeshmahapatra/llama2.npy
```

### 2. Install Numpy
```
pip install numpy==1.23.5
```

### 3. Run using command line
```
python run_npy.py -i "Once upon a time" -w weights/stories15M.bin -n 20
```


