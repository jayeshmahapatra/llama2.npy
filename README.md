Work in Progress

---
The idea is to create an inference only python script that uses just python + numpy to do inference.

To do:

- [X] Read weights off of bin files and store them into torch tensors in python
- [X] Update torch tensors using read weights and able to perform full inference
- [X] Reimplement torch specific functions ( like activations ) using pure matrix operations in torch
- [X] Port all torch tensors to numpy and all matrix operations to numpy
- [X] Finalize inference, remove dependencies other than numpy and python
- [X] Port Tokenizer to Python
