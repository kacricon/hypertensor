# hypertensor

Bare-bones library for tensor manipulation and autograd written in C.

## current goal

Train a neural net bigram model similar to the what is discussed in the [first video](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2) of the `makemore` tutorial series:

[x] Load names.txt data.
[x] Encode data as integers and generate x and y vectors.
[] One-hot encode input integer vector.
[] Random init of float tensors.
[] Matmul.
[] Exp.
[] Sum reduce op.
[] Broadcast division op.
[] Log op.
[] Mean reduce op.
[] Sum/sub op.
[] Autograd for all ops.

