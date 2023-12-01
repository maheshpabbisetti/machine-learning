import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X /= np.amax(X, axis=0)  
y /= 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

epoch = 5
lr = 0.1

inputlayer_neurons, hiddenlayer_neurons, output_neurons = 2, 3, 1

wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

for i in range(epoch):
    hlayer_act = sigmoid(np.dot(X, wh) + bh)
    output = sigmoid(np.dot(hlayer_act, wout) + bout)

    EO = y - output
    d_output = EO * derivatives_sigmoid(output)
    EH = d_output.dot(wout.T)
    d_hiddenlayer = EH * derivatives_sigmoid(hlayer_act)

    wout += hlayer_act.T.dot(d_output) * lr
    wh += X.T.dot(d_hiddenlayer) * lr

    print(f"-----------Epoch-{i+1} Starts----------")
    print("Input:\n", X)
    print("Actual Output:\n", y)
    print("Predicted Output:\n", output)
    print(f"-----------Epoch-{i+1} Ends----------\n")

print("Input:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", output)
