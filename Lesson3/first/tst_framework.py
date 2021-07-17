import numpy as np
import framework as fw
import matplotlib.pyplot as plt


model = fw.Sequential(
    fw.Linear(2, 1),
)
criterion = fw.MSE()
epochs = 100
learning_rate = 1e-3
batch_size = 1
X = np.array([[5, 4]])
Y = np.array([[21]])
print(X.shape)
print(Y.shape)



history = []

for i in range(epochs):
    for x, y_true in fw.loader(X, Y, batch_size):
        # forward -- считаем все значения до функции потерь
        #print(f'x : {x} x.shape = {x.shape}')
        #display(y_true.reshape(batch_size).shape)
        y_pred = model.forward(x)
        #print(f'y_true = {y_true} y_true.shape = {y_true.shape}')
        #print(f'y_pred = {y_pred} y_pred.shape = {y_pred.shape}')
        #print(f'model.W = {model.layers[0].W}')


        #display(y_pred)
        #display(y_true)
        #display(y_pred - y_true)
        loss = criterion.forward(y_pred, y_true)
        print(f'loss: {loss}')

        #print(y_pred, y_true)
        #print('SUM OF SQUARES:', np.mean(np.power(y_pred-y_true, 2)))
    
        # backward -- считаем все градиенты в обратном порядке
        grad = criterion.backward(y_pred, y_true)
        #print(f'grad = {grad} grad.shape = {grad.shape}')
        #display((y_pred - y_true))
        model.backward(x, grad)
        
        # обновляем веса
        fw.SGD(model.parameters(),
            model.grad_parameters(),
            learning_rate)
        
        #print(model.layers[0].W[0][0])
        print(loss)
        
        #print()
        #print('----------------------------------')
        #print()
        history.append(loss)

plt.title("Training loss")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.plot(history, 'b')
plt.show()