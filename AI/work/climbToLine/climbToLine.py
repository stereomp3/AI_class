import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)


def predict(a, xt):
    return a[0] + a[1] * xt


def MSE(a, x, y):
    total = 0
    for i in range(len(x)):
        total += (y[i] - predict(a, x[i])) ** 2
    return total


def loss(p):
    return MSE(p, x, y)


def ArrayAdd(l1, l2, sub=False):
    l1c = l1.copy()
    for i in range(len(l1c)):
        if not sub:
            l1c[i] += l2[i]
        else:
            l1c[i] -= l2[i]
    return l1c


def hillClimbing(f, hx, h=0.01):
    h_x = hx.copy()
    while True:
        In = False
        for i in range(len(h_x)):
            dh = []
            for num in range(len(h_x)):
                if num == i:
                    dh.append(h)
                else:
                    dh.append(0)
            nh_x = ArrayAdd(h_x, dh)
            nh_x2 = ArrayAdd(h_x, dh, True)
            if f(nh_x) < f(h_x):
                h_x = nh_x
                In = True
            if f(nh_x2) < f(h_x):
                h_x = nh_x2
                In = True
        if not In:
            break
    return h_x


def optimize():
    p = [0.0, 0.0]
    return hillClimbing(loss, p)


p = optimize()

# Plot the graph
y_predicted = list(map(lambda t: p[0] + p[1] * t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
