import numpy as np

## Create dataset

n = 10

x = np.ones([n, 2])
x[:, 0] = np.random.uniform(-1,0,n)





if __name__ == "__main__":
    print(x)
