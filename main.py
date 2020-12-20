import numpy as np

class DataGenerator:
    def __init__(self,N):
        self.N = N
    
    def generate(self):
        x = np.random.rand(self.N,3)
        y = []
        for i in range(self.N):   
            y.append(x[i][0]*x[i][1] + x[i][2])
        return x, y

def phi(x):
    y = np.tanh(x)
    return y

def phi_prime(x):
    y = 1 - (phi(x))**2
    return y

class Jacobian():
    def __init__(self,w,x):
        self.w = np.asarray(w)
        self.x = np.asarray(x)

    def one_row(self,w,x):
        row = np.ones((1,16))[0]
        for i in range(0,11,5):
            inner = w[i + 1] * x[0] + w[i + 2] * x[1] + w[i + 3] * x[2] + w[i + 4]
            row[i + 0] = phi(inner)
            row[i + 1] = w[i] * x[0] * phi_prime(inner)
            row[i + 2] = w[i] * x[1] * phi_prime(inner)
            row[i + 3] = w[i] * x[2] * phi_prime(inner)
            row[i + 4] = w[i] * phi_prime(inner)
        return row

    def calc(self):
        J = np.zeros((self.x.shape[0],16))
        for i in range(self.x.shape[0]):
            J[i, :] = self.one_row(self.w,self.x[i])
        return J

def calc_fw(w,x):
    y = w[15]
    for i in range(0,11,5):
        inner = w[i + 1] * x[0] + w[i + 2] * x[1] + w[i + 3] * x[2] + w[i + 4]
        y = y + w[i] * phi(inner)
    return y

def calc_error_vec(w,x,y):
    error = np.zeros((x.shape[0],1))
    for i in range(x.shape[0]):
        error[i][0] = calc_fw(w,x[i]) - y[i]
    return error

def train(w, x, y, lam, init_lam_k, ideal_norm_error):
    last_norm_error = 1e20
    lam_k = init_lam_k
    error_vec = calc_error_vec(w,x,y)
    this_norm_error = np.linalg.norm(error_vec)
    # while last_error - this_error > min_delta:
    while this_norm_error > ideal_norm_error:  
        j = Jacobian(w,x)
        D = j.calc()
        A1 = D
        A2 = np.eye(16)
        A3 = np.eye(16)

        b1 = (np.dot(D,w) - error_vec)
        b2 = np.zeros((16,1))
        b3 = w

        A = np.zeros((A1.shape[0]+A2.shape[0]+A3.shape[0],16))
        A[0:A1.shape[0]] = A1[:]
        A[A1.shape[0]:A1.shape[0] + A2.shape[0]] = np.sqrt(lam) * A2[:]
        A[A1.shape[0] + A2.shape[0]:A1.shape[0] + A2.shape[0] + A3.shape[0]] = np.sqrt(lam_k) * A3[:]

        b = np.zeros((b1.shape[0]+b2.shape[0]+b3.shape[0],1))
        b[0:b1.shape[0]] = b1[:]
        b[b1.shape[0]:b1.shape[0] + b2.shape[0] , :] = np.sqrt(lam) * b2[:]
        b[b1.shape[0] + b2.shape[0]:b1.shape[0] + b2.shape[0] + b3.shape[0], :] = np.sqrt(lam_k) * b3[:]

        w_k = np.linalg.lstsq(A,b,rcond=None)[0]
        this_error_vec = calc_error_vec(w_k,x,y)
        this_norm_error = np.linalg.norm(this_error_vec)

        if this_norm_error < last_norm_error:
            lam_k = lam_k * 0.8
            w = w_k
            # print(w_k)
            print(this_norm_error)
            last_norm_error = this_norm_error
            error_vec = this_error_vec
        else:
            lam_k = lam_k * 2

        

        
    return w 

gen = DataGenerator(500)
x, y = gen.generate()
w = np.random.rand(16,1)
train(w,x,y,.00005,.00000000005,.05)
