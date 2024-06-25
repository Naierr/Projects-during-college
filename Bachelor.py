#### READ-ME ####

# Most of the variable names are denoted by x,y,z instead
# of phi,theta,psi to make it easier for the project owner during working and debugging.#
# Roll, Pitch and Yaw --> Phi, Theta, Psi #
# State of UAV is assumed to be hovering#
# Time step 0.01#
# Measurements stands for the measurements of the IMU#

import random
import control.matlab as cnt
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import activations
from keras.layers import advanced_activations
from keras.optimizers import gradient_descent_v2
SGD = gradient_descent_v2.SGD
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score

# from sklearn.model_selection import KFold
x = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
print(x)
# plt.scatter(float(x[0]), float(x[1]), s=100)
# plt.title('Initial Location')
P = np.eye(6) * 1.0
dt = 0.01  # Time Step between Filter Steps

#Model matrices

A = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, -0.0080, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
B = np.array([[0.0, 0.0, 0.0, 0.0],
              [0, 30.6667, 0.0, 0.0],
              [0, 0, 0.0, 0.0],
              [0, 0.0, 0.0, 0.0],
              [0.0, 0, 0.0, 0.0],
              [0.0, 0.0, 0, 17.6923]])
D = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]])
u = np.array([[0],
              [0],
              [0],
              [0]])
ra = 1.0 ** 2
R = np.array([[ra, 0.0, 0.0],
              [0.0, ra, 0.0],
              [0.0, 0.0, ra]])
sv = 1.0

G = np.array([[0.5 * dt ** 2],
              [0.5 * dt ** 2],
              [0.5 * dt ** 2],
              [dt],
              [dt],
              [dt]])

Q = np.multiply(np.multiply(G, G.transpose()),np.multiply(sv,sv))
H = np.eye(6)
I = np.eye(6)
C = np.eye(6)
m = 500  # Measurements
# vxx = 0.0  # in X
# vyy = 0.0  # in Y
# vzz = 0.0
vx = np.zeros(shape=(1, m))
vy = np.zeros(shape=(1, m))
vz = np.zeros(shape=(1, m))
# IMU Angular Velocities Readings (Noisy Data)
mx = np.random.randn(m)
my = np.random.randn(m)
mz = np.random.randn(m)
# IMU
print(mx)
for i in range(m):
    if i > 0:
        vx[0][i] = vx[0][i - 1] + np.random.uniform(-0.01, 0.01)
        vy[0][i] = vy[0][i - 1] + np.random.uniform(-0.01, 0.01)
        vz[0][i] = vz[0][i - 1] + np.random.uniform(-0.01, 0.01)
    else:
        vx[0][i] = 0
        vy[0][i] = 0
        vz[0][i] = 0

xi = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
xout = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
thetaModel = np.array([])
psiModel = np.array([])
phiModel = np.array([])
# Model Real Values
for o in range(m):
    u = [[0], [0], [0], [0]]

    if o > 0:
        xi = [[xout[0] + vx[0][o - 1]],
              [xout[1] + vy[0][o - 1]],
              [xout[2] + vz[0][o - 1]],
              [vx[0][o]],
              [vy[0][o]],
              [vz[0][o]]]
    else:
        xi = [[xout[0]],
              [xout[1]],
              [xout[2]],
              [vx[0][o]],
              [vy[0][o]],
              [vz[0][o]]]
    # Uncomment if smth is not cool.
    # A = np.eye(6)
    xout = np.dot(A, xi) + np.dot(B, u)

    thetaModel = np.append(thetaModel, float(xout[0]))
    psiModel = np.append(psiModel, float(xout[1]))
    phiModel = np.append(phiModel, float(xout[2]))

# REAL angular velocities generated

mxx = np.random.randn(m)
myy = np.random.randn(m)
mzz = np.random.randn(m)
# Off the shelf Auto-pilot system, Real orientation values generated below
MeasureX = np.array([])
MeasureY = np.array([])
MeasureZ = np.array([])
for i in range(m):
    MeasureX = np.append(MeasureX, mxx[i] + thetaModel[i])
    MeasureY = np.append(MeasureY, myy[i] + psiModel[i])
    MeasureZ = np.append(MeasureZ, mzz[i] + phiModel[i])
measurement = np.zeros(shape=(6, m))
# print(mx[51])
for i in range(m):
    if i == 0:
        measurement[3][i] = vx[0][i]
        measurement[4][i] = vy[0][i]
        measurement[5][i] = vz[0][i]
    else:
        measurement[3][i] = vx[0][i] + measurement[3][i - 1]
        measurement[4][i] = vy[0][i] + measurement[4][i - 1]
        measurement[5][i] = vz[0][i] + measurement[5][i - 1]

# V-stacked the measurements real value to be able to plot them in matplot figures

measurements = np.vstack((MeasureX, MeasureY, MeasureZ, mx, my, mz))

# Defined Variables to manipulate data during the next section of the code
#Kalman filter variables mostly.

xt = np.zeros(shape=(1, m))
yt = np.zeros(shape=(1, m))
zt = np.zeros(shape=(1, m))
dxt = np.zeros(shape=(1, m))
dyt = np.zeros(shape=(1, m))
dzt = np.zeros(shape=(1, m))
Zx = np.zeros(shape=(1, m))
Zy = np.zeros(shape=(1, m))
Zz = np.zeros(shape=(1, m))
Px = np.zeros(shape=(1, m))
Py = np.zeros(shape=(1, m))
Pz = np.zeros(shape=(1, m))
Pdx = np.zeros(shape=(1, m))
Pdy = np.zeros(shape=(1, m))
Pdz = np.zeros(shape=(1, m))
Rdx = np.zeros(shape=(1, m))
Rdy = np.zeros(shape=(1, m))
Rdz = np.zeros(shape=(1, m))
Kx = np.zeros(shape=(1, m))
Ky = np.zeros(shape=(1, m))
Kz = np.zeros(shape=(1, m))
Kdx = np.zeros(shape=(1, m))
Kdy = np.zeros(shape=(1, m))
Kdz = np.zeros(shape=(1, m))
# The Kalman Filter

A = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, -0.0080, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
for n in range(len(measurements[0])):

    # Adaptive Measurement Covariance R from last i Measurements
    # as an Maximum Likelihood Estimation
    i = 10
    # If there's a problem uncomment below
    # A = np.eye(6)

    if n > i:
        R = np.array([[0.001, 0.0, 0.0, 0, 0, 0],
                      [0.0, 0.001, 0.0, 0, 0, 0],
                      [0.0, 0.0, 0.001, 0, 0, 0],
                      [0.0, 0.0, 0.0, 0.001, 0, 0],
                      [0.0, 0.01, 0.0, 0, 0.001, 0],
                      [0.0, 0.01, 0.0, 0, 0, 0.001]])
        # Project the state ahead
        x = np.dot(A, x) + np.dot(B, u)
        # x = np.dot(A, x)

        # Project the error covariance ahead
        P = np.dot(np.dot(A, P), np.transpose(A)) + Q

        # Measurement Update (Correction)
        # ===============================
        # Compute the Kalman Gain
        S = np.dot(np.dot(H, P), np.transpose(H)) + R
        K = np.dot(np.dot(P, np.transpose(H)), np.linalg.inv(S))
        # Update the estimate via z
        Z = measurements[:, n].reshape(6, 1)
        y = Z - np.dot(H, x)  # Innovation or Residual
        x = x + np.dot(K, y)
        P = np.dot((I - (np.dot(K, H))), P)

        # Save states for Plotting, Updating...
        xt[0][n] = float(x[0])
        yt[0][n] = float(x[1])
        zt[0][n] = float(x[2])
        dxt[0][n] = float(x[3])
        dyt[0][n] = float(x[4])
        dzt[0][n] = float(x[5])
        Zx[0][n] = float(Z[0])
        Zy[0][n] = float(Z[1])
        Zz[0][n] = float(Z[2])
        Px[0][n] = float(P[0, 0])
        Py[0][n] = float(P[1, 1])
        Pz[0][n] = float(P[2, 2])
        Pdx[0][n] = float(P[3, 3])
        Pdy[0][n] = float(P[4, 4])
        Pdz[0][n] = float(P[5, 5])
        Rdx[0][n] = float(R[0, 0])
        Rdy[0][n] = float(R[1, 1])
        Rdz[0][n] = float(R[2, 2])
        Kx[0][n] = float(K[0, 0])
        Ky[0][n] = float(K[1, 0])
        Kz[0][n] = float(K[2, 0])
        Kdx[0][n] = float(K[3, 0])
        Kdy[0][n] = float(K[4, 0])
        Kdz[0][n] = float(K[5, 0])


# fig = plt.figure(figsize=(19, 11))

# Variables to manipulate in it the data of the kalman and sensor data, K stands for kalman

xtt = np.array([])
ytt = np.array([])
ztt = np.array([])
Dxx = np.array([])
Dyy = np.array([])
Dzz = np.array([])
Kxx = np.array([])
Kyy = np.array([])
Kzz = np.array([])
Kdxx = np.array([])
Kdyy = np.array([])
Kdzz = np.array([])
xFinal = np.array([])
yFinal = np.array([])
zFinal = np.array([])
dxFinal = np.array([])
dyFinal = np.array([])
dzFinal = np.array([])
for o in range(m):
    Kxx = np.append(Kxx, Kx[0][o])
    Kyy = np.append(Kyy, Ky[0][o])
    Kzz = np.append(Kzz, Kz[0][o])
    Kdxx = np.append(Kdxx, Kdx[0][o])
    Kdyy = np.append(Kdyy, Kdy[0][o])
    Kdzz = np.append(Kdzz, Kdz[0][o])
    xtt = np.append(xtt, xt[0][o])
    ytt = np.append(ytt, yt[0][o])
    ztt = np.append(ztt, zt[0][o])
    Dxx = np.append(Dxx, dxt[0][o])
    Dyy = np.append(Dyy, dyt[0][o])
    Dzz = np.append(Dzz, dzt[0][o])
# Doing a variable to create the dataset in 2D at first
Datasett = np.zeros(shape=(m, 18))
for i in range(m):
    # Kalman estimates
    Datasett[i][0] = xtt[i]
    Datasett[i][1] = ytt[i]
    Datasett[i][2] = ztt[i]
    Datasett[i][3] = Dxx[i]
    Datasett[i][4] = Dyy[i]
    Datasett[i][5] = Dzz[i]
    # Real Values
    Datasett[i][6] = thetaModel[i]
    Datasett[i][7] = psiModel[i]
    Datasett[i][8] = phiModel[i]
    Datasett[i][9] = vx[0][i]
    Datasett[i][10] = vy[0][i]
    Datasett[i][11] = vz[0][i]
    # Neural Network only simulation with no Kalman
    # Aka IMU measurements
    Datasett[i][12] = MeasureX[i]
    Datasett[i][13] = MeasureY[i]
    Datasett[i][14] = MeasureZ[i]
    Datasett[i][15] = mx[i]
    Datasett[i][16] = my[i]
    Datasett[i][17] = mz[i]

# x,y,z for orientation , dx,dy,dz for rates ,
# mohem is an arabic word for important as this is going to be a crucial part in my code

xmohem = np.zeros(shape=(1, m))
ymohem = np.zeros(shape=(1, m))
zmohem = np.zeros(shape=(1, m))
dxmohem = np.zeros(shape=(1, m))
dymohem = np.zeros(shape=(1, m))
dzmohem = np.zeros(shape=(1, m))

# Arrays

Xo = np.array([0, 0, 0, 0, 0, 0])
Dataset = np.array([])
yo = np.array([0, 0, 0, 0, 0, 0])
# print(Datasett)
for i in range(m):
    for n in range(12):
        Dataset = np.append(Dataset, Datasett[i][n])
Xk = Datasett[:, 0:3] #Kalman Orientations Estimates
yk = Datasett[:, 6:9] #Real orientaion Values
xnkk = Datasett[:, 12:15] #IMU based measured orientations
Xk =xnkk
xnk = Datasett[:, 15:18] #IMU based measured angular rates
Xvk =xnk
Xvk = Datasett[:, 3:6] #Kalman Rate Estimates
yvk = Datasett[:, 9:12] #Real Rates values
ynk = Datasett[:, 6:12] #Real Values
# ynkk = Datasett[:, 9:12]
#
# define the keras model
X_train, X_test, y_train, y_test = train_test_split(Xk, yk, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
# model = Sequential()
# modelvelocity = Sequential()
modelStep1A = Sequential()
modelStep1B = Sequential()
# opt = SGD(lr=0.01, momentum=0.9)
sgd = SGD(learning_rate=0.2, momentum=0.5)
opt = tf.keras.optimizers.RMSprop()
modelStep1A.add(Dense(10, input_dim=3, activation=activations.relu))
modelStep1A.add(Dense(10, activation=activations.relu))
modelStep1A.add(Dense(10, activation=activations.relu))
modelStep1A.add(Dense(3, activation=activations.linear))
modelStep1B.add(Dense(10, input_dim=3, activation=activations.relu))
modelStep1B.add(Dense(10, activation=activations.relu))
modelStep1B.add(Dense(10, activation=activations.relu))
modelStep1B.add(Dense(3, activation=activations.linear))
modelStep1A.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
modelStep1B.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
modelStep1A.fit(Xvk, yvk, epochs=1000)
modelStep1B.fit(Xk, yk, epochs=1025)
for i in range(m):
    # Xo = [[xtt[i], ytt[i], ztt[i], Dxx[i], Dyy[i], Dzz[i]]]
    Xo = [[mx[i], my[i], mz[i]]]
    Xoo = [[Datasett[i][15], Datasett[i][16], Datasett[i][17]]]
    Xok = [[Dxx[i], Dyy[i], Dzz[i]]]
    Predict = modelStep1A.predict(Xo)
    Predicto = modelStep1B.predict(Xoo)
    dxmohem[0][i] = Predict[0][0]
    dymohem[0][i] = Predict[0][1]
    dzmohem[0][i] = Predict[0][2]
    # xmohem[0][i] = Predicto[0][0]
    # ymohem[0][i] = Predicto[0][1]
    # zmohem[0][i] = Predicto[0][2]
    if (i > 0):
        xmohem[0][i] = xmohem[0][i - 1] + dxmohem[0][i]
        ymohem[0][i] = ymohem[0][i - 1] + dymohem[0][i]
        zmohem[0][i] = zmohem[0][i - 1] + dzmohem[0][i]
    else:
        xmohem[0][i] = dxmohem[0][i]
        ymohem[0][i] = dymohem[0][i]
        zmohem[0][i] = dzmohem[0][i]
    Datasett[i][0] = xmohem[0][i]
    Datasett[i][1] = ymohem[0][i]
    Datasett[i][2] = zmohem[0][i]
    Datasett[i][3] = dxmohem[0][i]
    Datasett[i][4] = dymohem[0][i]
    Datasett[i][5] = dzmohem[0][i]
for i in range(m):
    for n in range(12):
        Dataset = np.append(Dataset, Datasett[i][n])

# y denotes to output, the target value which is real model values generated

Xk = Datasett[:, 0:3]
# Xk = Datasett[:, 12:15]
yk = Datasett[:, 6:9]
xnk = Datasett[:, 3:6]
# xnk = Datasett[:, 15:18]
ynk = Datasett[:, 9:12]
xnkk = Datasett[:, 12:15] #IMU based measured orientations
# Xk =xnkk
xnk = Datasett[:, 15:18] #IMU based measured angular rates
# Xvk =xnk

# define the keras model
X_train, X_test, y_train, y_test = train_test_split(Xk, yk, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
model = Sequential()
modelvelocity = Sequential()
# opt = SGD(lr=0.01, momentum=0.9)
sgd = SGD(learning_rate=0.2, momentum=0.5)
opt = tf.keras.optimizers.RMSprop()
model.add(Dense(10, input_dim=3, activation=activations.relu))
model.add(Dense(10, activation=activations.relu))
model.add(Dense(10, activation=activations.relu))
model.add(Dense(3, activation=activations.linear))
modelvelocity.add(Dense(10, input_dim=3, activation=activations.relu))
modelvelocity.add(Dense(10, activation=activations.relu))
modelvelocity.add(Dense(10, activation=activations.relu))
modelvelocity.add(Dense(3, activation=activations.linear))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
modelvelocity.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
# model.fit(Xk, yk, epochs=100, batch_size=10)
model.fit(Xk, yk, epochs=1000)
modelvelocity.fit(xnk, ynk, epochs=1000)
DirectX = np.array([])
DirectY = np.array([])
DirectZ = np.array([])
DirectXV = np.array([])
DirectYV = np.array([])
DirectZV = np.array([])
for i in range(m):
    # Xo = [[xtt[i], ytt[i], ztt[i], Dxx[i], Dyy[i], Dzz[i]]]
    Xo = [[Datasett[i][0], Datasett[i][1], Datasett[i][2]]]
    Xoo = [[Datasett[i][3], Datasett[i][4], Datasett[i][5]]]
    # Xok = [[Dxx[i], Dyy[i], Dzz[i]]]
    Predict = model.predict(Xo)
    PredictV = modelvelocity.predict(Xoo)
    DirectXV = dxmohem[0][i]
    DirectYV = dymohem[0][i]
    DirectZV = dzmohem[0][i]
    DirectX = np.append(DirectX, xmohem[0][i])
    DirectY = np.append(DirectY, ymohem[0][i])
    DirectZ = np.append(DirectZ, zmohem[0][i])
    dxmohem[0][i] = PredictV[0][0]
    dymohem[0][i] = PredictV[0][1]
    dzmohem[0][i] = PredictV[0][2]
    xmohem[0][i] = Predict[0][0]
    ymohem[0][i] = Predict[0][1]
    zmohem[0][i] = Predict[0][2]
for o in range(m):
    xFinal = np.append(xFinal, xmohem[0][o])
    yFinal = np.append(yFinal, ymohem[0][o])
    zFinal = np.append(zFinal, zmohem[0][o])
    dxFinal = np.append(dxFinal, dxmohem[0][o])
    dyFinal = np.append(dyFinal, dymohem[0][o])
    dzFinal = np.append(dzFinal, dzmohem[0][o])
Vxplot = np.array([])
Vyplot = np.array([])
Vzplot = np.array([])
KPhiError = np.array([])
KThetaError = np.array([])
KPsiError = np.array([])
NPhiError = np.array([])
NThetaError = np.array([])
NPsiError = np.array([])
KPhiErrord = np.array([])
KThetaErrord = np.array([])
KPsiErrord = np.array([])
NPhiErrord = np.array([])
NThetaErrord = np.array([])
NPsiErrord = np.array([])
for o in range(m):
    Vxplot = np.append(Vxplot, vx[0][o])
    Vyplot = np.append(Vyplot, vy[0][o])
    Vzplot = np.append(Vzplot, vz[0][o])
    KPhiError = np.append(KPhiError, thetaModel[o] - xtt[o])
    KThetaError = np.append(KThetaError, psiModel[o] - ytt[o])
    KPsiError = np.append(KPsiError, phiModel[o] - ztt[o])
    NPhiError = np.append(NPhiError, thetaModel[o] - xFinal[o])
    NThetaError = np.append(NThetaError, psiModel[o] - yFinal[o])
    NPsiError = np.append(NPsiError, phiModel[o] - zFinal[o])
    # Angular
    KPhiErrord = np.append(KPhiError, Vxplot[o] - Dxx[o])
    KThetaErrord = np.append(KThetaError, Vyplot[o] - Dyy[o])
    KPsiErrord = np.append(KPsiError, Vzplot[o] - Dzz[o])
    NPhiErrord = np.append(NPhiError, Vxplot[o] - dxFinal[o])
    NThetaErrord = np.append(NThetaError, Vyplot[o] - dyFinal[o])
    NPsiErrord = np.append(NPsiError, Vzplot[o] - dzFinal[o])
# Error Plots : Phi
fig = plt.figure(figsize=(16, 9))
# # plt.plot(range(m), KPhiError, label='${Kalman estimate error}$', color='green')
# plt.plot(range(m), NPhiError, label='${Deep Learning Training}$', color='blue')
# plt.xlabel('Time Step of 0.1 seconds')
# plt.title('Error (relative to the  real value)')
# plt.legend(loc='best', prop={'size': 22})
# plt.ylabel('phi (deg) error')
# # Theta
# fig = plt.figure(figsize=(16, 9))
# # plt.plot(range(m), KThetaError, label='${Kalman estimate error}$', color='green')
# plt.plot(range(m), NThetaError, label='${Deep Learning Training}$', color='blue')
# plt.xlabel('Time Step of 0.1 seconds')
# plt.title('Error (relative to the  real value)')
# plt.legend(loc='best', prop={'size': 22})
# plt.ylabel('Theta (deg) error')
# # Psi
# fig = plt.figure(figsize=(16, 9))
# # plt.plot(range(m), KPsiError, label='${Kalman estimate error}$', color='green')
# plt.plot(range(m), NPsiError, label='${Deep Learning Training}$', color='blue')
# plt.xlabel('Time Step of 0.1 seconds')
# plt.title('Error (relative to the  real value)')
# plt.legend(loc='best', prop={'size': 22})
# plt.ylabel('psi (deg) error')
# # dPhi
# fig = plt.figure(figsize=(16, 9))
# # plt.plot(range(m + 1), KPhiErrord, label='${Kalman estimate error}$', color='green')
# plt.plot(range(m + 1), NPhiErrord, label='${Deep Learning Training}$', color='blue')
# plt.xlabel('Time Step of 0.1 seconds')
# plt.title('Error (relative to the  real value)')
# plt.legend(loc='best', prop={'size': 22})
# plt.ylabel('phi rate (deg/sec) error')
# # dTheta
# fig = plt.figure(figsize=(16, 9))
# # plt.plot(range(m + 1), KThetaErrord, label='${Kalman estimate error}$', color='green')
# plt.plot(range(m + 1), NThetaErrord, label='${Deep Learning Training}$', color='blue')
# plt.xlabel('Time Step of 0.1 seconds')
# plt.title('Error (relative to the  real value)')
# plt.legend(loc='best', prop={'size': 22})
# plt.ylabel('theta rate (deg/sec) error')
# # dPsi
# fig = plt.figure(figsize=(16, 9))
# # plt.plot(range(m + 1), KPsiErrord, label='${Kalman estimate error}$', color='green')
# plt.plot(range(m + 1), NPsiErrord, label='${Deep Learning Training}$', color='blue')
# plt.xlabel('Time Step of 0.1 seconds')
# plt.title('Error (relative to the  real value)')
# plt.legend(loc='best', prop={'size': 22})
# plt.ylabel('psi rate (deg/sec) error')

# State Estimation
fig = plt.figure(figsize=(16, 9))
plt.plot(range(m), mx, label='${measurements}$', color='grey')
# plt.plot(range(m), Dxx, label='${Kalman estimate}$', color='green')
plt.plot(range(m), dxFinal, label='${Deep Learning Training}$', color='blue')
plt.plot(range(m), Vxplot, label='${real}$', linestyle='dashed', color='red')
plt.yticks(np.arange(-10,10,2.5))
plt.xlabel('Time Step of 0.1 seconds')
plt.title('Estimate (Elements from State Vector $x$)')
plt.legend(loc='best', prop={'size': 22})
plt.ylabel('phi rate (deg/sec)')
###
fig = plt.figure(figsize=(16, 9))
plt.plot(range(m), my, label='${measurements}$', color='grey')
# plt.plot(range(m), Dyy, label='${Kalman estimate}$', color='green')
plt.plot(range(m), dyFinal, label='${Deep Learning Training}$', color='blue')
plt.plot(range(m), Vyplot, label='${real}$', linestyle='dashed', color='red')
plt.yticks(np.arange(-10,10,2.5))
plt.xlabel('Time Step of 0.1 seconds')
plt.title('Estimate (Elements from State Vector $x$)')
plt.legend(loc='best', prop={'size': 22})
plt.ylabel('theta rate (deg/sec)')
####
fig = plt.figure(figsize=(16, 9))
plt.plot(range(m), mz, label='${measurements}$', color='grey')
# plt.plot(range(m), Dzz, label='${Kalman estimate}$', color='green')
plt.plot(range(m), dzFinal, label='${Deep Learning Training}$', color='blue')
plt.plot(range(m), Vzplot, label='${real}$', linestyle='dashed', color='red')
plt.xlabel('Time Step of 0.1 seconds')
plt.yticks(np.arange(-10,10,2.5))
plt.title('Estimate (Elements from State Vector $x$)')
plt.legend(loc='best', prop={'size': 22})
plt.ylabel('(psi rate) (deg/sec)')
###
measurementfinalx = np.array([])
measurementfinaly = np.array([])
measurementfinalz = np.array([])
for i in range(m):
    measurementfinalx = np.append(measurementfinalx, thetaModel[i])
    measurementfinaly = np.append(measurementfinaly, psiModel[i])
    measurementfinalz = np.append(measurementfinalz, phiModel[i])
fig = plt.figure(figsize=(16, 9))
plt.plot(range(m), MeasureX, label='${Measurements}$', color='grey')
# plt.plot(range(m), xtt, label='${Kalman estimate}$')
plt.plot(range(m), xFinal, label='${Deep Learning Training}$')
# plt.plot(range(m), DirectX, linestyle='dashed', label='${First step neural network}$')
plt.plot(range(m), measurementfinalx, linestyle='dashed', label='${real}')
plt.xlabel('Time Step of 0.1 seconds')
plt.yticks(np.arange(-90,90,15))
plt.title('Estimates of position (Elements from State Vector $x$)')
plt.legend(loc='best', prop={'size': 22})
plt.ylabel('Phi (deg)')
fig = plt.figure(figsize=(16, 9))
plt.plot(range(m), MeasureY, label='${Measurements}$', color='grey')
# plt.plot(range(m), ytt, label='${Kalman estimate}$')
plt.plot(range(m), yFinal, label='${Deep Learning Training}$')
# plt.plot(range(m), DirectZ, linestyle='dashed', label='${First step neural network}$')
plt.plot(range(m), measurementfinaly, linestyle='dashed', label='${real}$')
plt.yticks(np.arange(-90,90,15))
plt.xlabel('Time Step of 0.1 seconds')
plt.title('Estimates of position (Elements from State Vector $x$)')
plt.legend(loc='best', prop={'size': 22})
plt.ylabel('Theta (deg)')
fig = plt.figure(figsize=(16, 9))
plt.plot(range(m), MeasureZ, label='${Measurements}$', color='grey')
# plt.plot(range(m), ztt, label='${Kalman estimate}$')
plt.plot(range(m), zFinal, label='${Deep Learning Training}$')
# plt.plot(range(m), DirectZ, linestyle='dashed', label='${First step neural network}$')
plt.plot(range(m), measurementfinalz, linestyle='dashed', label='${real}$')
plt.yticks(np.arange(-90,90,15))
plt.xlabel('Time Step of 0.1 seconds')
plt.title('Estimates of position (Elements from State Vector $x$)')
plt.legend(loc='best', prop={'size': 22})
plt.ylabel('Psi (deg)')
plt.show()
