# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import csv
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
tf.reset_default_graph()

df = pd.read_csv(f'/kaggle/input/titanic/train.csv')
df['Sex'] = pd.Categorical(df['Sex'])
df.dtypes
df['Sex'] = df.Sex.cat.codes.astype('float64')
df['Age'] = df['Age'].fillna(0)

y = df.pop('Survived')
#print(df['Age'])
x = df[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
#x['Age'] = x['Age'] / 10
x = x.astype('float64')
x = tf.keras.utils.normalize(x, axis=1)

x_train = x.to_numpy()
y_train = y.to_numpy()
#y_train = y_train.reshape((y_train.shape[0], 1))
y_train = np.atleast_2d(y_train).T
#print(x_train[0:5, :])
#print(y_train[0:5, :])

xt = tf.placeholder(tf.float64, shape=[None, 6])
yt = tf.placeholder(tf.float64, shape=[None, 1])

W = tf.get_variable("W", [6, 1], initializer = tf.glorot_uniform_initializer(1), dtype=tf.float64)
b = tf.get_variable('b', [1,1], initializer = tf.zeros_initializer(), dtype=tf.float64)

dftest = pd.read_csv(f'/kaggle/input/titanic/test.csv')
dftest['Sex'] = pd.Categorical(dftest['Sex'])
dftest['Sex'] = dftest.Sex.cat.codes.astype('float64')
dftest['Age'] = dftest['Age'].fillna(0)

xtest = dftest[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
#x['Age'] = x['Age'] / 10
xtest = xtest.astype('float64')
xtest = tf.keras.utils.normalize(xtest, axis=1)

x_test = xtest.to_numpy()

Z = tf.add(tf.matmul(xt, W), b)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z, labels=yt))
#print(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #print(W.eval(), b.eval())
    for epoch in range(50000):            
        _ , ecost = sess.run([optimizer, cost], feed_dict={xt: x_train, yt: y_train})
        if (epoch%1000==0):
            #print(xt.eval({xt:x_train, yt:y_train}))
            #print(Z.eval({xt:x_train, yt:y_train}))
            print("Cost at", epoch, "=", ecost)
        
    W = sess.run(W)
    b = sess.run(b)
    #print(W, b)
    Z = tf.nn.sigmoid(tf.add(tf.matmul(xt, W), b))
    Z = tf.dtypes.cast(tf.equal(tf.dtypes.cast((Z >= 0.5), tf.float64),  yt), tf.float64)
    
    #Z = sess.run(Z, feed_dict={xt:x_train, yt:y_train})
    acc = tf.reduce_mean(Z)
    acc = sess.run(acc, feed_dict={xt:x_train, yt:y_train})
    print("Train accuracy: ", acc)
    
    Z = tf.nn.sigmoid(tf.add(tf.matmul(xt, W), b))
    Z = tf.dtypes.cast((Z >= 0.5), tf.float64)
    Z = sess.run(Z, feed_dict={xt:x_test})
    #print(type(Z))
    with open('submission.csv', 'a') as f:
        pid = dftest['PassengerId']
        res = csv.writer(f)
        res.writerow(['PassengerId', 'Survived'])
        for pid, pred in zip(pid, Z):
            res.writerow([pid, int(pred[0])])
    #print ("Test Accuracy:", accuracy.eval({X: x_test, Y: y_test}))