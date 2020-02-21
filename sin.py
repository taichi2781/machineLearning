import numpy as np
import tensorflow as tf
import tflearn
import matplotlib.pyplot as plt

input_data = np.arange(0,6,0.01)
output_data = np.sin(input_data)+1+np.random.randn(input_data.size)/3
dataset=np.vstack((input_data,output_data))
idx=np.random.choice(600,100)
idx.sort()
dataset_mini=dataset[0:,idx]
train_x=dataset_mini[0].reshape(100,1)
train_y=dataset_mini[1].reshape(100,1)



x=tf.placeholder(tf.float32,[None,1]) 

w1=tf.Variable(tf.truncated_normal([1,5]))
b1=tf.Variable(tf.zeros([5]))

w2=tf.Variable(tf.truncated_normal([5,1]))
b2=tf.Variable(tf.zeros([1]))

y0=tf.nn.sigmoid(tf.matmul(x,w1)+b1)
y1=tf.matmul(y0,w2)+b2

y=tf.placeholder(tf.float32,[None,1])

loss=tf.reduce_sum(tf.square(y1-y))

train_step=tf.train.AdamOptimizer().minimize(loss)

sess=tf.Session()

sess.run(tf.global_variables_initializer())

i=0
for step in range(50001): 
  i=i+1
  sess.run(train_step , feed_dict={x:train_x , y:train_y})
  if i % 500 == 0 :
    loss_val = sess.run(loss,feed_dict={x:train_x,y:train_y})
    print('Step: %d, loss : %f' % (i,loss_val))

#sess.close()


#y0=tf.nn.sigmoid(tf.matmul(x,w1)+b1)
#y1=tf.matmul(y0,w2)+b2

#fig=plt.figure()
yans = sess.run(y1,feed_dict={x:train_x})
log=open('log','w')
for i in range(100):
  log.write(str(train_x[i][0])+" "+str(yans[i][0])+"\n")  

log.close()
sinx=np.sin(train_x)+1

plt.plot(train_x,train_y,label="train date",marker="o",color="red",linestyle="None")
plt.plot(train_x,yans,label="result",color="blue" )
plt.plot(train_x,sinx,label="sinx+1",color="green")

plt.legend()
plt.show()
#fig.savefig("sin_600_50000.png")
