import numpy as np
import matplotlib.pyplot as plt


from NearestNeighbor import *
from Backpropagation import *

print("Hello World:)\n")
ds=DataSet()
util=Utilities()

util.test()

Train, Valid, Test = ds.load_MNIST()

Train_images=Train[0]
Train_labels=Train[1]
Valid_images=Valid[0]
Valid_labels=Valid[1]
A = Train_images[0:3000, :]
B = Train_labels[0:3000]
C = Valid_images[0:1000, : ]
D = Valid_labels[0:1000]

T1_im = Train_images[0:1001, :]
T2_im = Train_images[1001:2001, :]
T3_im = Train_images[2001:3001, :]
T1_lab = Train_labels[0:1001]
T2_lab = Train_labels[1001:2001]
T3_lab = Train_labels[2001:3001]
"""print A.shape
print B.shape"""

def PlotSample(ARR_im, ARR_lb, num):
    """Complete the code of the PlotSample function,
    which takes the number of a sample as an argument and plots it using the
    functionality of the matplotlib library."""
    #util.exit_with_error("COMPLETE THE FUNCTION ACCORDING TO LABSPEC!!\n")
    array = ARR_im[num, :]
    array=np.reshape(array, (28, 28))
    plt.imshow(array, cmap='gray', interpolation='nearest')
    plt.title(str(ARR_lb[num]))
    plt.show()

    return

def AnalyseData(ARR_im, num):
    #util.exit_with_error("COMPLETE THE FUNCTION ACCORDING TO LABSPEC!!\n")
    d = {}
    array = ARR_im[num, :]
    for i in array:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    lists = sorted(d.items())
    x, y = zip(*lists)
    mn = []
    sum_x = 0
    mx = 0
    print d
    mn_v = 1
    mx_v = 0
    for key in d:
        if d[key] <= mn_v:
            mn.append(key)
        if d[key] > mx_v:
            mx_v = d[key]
            mx = key
        sum_x += key
    sum_x = sum_x/len(d)
    mean = 0
    for key in d:
        mean += abs(key-sum_x)
    mean = mean/len(d)
    stand = 0
    for key in d:
        stand += ((key-sum_x)**2)
    stand = np.sqrt(stand / len(d))
    plt.plot(x, y)
    print "min: " + str(mn) + \
          "\nmax: " + str(mx) + \
          "\nmean deviation: " + str(mean) + \
          "\nstandard deviation: " + str(stand)
    plt.show()
    return


nn=NearestNeighborClass()
#nn.train(A, B) # train the classifier on the training images and labels
k = nn.cross_val(T1_im, T1_lab, T2_im, T2_lab, T3_im, T3_lab)
print "k: " + str(k)

Labels_predict = nn.predict(C, k) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print ('accuracy: %f' % ( np.mean(D == Labels_predict) ))



#PlotSample(Valid_images, Labels_predict, 10)
"""PlotSample(Valid_images, Labels_predict, 111)
PlotSample(Valid_images, Labels_predict, 200)
PlotSample(Valid_images, Labels_predict, 333)
PlotSample(Valid_images, Labels_predict, 445)
PlotSample(Valid_images, Labels_predict, 567)
PlotSample(Valid_images, Labels_predict, 666)
PlotSample(Valid_images, Labels_predict, 700)
PlotSample(Valid_images, Labels_predict, 832)
PlotSample(Valid_images, Labels_predict, 982)"""

#AnalyseData(Train_images, 2)

def prepare_for_backprop(batch_size, Train_images, Train_labels, Valid_images, Valid_labels):
    
    print "Creating data..."
    batched_train_data, batched_train_labels = util.create_batches(Train_images, Train_labels,
                                              batch_size,
                                              create_bit_vector=True)
    batched_valid_data, batched_valid_labels = util.create_batches(Valid_images, Valid_labels,
                                              batch_size,
                                              create_bit_vector=True)
    print "Done!"


    return batched_train_data, batched_train_labels,  batched_valid_data, batched_valid_labels

batch_size=100;

train_data, train_labels, valid_data, valid_labels=prepare_for_backprop(batch_size, Train_images, Train_labels, Valid_images, Valid_labels)

mlp = MultiLayerPerceptron(layer_config=[784, 100, 100, 10], batch_size=batch_size)

mlp.evaluate(train_data, train_labels, valid_data, valid_labels,
             eval_train=True)

PlotSample(Valid_images, Labels_predict, 10)
AnalyseData(Train_images, 2)
print("Done:)\n")
    
