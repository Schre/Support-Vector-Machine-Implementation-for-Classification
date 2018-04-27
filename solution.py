import sys
sys.path.append('libsvm/python/')
from svmutil import svm_train, svm_predict
'''
Homework5: support vector machine classifier

You need to use two functions 'svm_train' and 'svm_predict'
from libsvm library to start your homework. Please read the 
readme.txt file carefully to understand how to use these 
two functions.

'''




'''
Use 'svm_train' function with training label, data and different value of cost c to train a svm classify model. Then apply the trained model
on testing label and data.
The value of cost c you need to try is listing as follow:
c = [0.01, 0.1, 1, 2, 3, 5]
Please keep other parameter options as default.
No return value is needed
'''

def svm_with_diff_c(train_label, train_data, test_label, test_data):
    c = [0.01, 0.1, 1, 2, 3, 5]
    i = 0
    
    for el in c:
        cValue = '-c ' + str(c[i])
        print(cValue)
        m = svm_train(train_label, train_data, cValue)
        support_vectors = m.get_SV()
        print('num sv:' + str(len(support_vectors)))
        p_label, p_acc, p_val = svm_predict(test_label, test_data, m)
        i += 1


'''
Use 'svm_train' function with training label, data and different kernel
to train a svm classify model. Then apply the trained model
on testing label and data.
The kernel  you need to try is listing as follow:
1. linear kernel
2. polynomial kernel
3. radial basis function kernel
Please keep other parameter options as default.
No return value is needed
'''
def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
    print('linear kernel')
    m = svm_train(train_label, train_data, '-t 0')
    p_label, p_acc, p_val = svm_predict(test_label, test_data, m)
    support_vectors = m.get_SV()
    print('num sv:' + str(len(support_vectors)))
    
    
    print('polynomial kernel')
    m = svm_train(train_label, train_data, '-t 1')
    p_label, p_acc, p_val = svm_predict(test_label, test_data, m)
    support_vectors = m.get_SV()
    print('num sv:' + str(len(support_vectors)))
    
    
    print('radial kernel')
    m = svm_train(train_label, train_data, '-t 2')
    p_label, p_acc, p_val = svm_predict(test_label, test_data, m)
    support_vectors = m.get_SV()
    print('num sv:' + str(len(support_vectors)))