#%%
# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly
from keras.utils.vis_utils import plot_model

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')

from keras import activations
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Dropout ,CuDNNLSTM,SimpleRNN
import tensorflow as tf

#data process
from sklearn import preprocessing
from sklearn import model_selection

#%%
import math
import numpy as np
#save
import os
#csv read
from numpy import genfromtxt
#plot
import subprocess


#ode integerate
from scipy.integrate import odeint

# %%
def test():
    xs = np.linspace(0,20,100)
    y0 = 10.0  # the initial condition
    ys = odeint(dy_dx, y0, xs)
    ys = np.array(ys).flatten()

    #%%
    # Plot the numerical solution
    plt.rcParams.update({'font.size': 14})  # increase the font size
    plt.xlabel("t")
    plt.ylabel("x")
    plt.plot(xs, ys)

def train_NN(X_train_nor, X_test_nor, y_train, y_test,pred_data_set,sim_data_set,mean,std,look_back):
    #create np random
    model_struct = np.random.randint(2,12,np.random.randint(3,13))
    model_struct =[7,8,7,6]
    model_name = 'Ly_N_%s_'%(len(model_struct)+1) +'par_'+'_'.join(str(x) for x in model_struct)+'_lb_'+str(look_back)
    # create some data
    #X = np.linspace(-3.14, 3.14, 1200)
    #np.random.shuffle(X)    # randomize the data
    #Y = 6*np.sin(X)+6*np.cos(2*X)+ 2
    #np.random.normal(0, 0.05, (1200, ))
    
    # plot data
    #plt.scatter(csv_data[0:41,0],csv_data[0:41,1],c='red',label='train')
    #plt.show()


    '''
    X_train, Y_train = X[:1160], Y[:1160]     # first 160 data points
    X_test, Y_test = X[1160:], Y[1160:]       # last 40 data points
    '''
    model = build_cudnlstm_model(model_struct,X_train_nor.shape[1:],y_train.shape[-1])

    if train_b:
        #plot setting
        #plt.ion()
        plt.figure(1)
        xlabel_ar=['data number']*8
        #ylabel_ar=['x','z','x_dot','z_dot','theta','phi','theta_dot','phi_dot']
        ylabel_ar=['x','z','x_dot','z_dot','x2','z2','x_dot2','z_dot2']
        xlabel_com = 'x'
        ylabel_com = 'z'
        #use for save pic
        save_count=0
        #plot model
        os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'
        
        plot_model(model, to_file='model_plot\\%s.png'%model_name)
        #add folder
        if not os.path.exists('train_result\\%s'%model_name):
            os.makedirs('train_result\\%s'%model_name)
        # training
        print('Training ----------')
        print(X_train_nor.shape)
        callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
        history = model.fit(x=X_train_nor,y=y_train,batch_size=32,epochs =500,validation_data=[X_test_nor,y_test],use_multiprocessing= True,callbacks=[callback])
        #history = model.fit(x=X_train_nor,y=y_train,batch_size=32,epochs =300,validation_data=(X_test_nor, y_test),callbacks=[callback])
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        #save final model
        model.save_weights('train_model\\%s.h5'%model_name)
    else:
        model.load_weights('train_model\\%s.h5'%model_name)


    for sim_data,pred_data,IC in zip(sim_data_set,pred_data_set,IC_test):
        if plot_anim:
            plt.figure()
        title =('test_%s_IC%s'%(model_name,IC))

        temp = np.zeros(sim_data.shape)
        temp[0:look_back] = sim_data[0:look_back]
        for i in range(sim_data.shape[0]-look_back):
            pred_input = data_lookback(temp[i:i+look_back],look_back)
            temp[i+look_back:i+look_back+1] = model.predict(pred_input)
            '''
            print('i',i,'in',pred_input)
            print('out',temp[i+look_back:i+look_back+1])
            print('truth',sim_data[i+look_back:i+look_back+1])
            '''
            if plot_anim:
                plt.suptitle(title)
                plt.xlabel('t * 0.05')
                plt.ylabel('I (A)')
                plt.plot(np.arange(sim_data.shape[0]),sim_data,color='black')
                plt.plot(np.arange(temp[:i+look_back+1].shape[0]),temp[:i+look_back+1],color='blue')
                plt.scatter(np.arange(pred_input[0].shape[0])+i,pred_input[0],color = 'red')
                plt.scatter(np.arange(temp[i+look_back:i+look_back+1].shape[0])+look_back+i,temp[i+look_back:i+look_back+1],color = 'b')
                plt.scatter(np.arange(sim_data[i+look_back:i+look_back+1].shape[0])+look_back+i,sim_data[i+look_back:i+look_back+1],color = 'g')
                plt.pause(0.02)
                plt.cla()
        plt.show()
            
        
        plt.figure()
        plot_NN = plt.scatter(np.arange(temp.shape[0]),temp)#NN iteration
        plot_exp = plt.scatter(np.arange(sim_data.shape[0]),sim_data,color='red')#sim data
        #plt.savefig('test_com\\%s.png'%model_name)
        title =('test_%s_IC%s'%(model_name,IC))
        plt.suptitle(title)
        plt.xlabel('t * 0.05')
        plt.ylabel('I (A)')
        if not os.path.exists('test_pic\\%s'%model_name):
            os.makedirs('test_pic\\%s'%model_name)
        plt.savefig('test_pic\\%s\\%s.png'%(model_name,title))
        plt.show()
        plt.close()
    '''
    #plt.close()
    #pred 40 step
    i = 0 #row now
    j = 0 #row step start
    temp_40 = np.zeros(( pred_data.shape ))
    while i<len(pred_data[:,0]) - look_back+1:
        plt.figure()
        j = i
        temp_40[j:j+look_back+1,:] = pred_data[j:j+look_back+1,:].copy()
        while i<len(pred_data[:,0]):
            
            #print(model.predict(temp_40[i:i+1,:]))
            #print('out')
            #temp_40[i+1:i+2,:] = model.predict(temp_40[i:i+1,:])
            pred_input = []
            for m in range(look_back):
                #print('in',normalize_data(temp_40[i+m],mean,std))
                print('in',temp_40[i+m])
                if normalize:
                    #pred_input.append(normalize_data(temp_40[i+m],mean,std))
                    pred_input.append(temp_40[i+m])
                else:
                    pred_input.append(temp_40[i+m])
            pred_input = create_dataset(np.array(pred_input), look_back )[0]
            temp_40[i+look_back:i+look_back+1,:] = model.predict(pred_input)
            print('out',temp_40[i+look_back:i+look_back+1,:])
            try:
                if not csv_test[i+look_back ,44] == i+look_back+1-j:
                    print(csv_test[i+look_back ,44],i+look_back+1-j)
                    i = i+look_back
                    break
            except IndexError:
                print('IndexError')
                i = i+look_back
                break
            i=i+1
        
        print('j=%d i=%d'%(j,i))
        print('test_a=%s v=%s test=%s step_%s N=%s'%(csv_test[j,40],csv_test[j,41],csv_test[j,42],csv_test[j,43],csv_test[j,44]))
        #========plot inverse norm or not =======
        if plot_inverse:
            temp_40[j:i,:] = inv_normalize_data(temp_40[j:i,:],mean,std)
            pred_data[j:i,:] = inv_normalize_data(pred_data[j:i,:],mean,std)
            sim_data[j:i,:] = inv_normalize_data(sim_data[j:i,:],mean,std)
        plot_NN = plt.scatter(temp_40[j:i,0],temp_40[j:i,1])#NN iteration
        plot_exp = plt.scatter(pred_data[j:i,0],pred_data[j:i,1],color='red')#exp data
        plot_ode = plt.scatter(sim_data[j:i,0],sim_data[j:i,1] - sim_data[j,1]+temp_40[j,1] ,color='olive')#ode 45 onestep sim
        plt.legend([plot_NN, plot_exp, plot_ode], ['NN', 'exp' , 'ode'])
        #plt.legend([plot_NN, plot_exp], [ 'NN' , 'exp'])
        plt.xlabel('x')
        plt.ylabel('z')
        #plt.savefig('test_com\\%s.png'%model_name)
        title =('test_a=%s v=%s test=%s step_%s'%(csv_test[j,40],csv_test[j,41],csv_test[j,42],csv_test[j,43]))
        plt.suptitle(title)
        if not os.path.exists('test_com\\%s'%model_name):
            os.makedirs('test_com\\%s'%model_name)
        plt.savefig('test_com\\%s\\%s.png'%(model_name,title))
        plt.show()
        plt.close()
        '''

def build_cudnlstm_model(model_struct,input_size,output_size):
    #build a neural network from the 1st layer to the last layer
    model = Sequential()
    model.add(LSTM(2**model_struct[0],input_shape = input_size, return_sequences = True , activation=activations.tanh))
    model.add(Dropout(0.2))
    for i in model_struct[1:-1]:
        model.add(LSTM(2**i,return_sequences = True,activation=activations.tanh))
        model.add(Dropout(0.2))
    model.add(LSTM(2**model_struct[-1],return_sequences = False,activation=activations.tanh))
    model.add(Dropout(0.2))
    #model.add(Dense(2**model_struct[0], input_dim=input_size,activation=activations.relu))
    '''
    for i in model_struct[1:]:
        model.add(Dense(2**i))
        #model.add(LSTM(32))
    '''
    '''
    model.add(Dense(32,activation=activations.relu))
    model.add(Dense(64,activation=activations.relu))
    model.add(Dense(128,activation=activations.relu))
    model.add(Dense(64,activation=activations.relu))
    model.add(Dense(16,activation=activations.relu))
    '''
    #model.add(Dense(64,activation=activations.tanh))
    model.add(Dense(32,activation=activations.tanh))
    
    model.add(Dense(output_size))
    
   

    ad_op = optimizers.Adam(learning_rate=0.0011, decay=0.0001)#, amsgrad=False
    #model.add(Dense(units=1, input_dim=1)) 
    # choose loss function and optimizing method
    model.compile(loss='mse', optimizer=ad_op,metrics=['accuracy'])
    return model

#%%
def data_lookback(dataset, look_back=1):
    if len(dataset.shape) == 1:
        dataset = dataset.reshape(-1,1)
        print(dataset)
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back+1): 
        a = dataset[i:(i+look_back), :] 
        dataX.append(a) 
        #dataY.append(dataset[i + look_back - 1, :]) 
    return np.array(dataX)

#%%
def ode_predict(t,I0,L=0.075,R=0.1):
    #L = 7.50 mH
    #R = 3 ohm
    return I0*np.exp(-t/L*R)
    #return np.exp(-t/L*I0*0.1)

# %%
def dy_dx(y, x):
    return x - y
#%%
def normalize_data(Matrix,mean,std):
    return (Matrix-mean)/std
def inv_normalize_data(Matrix,mean,std):
    return Matrix*std+ mean
#%%
def create_dataset(IC_array,time_delta = 0.75*2/200,time_data_num = 200):
    time_data_set = []
    time_init = 0
    #add data
    for I0 in IC_array:   
        timestamp = np.arange(time_init,time_init+time_data_num)*time_delta
        time_data = ode_predict( timestamp, I0).reshape(-1,ode_predict(np.array([0]),I0).shape[-1])    
        time_data_set.append(time_data)
    return np.array(time_data_set)

#%%
def dataset_arrange(time_data_set,lookback = 3):
    
    output_size = 1
    data_set = []
    y_data = []
    
    #create lookback
    for time_data in time_data_set:
        data_set.append(data_lookback(time_data[:-1],lookback))
        y_data.append( time_data[lookback::] )
    
    #list to np
    data_set = np.array(data_set).reshape(-1,lookback,output_size)
    y_data = np.array(y_data).reshape(-1,output_size)

    return data_set, y_data
#%%
def main():
    #========user val===========
    IC = np.repeat(np.arange(1,30)*0.1,2,axis=0)
    #IC = np.append(IC,np.arange(1,40)*0.03)

    look_back = 10
    noise_scale = 0.00045#0.007#0.0015
    global train_b,IC_test,plot_anim
    train_b = False
    plot_anim =False
    IC_test = np.array([0.1,0.2,0.4,0.6,1.0,1.2 ,0.5,0.7,1.3])
    #IC_test = np.array([1.1,1.2,1.4,1.6,2.0,2.2 ,2.5,2.7,2.3])
    
    #=====data handle ====
    time_data_set = create_dataset(IC)
    #add noise
    time_data_set += np.random.normal(0,scale=noise_scale,size=time_data_set.shape)    
    #=normalize datasss
    global mean,std
    mean = np.mean(time_data_set,axis=1)[0]
    std = np.std(time_data_set,axis=1)[0]

    for i in range(time_data_set.shape[0]):
        time_data_set[i]=normalize_data(time_data_set[i],mean,std)
    #plot time_data_set
    for data in time_data_set:
        plt.plot(np.arange(data.shape[0]),data)
    plt.title('Dif. R fot RL circuit')
    plt.xlabel('t * 0.05')
    plt.ylabel('I (A)')
    plt.show()

    data_x_nor,data_y_nor = dataset_arrange(time_data_set,look_back)
    '''
    plt.figure()
    for x,y in zip(data_x_nor,data_y_nor):
        plt.scatter(np.arange(x.shape[0]),x,color = 'r')
        plt.scatter(np.arange(y.shape[0])+look_back,y,color = 'b')
        plt.pause(0.1)
        plt.cla()
    plt.show()
    '''
    #split data to train vali data
    X_train_nor, X_test_nor, y_train_nor, y_test_nor =model_selection.train_test_split(data_x_nor,data_y_nor,test_size=0.2)
    

    

    
    #create test data
    sim_data_set = create_dataset(IC_test)
    sim_data_set +=  np.random.normal(0,scale=noise_scale,size=sim_data_set.shape)
    for i in range(sim_data_set.shape[0]):
        sim_data_set[i] = normalize_data(sim_data_set[i],mean,std)
    pred_data_set = sim_data_set[:,:look_back,:].copy()
    '''
    for s,p in zip(sim_data_set,pred_data_set):
        plt.figure()
        plt.scatter(np.arange(s.shape[0]),s,color = 'r')
        plt.scatter(np.arange(p.shape[0])+look_back,p,color = 'b')
        plt.show()
    '''
    train_NN(X_train_nor, X_test_nor, y_train_nor, y_test_nor,pred_data_set,sim_data_set,mean,std,look_back)

#%%

if __name__ == "__main__":
    main()


