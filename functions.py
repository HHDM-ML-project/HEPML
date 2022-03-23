import sys
import numpy as np
import pandas as pd
import os
import concurrent.futures as cf
from operator import itemgetter
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, PReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
#from tensorflow.keras.models import load_model as tf_load_model
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
pd.set_option('display.expand_frame_repr', False)
plt.style.use('classic')
from tqdm import tqdm
import json
import h5py


"""
-> model training minimize classification and domain at the same time (affect all weights)
-> The weights of the domain branch are recovered
-> domain model training minimize domain (classification branch weights are not affected)
-> The weights of the comum branch are recovered (the weights of the classification model are effectivily updated in the "model training")

-> The target samples don't influence the classification training (sample_weights)
-> During the model training, the weights are modified to improved the accuaracy of the domain and classification prediction
-> During the domain model training, the weights are modified to make the network doesn't be able to distinguish the domains
"""

#=====================================================================================================================
def read_files(basedir, period, mode="normal", features=[]):
    
    print("")
    print("Loading datasets...")
    
    datasets_dir = os.path.join(basedir, period)
    datasets_abspath = [(f, os.path.join(datasets_dir, f)) for f in os.listdir(datasets_dir)]
    datasets = {}

    for dataset, abspath in tqdm(datasets_abspath):
        dataset_name = dataset.split(".")[0]
        
        if mode == "normal" or mode == "scalars":
            if dataset.endswith(".h5"):
                variables_dict = {}
                f = h5py.File(abspath, "r")
                if "scalars" in f.keys():
                    group = "scalars"
                    for variable in f[group].keys():
                        if len(features) == 0:
                            variables_dict[variable] = np.array(f[group+"/"+variable])
                        elif len(features) > 0: 
                            if variable in features:
                                variables_dict[variable] = np.array(f[group+"/"+variable])
                    if mode == "normal":
                        datasets[dataset_name] = pd.DataFrame(variables_dict)
                    if mode == "scalars":
                        datasets[dataset_name] = variables_dict
                else:
                    print("Warning: Dataset " + dataset_name + " is empty!")
                    
        if mode == "vectors":
            if dataset.endswith(".h5"):
                variables_dict = {}
                f = h5py.File(abspath, "r")
                if "vectors" in f.keys():
                    group = "vectors"
                    for variable in f[group].keys():
                        if len(features) == 0:
                            variables_dict[variable] = np.array(f[group+"/"+variable])
                        elif len(features) > 0: 
                            if variable in features:
                                variables_dict[variable] = np.array(f[group+"/"+variable])
                    datasets[dataset_name] = variables_dict  
                else:
                    print("Warning: Dataset " + dataset_name + " is empty!")
                
        if mode == "metadata":
            if dataset.endswith(".h5"):
                variables_dict = {}
                f = h5py.File(abspath, "r")
                group = "metadata"
                for variable in f[group].keys():
                    if len(features) == 0:
                        variables_dict[variable] = np.array(f[group+"/"+variable])
                    elif len(features) > 0: 
                        if variable in features:
                            variables_dict[variable] = np.array(f[group+"/"+variable])
                datasets[dataset_name] = variables_dict        
                
        if mode == "syst":
            if dataset.endswith(".json"):
                with open(abspath) as json_file:
                    datasets[dataset_name] = json.load(json_file)

    return datasets


#=====================================================================================================================
def join_datasets(ds, new_name, input_list, mode="normal"):
    
    datasets_list = []
    for input_name in input_list:
        datasets_list.append(ds[input_name])

    good_list = False
    if mode == "normal":
        ds[new_name] = pd.concat(datasets_list).reset_index(drop=True)
        good_list = True
    elif mode == "syst":
        ds[new_name] = datasets_list
        good_list = True
    elif mode == "scalars" or mode == "vectors":
        ds[new_name] = {}
        first = True
        for dataset in datasets_list:
            if first:
                for variable in dataset.keys():
                    ds[new_name][variable] = dataset[variable].copy()
            else:
                for variable in dataset.keys():
                    if mode == "vectors":
                        out_size = len(ds[new_name][variable][0])
                        dataset_size = len(dataset[variable][0])
                        diff_size = abs(out_size-dataset_size)
                        if out_size > dataset_size:
                            number_of_events = len(dataset[variable])
                            for i in range(diff_size):
                                dataset[variable] = np.c_[ dataset[variable], np.zeros(number_of_events) ]
                        elif dataset_size > out_size:
                            number_of_events = len(ds[new_name][variable])
                            for i in range(diff_size):
                                ds[new_name][variable] = np.c_[ ds[new_name][variable], np.zeros(number_of_events) ]
                    ds[new_name][variable] = np.concatenate((ds[new_name][variable],dataset[variable]))
            first = False
        good_list = True
        
        
    
        
        
    else:
        print("Type of the items is not supported!")
    
    if good_list:
        for input_name in input_list:
            del ds[input_name]
    
    del datasets_list
    
    
#======================================================================================================================    
def step_plot( ax, var, dataframe, label, color='black', weight=None, error=False, normalize=False, bins=np.linspace(0,100,5), linestyle='solid', overflow=False, underflow=False ):
    

    if weight is None:
        W = None
        W2 = None
    else:
        W = dataframe[weight]
        W2 = dataframe[weight]*dataframe[weight]

    eff_bins = bins[:]
    if overflow:
        eff_bins[-1] = np.inf
    if underflow:
        eff_bins[0] = -np.inf
    
    counts, binsW = np.histogram(
        dataframe[var], 
        bins=eff_bins, 
        weights=W
    )
    yMC = np.array(counts)
    
    countsW2, binsW2 = np.histogram(
        dataframe[var], 
        bins=eff_bins, 
        weights=W2
    )
    errMC = np.sqrt(np.array(countsW2))
    
    if normalize:
        if weight is None:
            norm_factor = len(dataframe[var])
        else:
            norm_factor = dataframe[weight].sum()
        yMC = yMC/norm_factor
        errMC = errMC/norm_factor
    
    ext_yMC = np.append([yMC[0]], yMC)
    
    plt.step(bins, ext_yMC, color=color, label=label, linewidth=1.5, linestyle=linestyle)
    
    if error:
        x = np.array(bins)
        dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
        x = x[:-1]
        
        ax.errorbar(
            x+0.5*dx, 
            yMC, 
            yerr=[errMC, errMC], 
            fmt=',',
            color=color, 
            elinewidth=1
        ) 
    
    return yMC, errMC
    
    
#======================================================================================================================
def ratio_plot( ax, ynum, errnum, yden, errden, bins=np.linspace(0,100,5), color='black', numerator="data" ):
    x = np.array(bins)
    dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
    x = x[:-1]
    yratio = np.zeros(ynum.size)
    yeratio = np.zeros(ynum.size)
    y2ratio = np.zeros(ynum.size)
    ye2ratio = np.zeros(ynum.size)
    for i in range(ynum.size):
        if yden[i] == 0:
            yratio[i] = 99999
            yeratio[i] = 0
            ye2ratio[i] = 0
        else:
            yratio[i] = ynum[i]/yden[i]
            yeratio[i] = errnum[i]/yden[i]
            y2ratio[i] = yden[i]/yden[i]
            ye2ratio[i] = errden[i]/yden[i]
            
    if numerator == "data":
        yl = (yden - errden)/yden
        yh = (yden + errden)/yden
        dy = yh - yl
        pats = [ pat.Rectangle( (x[i], yl[i]), dx[i], dy[i], hatch='/////', fill=False, linewidth=0, edgecolor='grey' ) for i in range(len(x)-1) ]
        pats.append(pat.Rectangle( (x[len(x)-1], yl[len(x)-1]), dx[len(x)-1], dy[len(x)-1], hatch='/////', fill=False, linewidth=0, edgecolor='grey' ))
        for p in pats:
            ax.add_patch(p) 
    
        ax.axhline(1, color='red', linestyle='-', linewidth=0.5)
    
        ax.errorbar(x+0.5*dx, yratio, yerr=[yeratio, yeratio], xerr=0.5*dx, fmt='.', ecolor='black', color='black', elinewidth=0.7, capsize=0)
    elif numerator == "mc":
        ax.errorbar(x+0.5*dx, y2ratio, yerr=[ye2ratio, ye2ratio], xerr=0.5*dx, fmt=',', ecolor="red", color="red", elinewidth=1.2, capsize=0)
    
        ax.errorbar(x+0.5*dx, yratio, yerr=[yeratio, yeratio], xerr=0.5*dx, fmt=',', ecolor=color, color=color, elinewidth=1.2, capsize=0)
    
    return yratio
  
    

#=====================================================================================================================
def build_DANN(parameters, n_var, n_classes):
    #Creates three different models, one used for source only training, two used for domain adaptation
    
    # Base network -> x4
    for i in range(len(parameters[0])):
        if i == 0:
            inputs = Input(shape=(n_var,)) 
            x4 = Dense(parameters[0][i], activation=parameters[1])(inputs)
        if i > 0:
            x4 = Dense(parameters[0][i], activation=parameters[1])(x4)
    
    if parameters[3] == 'mixed':
        activ_source = 'softmax'
        activ_domain = 'sigmoid'
    else:
        activ_source = parameters[3]
        activ_domain = parameters[3]
    
    # Source network
    class_discriminator = Dense(n_classes, activation=activ_source, name="class")(x4)  
    
    # Domain network
    #domain_discriminator = Dense(2, activation=activ_domain, name="domain")(x4)
    domain_discriminator = Dense(2, activation=activ_domain, name="domain")(class_discriminator)

    # Full model
    comb_model = Model(inputs=inputs, outputs=[class_discriminator, domain_discriminator])
    comb_model.compile(optimizer=parameters[2], loss={"class": 'categorical_crossentropy', "domain": 'categorical_crossentropy'}, loss_weights={"class": 1, "domain": 2}, metrics=['accuracy'], )

    # Source model
    class_discriminator_model = Model(inputs=inputs, outputs=[class_discriminator])
    class_discriminator_model.compile(optimizer=parameters[2], loss={"class": 'categorical_crossentropy'}, metrics=['accuracy'], )

    # Domain model
    domain_discriminator_model = Model(inputs=inputs, outputs=[domain_discriminator])
    domain_discriminator_model.compile(optimizer=parameters[2], loss={"domain": 'categorical_crossentropy'}, metrics=['accuracy'])
                        
    return comb_model, class_discriminator_model, domain_discriminator_model


#=====================================================================================================================
def build_NN(parameters, n_var, n_classes):
    #Creates three different models, one used for source only training, two used for domain adaptation
    
    # Base network -> x4
    for i in range(len(parameters[0])):
        if i == 0:
            inputs = Input(shape=(n_var,)) 
            x4 = Dense(parameters[0][i], activation=parameters[1])(inputs)
        if i > 0:
            x4 = Dense(parameters[0][i], activation=parameters[1])(x4)
    
    # Source network
    class_discriminator = Dense(n_classes, activation=parameters[3], name="class")(x4)  

    # Source model
    class_discriminator_model = Model(inputs=inputs, outputs=[class_discriminator])
    class_discriminator_model.compile(optimizer=parameters[2], loss={"class": 'categorical_crossentropy'}, metrics=['accuracy'], )

                        
    return class_discriminator_model

#=====================================================================================================================
def batch_generator(data, batch_size):
    #Generate batches of data.

    #Given a list of numpy data, it iterates over the list and returns batches of the same size
    #This
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr
        
        
        
#=====================================================================================================================
def train_model(train_x, train_y, train_w, test_x, test_y, test_w, source_x, source_w, target_x, target_w, parameters, n_var, n_classes, n_iterations = 5000, batch_size = 100, model_type = "NN"):
    
    #---------------------------------------------------------------------------------------
    # DANN training
    #---------------------------------------------------------------------------------------
    if model_type == "DANN":
        # Create models specifying the number of neurons of the first layer
        model, class_discriminator_model, domain_discriminator_model = build_DANN(parameters, n_var, n_classes)

        # Correct classification fo the domains to be used in the full model training: source = 1 and target = 0: (MC-CR <-> Data-CR) or (MC-SR <-> MC-CR)
        y_adversarial = to_categorical(np.array([0]*batch_size + [1]*batch_size + [0]*batch_size), num_classes=2)

        # Inverse classification of the domains to be used in the domain model confusion training 
        y_adv = to_categorical(np.array([0]*batch_size + [1]*batch_size), num_classes=2)

        # Create batch samples
        train_batches = batch_generator([train_x, to_categorical(train_y, num_classes=n_classes), train_w], batch_size)
        source_batches = batch_generator([source_x, source_w], batch_size)
        target_batches = batch_generator([target_x, target_w], batch_size)
    
        interation = []
        train_acc = []
        test_acc = []
        train_loss = []
        test_loss = []
        adv_source_acc = []
        adv_target_acc = []
    
        best_weights = []
        position = 0
        min_loss = 99999
        for i in range(n_iterations):
        
            train_x_b, train_y_b, train_w_b = next(train_batches)
            source_x_b, source_w_b = next(source_batches)
            target_x_b, target_w_b = next(target_batches)
        
            # Input for the training ([train events, source events, target events])
            x_full = np.concatenate([train_x_b, source_x_b, target_x_b])
            x_adv = np.concatenate([source_x_b, target_x_b])
        
            # Class of the events in the source. Target events are set to have y = 0 but they don't contribute in the source classification as the weights are 0 for the full model training
            y_class = np.concatenate([train_y_b, np.zeros_like(train_y_b), np.zeros_like(train_y_b)])
        
            # In the full model, all events have weight 1 for the domain cassification but only source events have weight 1 for the class discrimination, the target events have weight 0 and don't participate
        
            sample_weights_class = np.concatenate([train_w_b, np.zeros_like(train_w_b), np.zeros_like(train_w_b)])
            sample_weights_adversarial = np.concatenate([np.zeros_like(train_w_b), source_w_b, target_w_b])
            sample_weights_adv = np.concatenate([source_w_b, target_w_b])

            # Save weights of the domain branch
            adv_weights = []
            for layer in model.layers:
                if (layer.name.startswith("domain")):
                    adv_weights.append(layer.get_weights())

            
            # Train full model to learn class and domain prediction
            stats = model.train_on_batch(x_full, [y_class, y_adversarial], sample_weight=[sample_weights_class, sample_weights_adversarial])
        
            test_acc_i = []
            adv_source_test_acc_i = []
            adv_target_test_acc_i = []
            if ((i + 1) % 10 == 0):
                train_acc_i = class_discriminator_model.evaluate(train_x,to_categorical(train_y, num_classes=n_classes), sample_weight=train_w, verbose = 0)
                test_acc_i = class_discriminator_model.evaluate(test_x,to_categorical(test_y, num_classes=n_classes), sample_weight=test_w, verbose = 0)
                adv_source_acc_i = domain_discriminator_model.evaluate(source_x,to_categorical(np.ones_like(source_w),  num_classes=2), sample_weight=source_w, verbose = 0)
                adv_target_acc_i = domain_discriminator_model.evaluate(target_x,to_categorical(np.zeros_like(target_w), num_classes=2), sample_weight=target_w, verbose = 0)
                
                interation.append(i+1)
                train_acc.append(train_acc_i[1])
                test_acc.append(test_acc_i[1]) 
                train_loss.append(train_acc_i[0])
                test_loss.append(test_acc_i[0]) 
                adv_source_acc.append(adv_source_acc_i[1])
                adv_target_acc.append(adv_target_acc_i[1])
               
                if( (test_acc_i[0] < min_loss) ):
                    min_loss = test_acc_i[0]
                    position = i+1
                    # Save best weights
                    best_weights[:] = []
                    for layer in model.layers:
                        best_weights.append(layer.get_weights())
               
            # Recover weights of the domain branch
            k = 0
            for layer in model.layers:
                if (layer.name.startswith("domain")):
                    layer.set_weights(adv_weights[k])
                    k += 1

            # Save weights of the comum and classification branches
            class_weights = []
            for layer in model.layers:
                if (not layer.name.startswith("domain")):
                    class_weights.append(layer.get_weights())
            
            # Train Domain model to confuse the domain classification
            stats2 = domain_discriminator_model.train_on_batch(x_adv, y_adv, sample_weight=sample_weights_adv)

            # Recover weights of the comum and classification branches
            k = 0
            for layer in model.layers:
                if (not layer.name.startswith("domain")):
                    layer.set_weights(class_weights[k])
                    k += 1

            if ((i + 1) % 10 == 0):
                test_eval = class_discriminator_model.evaluate(test_x,to_categorical(test_y, num_classes=n_classes), sample_weight=test_w, verbose = 0)
                target_eval = domain_discriminator_model.evaluate(target_x,to_categorical(np.zeros_like(target_w), num_classes=2), sample_weight=target_w, verbose = 0)
                source_eval = domain_discriminator_model.evaluate(source_x,to_categorical(np.ones_like(source_w), num_classes=2), sample_weight=source_w, verbose = 0)
                print("Iteration %d, class loss =  %.10f, class accuracy =  %.3f, domain accuracies sum =  %.5f"%(i, test_eval[0], test_eval[1], target_eval[1]+source_eval[1] ))
        
        if( position > 0 ):
            # Set weights of the best classification model
            k = 0
            for layer in model.layers:
                layer.set_weights(best_weights[k])
                k += 1
    
    #---------------------------------------------------------------------------------------
    # NN training
    #---------------------------------------------------------------------------------------        
    if model_type == "NN":
        # Create models specifying the number of neurons of the first layer
        class_discriminator_model = build_NN(parameters, n_var, n_classes)

        # Create batch samples
        train_batches = batch_generator([train_x, to_categorical(train_y, num_classes=n_classes), train_w], batch_size)
    
        interation = []
        train_acc = []
        test_acc = []
        train_loss = []
        test_loss = []
    
        best_weights = []
        position = 0
        min_loss = 99999
        early_stopping_count = 0
        for i in range(n_iterations):
        
            train_x_b, train_y_b, train_w_b = next(train_batches)
        
            # Train model to learn class
            stats = class_discriminator_model.train_on_batch(train_x_b, train_y_b, sample_weight=train_w_b)
        
            test_acc_i = []
            if ((i + 1) % 10 == 0):
                train_acc_i = class_discriminator_model.evaluate(train_x,to_categorical(train_y, num_classes=n_classes), sample_weight=train_w, verbose = 0)
                test_acc_i = class_discriminator_model.evaluate(test_x,to_categorical(test_y, num_classes=n_classes), sample_weight=test_w, verbose = 0)
                
                interation.append(i+1)
                train_acc.append(train_acc_i[1])
                test_acc.append(test_acc_i[1]) 
                train_loss.append(train_acc_i[0])
                test_loss.append(test_acc_i[0]) 
               
                if( (test_acc_i[0] < min_loss) ):
                    min_loss = test_acc_i[0]
                    position = i+1
                    # Save best weights
                    best_weights[:] = []
                    for layer in class_discriminator_model.layers:
                        best_weights.append(layer.get_weights())
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1
        
                print("Iterations %d, class loss =  %.10f, class accuracy =  %.3f"%(i+1, test_acc_i[0], test_acc_i[1] ))
            
                if early_stopping_count == 10:
                    print("Early stopping activated!")
                    break
                
        if( position > 0 ):
            # Set weights of the best classification model
            k = 0
            for layer in class_discriminator_model.layers:
                layer.set_weights(best_weights[k])
                k += 1
    
        adv_source_acc = np.zeros_like(test_acc)
        adv_target_acc = np.zeros_like(test_acc)
    
    
    #plot_model(model, "plots/combined_model.pdf", show_shapes=True)
    #plot_model(class_discriminator_model, "plots/class_discriminator_model.pdf", show_shapes=True)
    #plot_model(domain_discriminator_model, "plots/domain_discriminator_model.pdf", show_shapes=True)
            
            
    return class_discriminator_model, np.array(interation), np.array(train_acc), np.array(test_acc), np.array(train_loss), np.array(test_loss), np.array(adv_source_acc), np.array(adv_target_acc)


