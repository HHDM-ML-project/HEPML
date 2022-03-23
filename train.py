import sys
import numpy as np
import pandas as pd
import os
import time
import concurrent.futures as cf
import argparse
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.ticker import AutoMinorLocator
import json
from functions import read_files
from functions import join_datasets
from functions import train_model
from functions import step_plot

parser = argparse.ArgumentParser()
parser.add_argument("-j", "--job", type=int, default=0)
parser.add_argument("-s", "--signal", default="Signal_1000_100")
parser.add_argument("--check", dest='check_flag', action='store_true')
parser.set_defaults(check_flag=False)
args = parser.parse_args()

#=============================DATASETS=============================================================
selection = "ML"
period = '17'

datasets = read_files(os.path.join(os.environ.get("HEP_OUTPATH"), "HHDM", selection, "datasets"), period)

join_datasets(datasets, "Bkg", [
    "DYJetsToLL_Pt-Inclusive",
    "TTTo2L2Nu", 
    "TTToSemiLeptonic",
    "ST_tW_antitop", 
    "ST_tW_top", 
    #"ST_s-channel",
    "ST_t-channel_top", 
    "ST_t-channel_antitop",
    "WZZ", 
    "WWZ", 
    "ZZZ", 
    "WWW", 
    "ZGToLLG", 
    "TTGamma", 
    "TTGJets", 
    "WW",
    "WZ",
    "ZZ",
    ])


#=============================CLASSES AND INPUT VARIABLES==========================================
classes = [args.signal, "Bkg"]
class_names = [args.signal, 'Background']
colors = ['green', 'red']

# Example with more than 2 classes:
#classes = [args.signal, "DYJetsToLL", "TT", "Others"]
#class_names = [args.signal, 'Drell-Yan', r'$t\bar{t}$', "Others"]
#colors = ['green', 'darkgoldenrod', 'skyblue', 'grey']


variables = ["LeadingLep_pt", "LepLep_pt", "LepLep_deltaR", "LepLep_deltaM", "MET_pt", "MET_LepLep_Mt", "MET_LepLep_deltaPhi"]

var_names = [r"$\mathrm{leading}\,p_\mathrm{T}^\mathrm{l}$", r"$p_\mathrm{T}^\mathrm{ll}$", r"$\Delta R^\mathrm{ll}$", r"$\Delta M^\mathrm{ll}$", r"$E_\mathrm{T}^\mathrm{miss}$", r"$M_\mathrm{T}^\mathrm{ll,MET}$", r"$\Delta \phi^\mathrm{ll,MET}$"]


#=============================MODELS AND TRAINING==================================================
train_frac = 0.5
n_iterations = 2000 #20000
batch_size = 500

modelName = [    
"NN_1_20_elu_adam_soft",
"NN_1_50_elu_adam_soft",
"NN_1_100_elu_adam_soft",
"NN_2_20_elu_adam_soft",
"NN_2_50_elu_adam_soft",
"NN_2_100_elu_adam_soft",
"NN_3_20_elu_adam_soft",
"NN_3_50_elu_adam_soft",
"NN_3_100_elu_adam_soft",
]

model = [
[[20], "elu", "adam", "softmax"],
[[50], "elu", "adam", "softmax"],
[[100], "elu", "adam", "softmax"],
[[20,20], "elu", "adam", "softmax"],
[[50,50], "elu", "adam", "softmax"],
[[100,100], "elu", "adam", "softmax"],
[[20,20,20], "elu", "adam", "softmax"],
[[50,50,50], "elu", "adam", "softmax"],
[[100,100,100], "elu", "adam", "softmax"],
]

# The training rotine identify the model using the first patern of the model name. 
# Examlpe: NN, DANN



#--------------------------------------------------------------------------------------------------------------------------------------------------
# [DO NOT TOUCH THIS PART] 
#--------------------------------------------------------------------------------------------------------------------------------------------------

#=====================================================================================================================
# CHECK ARGUMENT
#=====================================================================================================================
#inpath = "files"
n_var = len(variables)

N = int(args.job)
if N == -1:
    sys.exit("Number of jobs: " + str(len(model)))
if N <= -2:
    sys.exit(">> Enter an integer >= -1")
if N >= len(model):
    sys.exit("There are only " + str(len(model)) + " models")

#=====================================================================================================================
# Output setup
#=====================================================================================================================
outpath = os.environ.get("HEP_OUTPATH")
ml_outpath = os.path.join(outpath, "HHDM", selection, "datasets", period, "ML")
if not os.path.exists(ml_outpath):
    os.makedirs(ml_outpath)

signal_outpath = os.path.join(ml_outpath, args.signal)
if not os.path.exists(ml_outpath):
    os.makedirs(ml_outpath)

plots_outpath = os.path.join(signal_outpath, "features")    
if not os.path.exists(plots_outpath):
    os.makedirs(plots_outpath)
    
if not os.path.exists(os.path.join(signal_outpath, "models")):
    os.makedirs(os.path.join(signal_outpath, "models")) 
    
model_outpath = os.path.join(signal_outpath, "models", modelName[int(args.job)])
if not os.path.exists(model_outpath):
    os.makedirs(model_outpath)
    
print('Results will be stored in ' + ml_outpath)


#=====================================================================================================================
# Preprocessing input data
#=====================================================================================================================
print("")
print("Preprocessing input data...")

df = {}
for i in range(len(classes)):
    df[i] = datasets[classes[i]].copy()

df_train = {}
df_test = {}
for key in df.keys():
    dataset = df[key] 
    #dataset = dataset[(dataset["RecoLepID"] < 1000) & (dataset["Nbjets"] > 0)]
    if len(dataset) > 0 :
        dataset = dataset.sample(frac=1)
        dataset = dataset.reset_index(drop=True)
        dataset["class"] = key

        train_limit = int(train_frac*len(dataset))
        df_train_i = dataset.loc[0:(train_limit-1),:].copy()
        df_test_i = dataset.loc[train_limit:,:].copy()
    
        sum_weights = dataset["evtWeight"].sum()
        train_factor = dataset["evtWeight"].sum()/df_train_i["evtWeight"].sum()
        test_factor = dataset["evtWeight"].sum()/df_test_i["evtWeight"].sum()
        df_train_i["evtWeight"] = df_train_i["evtWeight"]*train_factor 
        df_test_i["evtWeight"] = df_test_i["evtWeight"]*test_factor
        df_train_i['mvaWeight'] = df_train_i['evtWeight']/df_train_i['evtWeight'].sum()
        df_test_i['mvaWeight'] = df_test_i['evtWeight']/df_test_i['evtWeight'].sum()

        df_train[key] = df_train_i
        df_test[key] = df_test_i


list_train = [df_train[key] for key in df.keys()]
df_mva = pd.concat(list_train).reset_index(drop=True)

mean = []
std = []
for i in range(len(variables)):
    weighted_stats = DescrStatsW(df_mva[variables[i]], weights=df_mva["mvaWeight"], ddof=0)
    mean.append(weighted_stats.mean)
    std.append(weighted_stats.std)
print("mean: " + str(mean))
print("std: " + str(std))

stat_values={"mean": mean, "std": std}
with open(os.path.join(signal_outpath, "models", 'preprocessing.json'), 'w') as json_file:
    json.dump(stat_values, json_file)

for key in df.keys():
    df_train[key][variables] = (df_train[key][variables] - mean) / std
    df_test[key][variables] = (df_test[key][variables] - mean) / std

control = True
for key in df.keys():
    if control:
        df_full_train = df_train[key].copy()
        df_full_test = df_test[key].copy()
        control = False
    else:
        df_full_train = pd.concat([df_full_train, df_train[key]])
        df_full_test = pd.concat([df_full_test, df_test[key]])

#df_full_train.to_csv("files/train.csv", index=False)
#df_full_test.to_csv("files/test.csv", index=False)
#del df_full_train, df_full_test


#=====================================================================================================================
# Training and test distributions
#=====================================================================================================================
for ivar in range(len(variables)):
    
    fig1 = plt.figure(figsize=(10,7))
    gs1 = gs.GridSpec(1, 1)
    #==================================================
    ax1 = plt.subplot(gs1[0])            
    #==================================================
    var = variables[ivar]
    bins = np.linspace(-2.5,2.5,51)
    for key in df.keys():
        step_plot( ax1, var, df_train[key], label=class_names[key]+" (train)", color=colors[key], weight="mvaWeight", bins=bins, error=True )
        step_plot( ax1, var, df_test[key], label=class_names[key]+" (test)", color=colors[key], weight="mvaWeight", bins=bins, error=True, linestyle='dotted' )
    ax1.set_xlabel(var_names[ivar], size=14, horizontalalignment='right', x=1.0)
    ax1.set_ylabel("Events normalized", size=14, horizontalalignment='right', y=1.0)
    
    ax1.tick_params(which='major', length=8)
    ax1.tick_params(which='minor', length=4)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.margins(x=0)
    ax1.legend(numpoints=1, ncol=2, prop={'size': 10.5}, frameon=False)

    plt.savefig(os.path.join(plots_outpath, var + '.png'))

if args.check_flag:    
    sys.exit()


#=====================================================================================================================
# Load Datasets
#=====================================================================================================================

#df_full_train = pd.read_csv(os.path.join(inpath,"train.csv"))
df_full_train = df_full_train.sample(frac=1)
train_x = df_full_train[variables]
train_x = train_x.values
train_y = np.array(df_full_train['class']).ravel()
train_w = np.array(df_full_train['mvaWeight']).ravel()                    # weight to signal x bkg comparison
print("Variables shape = " + str(train_x.shape))
print("Labels shape = " + str(train_y.shape))
print("Weights shape = " + str(train_w.shape))

#df_full_test = pd.read_csv(os.path.join(inpath,"test.csv"))
df_full_test = df_full_test.sample(frac=1)
test_x = df_full_test[variables]
test_x = test_x.values
test_y = np.array(df_full_test['class']).ravel()
test_w = np.array(df_full_test['mvaWeight']).ravel()                      # weight to signal x bkg comparison

#df_source = pd.read_csv(os.path.join(inpath,"source.csv"))
df_source = df_full_train.copy()
df_source = df_source.sample(frac=1)
source_x = df_source[variables]
source_x = source_x.values
source_w = np.array(df_source['mvaWeight']).ravel()                  # weight to source x target comparison

#df_target = pd.read_csv(os.path.join(inpath,"target.csv"))
df_target = df_full_test.copy()
df_target = df_target.sample(frac=1)
target_x = df_target[variables]
target_x = target_x.values
target_w = np.array(df_target['mvaWeight']).ravel()                  # weight to source x target comparison
         

n_classes = len(df_full_train["class"].unique())
model_type = modelName[N].split("_")[0]

#=====================================================================================================================
# RUN TRAINING
#=====================================================================================================================
print("")
print("Training...")

start = time.time()
        
  
class_model, interation, train_acc, test_acc, train_loss, test_loss, adv_source_acc, adv_target_acc = train_model(
    train_x, 
    train_y, 
    train_w, 
    test_x, 
    test_y, 
    test_w,
    source_x, 
    source_w, 
    target_x, 
    target_w, 
    model[N], 
    n_var,
    n_classes,
    n_iterations = n_iterations, 
    batch_size = batch_size,
    model_type = model_type
    )


#=====================================================================================================================
# SAVE TRAINING INFORMATION AND MODEL
#=====================================================================================================================
df_training = pd.DataFrame(list(zip(interation, train_acc, test_acc, train_loss, test_loss, adv_source_acc, adv_target_acc)),columns=["interation", "train_acc", "test_acc", "train_loss", "test_loss", "adv_source_acc", "adv_target_acc"])

df_training.to_csv(os.path.join(model_outpath, 'training.csv'), index=False)

class_model.save(os.path.join(model_outpath, "model.h5"))    

interation = df_training['interation']
train_acc = df_training['train_acc']
test_acc = df_training['test_acc'] 
train_loss = df_training['train_loss']
test_loss = df_training['test_loss']
adv_source_acc = df_training['adv_source_acc'] 
adv_target_acc = df_training['adv_target_acc']
adv_sum_acc = np.array(df_training['adv_source_acc']) + np.array(df_training['adv_target_acc'])

min_loss = np.amin(test_loss)
position = np.array(interation[test_loss == min_loss])[0]


fig1 = plt.figure(figsize=(18,5))
grid = [1, 2]
gs1 = gs.GridSpec(grid[0], grid[1])
#-----------------------------------------------------------------------------------------------------------------
# Accuracy
#-----------------------------------------------------------------------------------------------------------------
ax1 = plt.subplot(gs1[0])
plt.axvline(position, color='grey')
plt.plot(interation, train_acc, "-", color='red', label='Train (Class Accuracy)')
plt.plot(interation, test_acc, "-", color='blue', label='Test (Class Accuracy)')
#plt.plot(interation, adv_target_acc, "-", color='orange', label='Target (Domain Accuracy)')
#plt.plot(interation, adv_source_acc, "-", color='green', label='Source (Domain Accuracy)')
#plt.plot(interation, adv_sum_acc, "-", color='orchid', label='Sum (Domain Accuracy)')
plt.axhline(1, color='grey', linestyle='--')
ax1.set_xlabel("Interations", size=14, horizontalalignment='right', x=1.0)
ax1.set_ylabel("Accuracy", size=14, horizontalalignment='right', y=1.0)
ax1.tick_params(which='major', length=8)
ax1.tick_params(which='minor', length=4)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
#ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
ax1.grid(which='major', axis='y', linewidth=0.2, linestyle='-', color='0.75')
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['top'].set_linewidth(1)
ax1.spines['left'].set_linewidth(1)
ax1.spines['right'].set_linewidth(1)
ax1.margins(x=0)
ax1.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False, loc='lower right')

#-----------------------------------------------------------------------------------------------------------------
# Loss
#-----------------------------------------------------------------------------------------------------------------
ax2 = plt.subplot(gs1[1])
plt.axvline(position, color='grey')
plt.plot(interation, train_loss, "-", color='red', label='Train (Class Loss)')
plt.plot(interation, test_loss, "-", color='blue', label='Test (Class Loss)')
plt.yscale('log')
ax2.set_xlabel("Interations", size=14, horizontalalignment='right', x=1.0)
ax2.set_ylabel("Loss", size=14, horizontalalignment='right', y=1.0)
ax2.tick_params(which='major', length=8)
ax2.tick_params(which='minor', length=4)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
#ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
ax2.grid(which='major', axis='y', linewidth=0.2, linestyle='-', color='0.75')
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['top'].set_linewidth(1)
ax2.spines['left'].set_linewidth(1)
ax2.spines['right'].set_linewidth(1)
ax2.margins(x=0)
ax2.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False)

plt.subplots_adjust(left=0.055, bottom=0.115, right=0.990, top=0.95, wspace=0.18, hspace=0.165)
plt.savefig(os.path.join(model_outpath, "training.png"))


#=====================================================================================================================
# CHECK OVERTRAINING
#=====================================================================================================================
for key in df.keys():
    train_x = df_train[key][variables]
    train_x = train_x.values
    train_class_pred = class_model.predict(train_x)
    test_x = df_test[key][variables]
    test_x = test_x.values
    test_class_pred = class_model.predict(test_x)
    for i in range(n_classes):
        pred_name = 'prob_C'+str(i)
        df_test[key][pred_name] = test_class_pred[:,i]
        df_train[key][pred_name] = train_class_pred[:,i]
    
    
for i in range(n_classes):
    fig1 = plt.figure(figsize=(20,7))
    gs1 = gs.GridSpec(1,1)
    #==================================================
    ax1 = plt.subplot(gs1[0])            
    #==================================================
    var = 'prob_C'+str(i)
    bins = np.linspace(0,1,201)
    for key in df.keys():
        step_plot( ax1, var, df_train[key], label=class_names[key]+" (train)", color=colors[key], weight="mvaWeight", bins=bins, error=True )
        step_plot( ax1, var, df_test[key], label=class_names[key]+" (test)", color=colors[key], weight="mvaWeight", bins=bins, error=True, linestyle='dotted' )
    ax1.set_xlabel(classes[i] + " classifier", size=14, horizontalalignment='right', x=1.0)
    ax1.set_ylabel("Events normalized", size=14, horizontalalignment='right', y=1.0)

    ax1.tick_params(which='major', length=8)
    ax1.tick_params(which='minor', length=4)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.margins(x=0)
    ax1.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False, loc='upper center')
    
    plt.savefig(os.path.join(model_outpath, var + ".png"))
    
   
#=====================================================================================================================   
end = time.time()
hours = int((end - start)/3600)
minutes = int(((end - start)%3600)/60)
seconds = int(((end - start)%3600)%60)

print("")
print("-----------------------------------------------------------------------------------")
print("Total process duration: " + str(hours) + " hours " + str(minutes) + " minutes " + str(seconds) + " seconds")  
print("-----------------------------------------------------------------------------------")
print("")







