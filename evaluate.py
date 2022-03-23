import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--period", default="17")
parser.add_argument("-s", "--selection", default="ML")
parser.add_argument("--check", dest='check_flag', action='store_true')
parser.set_defaults(check_flag=False)
args = parser.parse_args()


selection = args.selection
period = args.period

outpath = os.environ.get("HEP_OUTPATH")
ml_outpath = os.path.join(outpath, "HHDM", selection, "datasets", period, "ML")

list_signals = os.listdir(ml_outpath)

for signal in list_signals:
    list_models = os.listdir(os.path.join(ml_outpath, signal, "models"))
    models_loss = []
    models_accuracy = []
    models_iterations = []
    models_name = []
    for model in list_models:
        training_file = os.path.join(ml_outpath, signal, "models", model, "training.csv")
        if os.path.isfile(training_file):
            df_training = pd.read_csv(training_file)
            if len(df_training) > 0:
                min_loss = np.amin(df_training["test_loss"])
                models_loss.append(min_loss)
                models_accuracy.append(np.array(df_training[df_training["test_loss"] == min_loss]["test_acc"])[-1])
                models_iterations.append(np.array(df_training[df_training["test_loss"] == min_loss]["interation"])[-1])
                models_name.append(model)
    df_result = pd.DataFrame({"Model": models_name, "Loss": models_loss, "Accuracy": models_accuracy, "Iterations": models_iterations})
    df_result = df_result.sort_values("Loss")
    df_result = df_result.reset_index()
    pd.set_option("display.precision", 15)
    print("============================================================================================================")
    print(signal)
    print("============================================================================================================")
    print(df_result)
    print("")
    df_result.to_csv(os.path.join(ml_outpath, signal, "training_result.csv"))














