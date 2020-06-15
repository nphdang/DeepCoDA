import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# set big font
import seaborn as sns
sns.set_context("notebook", font_scale=1.8)
plt.style.use('fivethirtyeight')
import timeit
import datetime
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Concatenate, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from keras import regularizers
from keras import optimizers

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="5a", type=str, nargs='?', help='dataset')
parser.add_argument('--level', default="5", type=str, nargs='?', help='level')
parser.add_argument('--sumzero', default="1e0", type=str, nargs='?', help='sumzero')
parser.add_argument('--l1', default="1e-2", type=str, nargs='?', help='l1')
args = parser.parse_args()
print("data_id={}, cascade_lvl={}, sumzero_lambda={}, l1_lambda={}".format(args.dataset, args.level, args.sumzero, args.l1))

np.random.seed(123)

# discovery set
"""
datasets = {"1a": "1a-selbal-crohn", "1b": "1b-selbal-msm-hivonly",
            "2a": "2a-franzosa-mic-ibd-v-hc", "2b": "2b-franzosa-mic-cd-v-uc", "2c": "2c-franzosa-met10-ibd-v-hc", "2d": "2d-franzosa-met10-cd-v-uc",
            "3a": "3a-duvallet-schubert-case-v-diarrhea", "3b": "3b-duvallet-schubert-case-v-nondc", "3c": "3c-duvallet-baxter-crc-v-h", "3d": "3d-duvallet-baxter-crc-v-noncrc",
            "4a": "4a-brca-mirna10-ca-v-hc", "4b": "4b-brca-mirna10-her2-v-ca", "4c": "4c-brca-mirna10-lumA-v-lumB"}
"""

# verification set
datasets = {"5a": "5a-gevers-task-ileum", "5b": "5b-gevers-task-rectum",
            "6a": "6a-hmp-task-gastro-oral", "6b": "6b-hmp-task-sex", "6c": "6c-hmp-task-stool-tongue-paired", "6d": "6d-hmp-task-sub-supragingivalplaque-paired",
            "7a": "7a-kostic-task",
            "8a": "8a-qin2012-task-healthy-diabetes", "8b": "8b-qin2014-task-healthy-cirrhosis",
            "9a": "9a-ravel-task-black-hispanic", "9b": "9b-ravel-task-nugent-category", "9c": "9c-ravel-task-white-black"}

dataset = args.dataset
if dataset != "all":
    datasets = {dataset: datasets[dataset]}
test_size = 0.1
sumzero_lambda = float(args.sumzero)
l1_lambda = float(args.l1)
use_weight_constraint = True
n_run = 20

# hyper-parameters for network
cascade_level = int(args.level)
bottle_dim = 1
latent_dim = 1
output_dim = 1
batch_size = 32
epochs = 200

# regularize sum of weights at each cascade to be 0 and regularize weights to be sparse
class SumZeroL1Reg(regularizers.Regularizer):
    def __init__(self, sumzero_lambda=1e0, l1_lambda=1e-2):
        self.sumzero_lambda = K.cast_to_floatx(sumzero_lambda)
        self.l1_lambda = K.cast_to_floatx(l1_lambda)

    def __call__(self, w):
        sumzero_reg = 0
        sumzero_reg += self.sumzero_lambda * K.square(K.sum(w))

        l1_reg = 0
        l1_reg += self.l1_lambda * K.sum(K.abs(w))

        return sumzero_reg + l1_reg

    def get_config(self):
        return {'sumzero_lambda': float(self.sumzero_lambda),
                'l1_lambda': float(self.l1_lambda)}

auc_dataset_run = []
for dataset in list(datasets.values()):
    start_date_time = datetime.datetime.now()
    start_time = timeit.default_timer()

    # read data from file
    df_x = pd.read_csv("./data/{}-x.csv".format(dataset), header=0, sep=",")
    df_y = pd.read_csv("./data/{}-y.csv".format(dataset), header=0, sep=",")
    X = df_x.iloc[:, 1:]
    X_rowname = df_x.iloc[:, 0]
    y = df_y.iloc[:, 1]
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(-1, 1)
    n_sample = X.shape[0]
    n_feature = X.shape[1]
    print("dataset={}, n_samples={}, n_features={}".format(dataset, n_sample, n_feature))
    print("cascade_lvl={}, sumzero_lambda={}, l1_lambda={}".format(cascade_level, sumzero_lambda, l1_lambda))

    auc_run = np.zeros(n_run)
    for run in range(n_run):
        print("run={}".format(run))
        # split data to train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=run, stratify=y)
        # construct network
        time1 = timeit.default_timer()
        input_dim = n_feature
        # Encoder
        x = Input(shape=(input_dim,))
        # concat layer for all z
        concat_z = []
        for level_id in range(cascade_level):
            x_log = Lambda(lambda t: K.log(t))(x)
            if use_weight_constraint == True:
                b = Dense(bottle_dim, activation='linear',
                          kernel_regularizer=SumZeroL1Reg(sumzero_lambda=sumzero_lambda, l1_lambda=l1_lambda))(x_log)
            else:
                b = Dense(bottle_dim, activation='linear')(x_log)
            z = b
            concat_z.append(z)
        if cascade_level == 1:
            all_z = z
        else:
            all_z = Concatenate()(concat_z)
        # Decoder
        decoder = Sequential([Dense(output_dim, input_dim=cascade_level*latent_dim, activation='sigmoid')])
        y_pred = decoder(all_z)
        # train network
        model = Model(inputs=x, outputs=y_pred, name='bottleneck_model')
        opt = optimizers.Adam()
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        hist = model.fit(X_train, y_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
        time2 = timeit.default_timer()
        print("runtime of training network: {}(s)".format(round(time2 - time1, 2)))

        # compute auc on test set
        y_test_pred = model.predict(X_test)
        y_test_pred_round = np.around(y_test_pred)
        auc = roc_auc_score(y_test, y_test_pred)
        print("auc={}".format(round(auc, 4)))
        auc_run[run] = auc

        # save weight matrices when only train network on full dataset once
        if n_run == 1:
            # obtain weights of the log-contrast layer for each cascade level
            # (i.e. layers at position 0, 2, ...)
            with open('wo_attn_wei_logs_{}_lvl_{}_s0_{}_l1_{}_run{}.csv'.format(dataset, cascade_level, sumzero_lambda,
                                                                                l1_lambda, run), 'w') as f:
                f.write("cascade_lvl_id, weight_id, weight_value\n")
                W_z = []
                for level_id in range(cascade_level):
                    weights_first_layer = model.get_weights()[2*level_id]
                    w_zi = [weight[0] for weight in weights_first_layer]
                    w_zi = np.array(w_zi)
                    w_zi = np.around(w_zi, 2) + 0.0
                    w_zi_sum = round(w_zi.sum(), 2)
                    print("level_id={}, w_zi={}, w_zi_sum={}".format(level_id, w_zi, str(w_zi_sum)))
                    # save result to file
                    for weight_id in range(len(w_zi)):
                        line = str(level_id+1) + "," + str(weight_id+1) + "," + str(w_zi[weight_id]) + "\n"
                        f.write(line)
                    W_z.append(w_zi)
                W_z = np.array(W_z)

        # delete trained network to free memory
        del model, decoder
        K.clear_session()

    end_date_time = datetime.datetime.now()
    end_time = timeit.default_timer()
    runtime = round(end_time-start_time, 2)

    auc_run = np.around(auc_run, 4)
    avg_auc = round(np.mean(auc_run), 4)
    std_err = round(np.std(auc_run) / np.sqrt(n_run), 2)
    print("dataset={}, cascade_lvl={}, sumzero_lambda={}, l1_lambda={}".format(dataset, cascade_level, sumzero_lambda, l1_lambda))
    print("auc_run=[{}]".format(",".join(map(str, auc_run))))
    print("avg_auc={}, std_err={}".format(avg_auc, std_err))
    print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
    print("runtime: {}(s)".format(runtime))
    # save result to file
    with open('wo_attn_auc_{}_lvl_{}_s0_{}_l1_{}.txt'.format(dataset, cascade_level, sumzero_lambda, l1_lambda), 'w') as f:
        f.write("dataset={}, cascade_lvl={}, sumzero_lambda={}, l1_lambda={}\n".format(dataset, cascade_level, sumzero_lambda, l1_lambda))
        f.write("auc_run=[{}]\n".format(",".join(map(str, auc_run))))
        f.write("avg_auc={}, std_err={}\n".format(avg_auc, std_err))
        f.write("runtime: {}(s)".format(runtime))
    # store auc of n_run of each dataset
    auc_dataset_run.append(auc_run)


# plot auc for each dataset
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x=list(datasets.values()), y=auc_dataset_run, notch=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylabel("AUC")
plt.savefig("wo_attn_auc_{}_lvl_{}_s0_{}_l1_{}.pdf".format(args.dataset, cascade_level, sumzero_lambda, l1_lambda), bbox_inches="tight")
plt.close()

# save result to csv file
n_dataset = len(datasets)
with open('wo_attn_auc_{}_lvl_{}_s0_{}_l1_{}.csv'.format(args.dataset, cascade_level, sumzero_lambda, l1_lambda), 'w') as f:
    f.write("dataset, cascade_lvl, sumzero_lambda, l1_lambda, bootstrapp, auc\n")
    for data_id in range(n_dataset):
        for auc_id in range(n_run):
            data_name = list(datasets.values())[data_id]
            line = data_name + "," + str(cascade_level) + "," + str(sumzero_lambda) + "," + str(l1_lambda) + "," + \
                   str(auc_id) + "," + str(auc_dataset_run[data_id][auc_id]) + "\n"
            f.write(line)


