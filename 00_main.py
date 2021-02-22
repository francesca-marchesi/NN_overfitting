import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense
import net
import graphics as gr
import argparse
import manipulation as man
import matplotlib.pyplot as plt
import numpy as np


def main():
    # df = man.import_data()
    df = pd.read_excel('/Users/Francesca/Desktop/tesi2021/Codice/NN_overfitting/Data/originale.xlsx')
    variable_name = ["nome", "dim", "HU_bas", "HU_art", "HU_ven", "HU_tard", "HU_rem", "HU_art/HU_bas", "HU_ven/HU_bas", \
                     "HU_tard/HU_bas", "necrosi", "ENH_perif", "ipodenso_pan", "ipodenso_ven", "linfadenopatie", \
                     "fat_stranding", "est_duo", "est_bil", "est_lam", "est_ven", "est_art", "inf_organi", "est_ext", \
                     "plexus1", "plexus2", "ante_path", "root", "n_plessi", "Age", "Sex", "Ca", "CaStrat", "size_tum", \
                     "inv_peri", "inv_vas", "grading", "G1G2", "AJCC8", "LNP", "LNP_LNM", "LNR", "Followup",
                     "Lenght_Followup","LOCAL_REL", "Time_Local", "DIST_REL", "Time_Dist", "LOCAL_DIST", "Time_LD"]

    df_medico = df.loc[:, variable_name]
    list = ['DIST_REL', 'Time_Dist']
    my_df = df_medico[df_medico.columns.difference(['LOCAL_REL', 'Time_Local', 'nome'])] #questo è buona
    my_df = man.del_missing(my_df, col_name=list)
    my_df = man.del_missing(my_df, col_name=None)
    df_target = pd.DataFrame(columns=['target'])
    df_target.target = (my_df.Time_Dist > 9)
    df_dummies = pd.get_dummies(df_target['target'])
    my_df['target'] = df_dummies.iloc[:, 0]
    my_df = my_df.drop(['DIST_REL', 'Time_Dist'], axis=1)

    df_dim = len(my_df.columns) - 1
    X_train, X_val, Y_train, Y_val = train_test_split(my_df.iloc[:, 0:df_dim],
                                                      my_df.iloc[:, df_dim],
                                                      test_size=0.33,
                                                      random_state=42)
    model = net.simple_net(X_train)
    train_dim = len(X_train.columns)
    opt = tf.keras.optimizers.Adam(learning_rate=0.06)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=opt, loss=loss, metrics='binary_accuracy')
    model.summary()
    history = model.fit(X_train,
                        Y_train,
                        batch_size=train_dim,
                        epochs=100,
                        shuffle=True,
                        validation_data=(X_val, Y_val)
                        )
    gr.plot_history(history)

if __name__ == "__main__":
    main()
