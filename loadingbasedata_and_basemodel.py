

"""
Created on Fri Jan  9 13:38:33 2026

@author: 30699
"""
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import moduleUtils as U
from tensorflow.keras.layers import SpatialDropout1D

def accdipiction (history):
    ########################################
    #graph
    val_acc = history.history['val_accuracy']
    acc = history.history['accuracy']
    
    epochs = range(1, len(acc) + 1)
    
    # 2. Plot Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')     # Blue line
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy') # Red line
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()    
    plt.grid(True)
    print('\n \n max val accouracy',max(val_acc),'in epoch',val_acc.index(max(val_acc)))



U.cls()
seed = 7
np.random.seed(seed)

prc=1
if prc==1:
    path = "./data/"
    
    def load_and_stack(file_pattern):
        files = sorted(glob.glob(file_pattern))
        if not files:
            print(f"Warning: No files found for {file_pattern}")
            return None
        
        data_3d = []
        for f in files:
            df = pd.read_csv(f)
            data_3d.append(df.values)
        
        # Returns shape: (Samples, Time_Rows, Channel_Cols)
        return np.stack(data_3d, axis=0)
    
    # 1. Load Data
    # ------------
    # Baseline (Class 0)
    xbl_raw = load_and_stack(os.path.join(path, "*_trial_ima_bl*.csv"))
    ybl = np.zeros(len(xbl_raw), int)
    
    # Right (Class 1)
    xrl_raw = load_and_stack(os.path.join(path, "*_trial_ima_rl*.csv"))
    yrl = np.ones(len(xrl_raw), int)
    
    # Left (Class 2)
    xll_raw = load_and_stack(os.path.join(path, "*_trial_ima_ll*.csv"))
    yll = np.ones(len(xll_raw), int) * 2
    
    # NN (Class 3) - Fixed the path logic here
    xnn_raw = load_and_stack(os.path.join(path, "*_trial_ima_nn*.csv")) 
    ynn = np.ones(len(xnn_raw), int) * 3
    
    # 2. Concatenate
    # --------------
    X_raw = np.concatenate((xbl_raw, xrl_raw, xll_raw, xnn_raw), axis=0)
    del xbl_raw, xrl_raw, xll_raw, xnn_raw#gia xwro ram
    
    
    Y_labels = np.concatenate((ybl, yrl, yll, ynn), axis=0)
    
    
    '''
    ####################-----------------
 
    '''
   
    
    
    
    # 3.  Normalization
    # -----------------------
    
    
 
    print(f"Old Max: {np.max(X_raw)}, Old Min: {np.min(X_raw)}")
        # Do this BEFORE (X - mean) / std
    lower = np.percentile(X_raw, 1)
    upper = np.percentile(X_raw, 99)
    X_clipped = np.clip(X_raw, lower, upper)
    print(f"before x-max/std Max: {np.max(X_clipped)}, x-max/std  Min: {np.min(X_clipped)}")
    # Now calculate mean/std from the cleaned data
    mean = X_clipped.mean(axis=(0, 1), keepdims=True)
    std = X_clipped.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X_clipped - mean) / std
        
        
    print(f"New Max: {np.max(X)}, New Min: {np.min(X)}")
       
    #(N_Samples, Time_Steps, Features)
  
    # 4. One-Hot Encoding Targets
    # ------------------------------
    #  categorical crossentropy
    Y = to_categorical(Y_labels, num_classes=4)
 #############################################################################################
 
 ###############################################################################################
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import numpy as np

# 1. Check and Fix Shape
# ----------------------
print(f"Original Shape: {X.shape}") # Likely (2240, 62, 819)



if X.shape[2] > X.shape[1]:
    print("Transposing data to put Time in the correct dimension...")
    X = np.transpose(X, (0, 2, 1))

    print(f"Corrected Shape for CNN: {X.shape}") 
# Ensure this prints something like (2240, 819, 62) before proceeding


# trimmining time 
# and fs = 256
fs = 256 
start_sample = int(0.3 * fs) # ~77
end_sample = int(1.3 * fs)   # ~768

Xcl = X[:, start_sample:end_sample, :]
print(f"old windowed shape: {X.shape}")

print(f"New windowed shape: {Xcl.shape}")



# 2. Split Data Again (with new shape)
X_train, X_test, y_train, y_test = train_test_split(Xcl, Y, test_size=0.2, random_state=seed)
##############################




# --- HYPERPARAMETERS ---
n_timesteps = X_train.shape[1]
n_features  = X_train.shape[2] 




from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, AveragePooling1D, SpatialDropout1D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# --- COMPACT MODEL (Designed for 62 Channels) ---
input_layer = Input(shape=(n_timesteps, n_features))

# 1. SPATIAL FILTERING (The "Bottleneck")
# We use kernel_size=1 to look at all 62 channels at a SINGLE time point.
# This learns to combine electrodes (like C3-C4) without worrying about time yet.
# We shrink 62 channels -> 8 'Spatial Patterns'.
x = Conv1D(filters=8, kernel_size=1, padding='same', use_bias=False, 
           kernel_constraint=max_norm(1.0))(input_layer)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = SpatialDropout1D(0.2)(x) # Light dropout

# 2. TEMPORAL FILTERING
# Now we look at time, but only on our 8 calculated spatial patterns.
# kernel_size=32 captures ~125ms of data (at 256Hz)
x = Conv1D(filters=16, kernel_size=32, padding='same', use_bias=False, 
           kernel_regularizer=l2(0.01))(x) # L2 regularization fights overfitting
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = AveragePooling1D(pool_size=4)(x) # Average Pooling is smoother than Max for EEG
x = SpatialDropout1D(0.5)(x) # Stronger dropout here

# 3. SECOND TEMPORAL BLOCK (Refinement)
x = Conv1D(filters=32, kernel_size=16, padding='same', use_bias=False,
           kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = AveragePooling1D(pool_size=8)(x)
x = SpatialDropout1D(0.5)(x)

# 4. CLASSIFICATION
x = Flatten()(x)
output_layer = Dense(4, activation='softmax', kernel_constraint=max_norm(0.5))(x)

model = Model(inputs=input_layer, outputs=output_layer)

# --- COMPILE ---
# A slightly lower learning rate helps when using L2 regularization
opt = Adam(learning_rate=0.0005)

model.compile(
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    optimizer=opt,
    metrics=['accuracy']
)

model.summary()
# --- CALLBACKS ---
early_stopping = EarlyStopping(monitor='val_accuracy', patience=55, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=100, min_lr=0.00001, verbose=1)

print(f"Training Step: Switching to 1D Convolution (Corrected Syntax)...")
history = model.fit(
    X_train, y_train, 
    epochs=250, 
    batch_size=64, 
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
    
)




accdipiction(history)
