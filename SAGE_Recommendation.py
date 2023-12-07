import pandas as pd
import numpy as np
import zipfile
import os
import math

import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from collections import Counter

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import warnings
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive')

local_zip = 'drive/MyDrive/Dataset/Dataset_Fix/job-recommendation.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()

n_user = 10000

users = pd.read_csv ("users.tsv", sep = '\t')
users = users.loc[users['UserID'] < n_user]
users = users.loc[users['Country'] == 'US']
users = users.reset_index()
users = users.drop(columns=['index','WindowID', 'Split','ZipCode','Country'])
users

users.isnull().sum()

users.replace('', np.nan, inplace=True)
users.dropna(inplace=True)
users.isnull().sum()

users['GraduationDate'] = pd.to_datetime(users['GraduationDate'])
users['GraduationYear'] = pd.DatetimeIndex(users['GraduationDate']).year
users = users.drop(columns=['GraduationDate'])

users['CurrentlyEmployed'] = users['CurrentlyEmployed'].replace('Yes', 1)
users['CurrentlyEmployed'] = users['CurrentlyEmployed'].replace('No', 0)

users['ManagedOthers'] = users['ManagedOthers'].replace('Yes', 1)
users['ManagedOthers'] = users['ManagedOthers'].replace('No', 0)

enc_city = LabelEncoder()
enc_city.fit(users['City'])
users['City'] = enc_city.transform(users['City'])

print(list(enc_city.classes_))
print(list(enc_city.transform(enc_city.classes_)))

enc_state = LabelEncoder()
enc_state.fit(users['State'])
users['State'] = enc_state.transform(users['State'])

print(list(enc_state.classes_))
print(list(enc_state.transform(enc_state.classes_)))

enc_degree = LabelEncoder()
enc_degree.fit(users['DegreeType'])
users['DegreeType'] = enc_degree.transform(users['DegreeType'])

print(list(enc_degree.classes_))
print(list(enc_degree.transform(enc_degree.classes_)))

enc_major = LabelEncoder()
enc_major.fit(users['Major'])
users['Major'] = enc_major.transform(users['Major'])

print(list(enc_major.classes_))
print(list(enc_major.transform(enc_major.classes_)))

list_user = users['UserID'].to_list()
print(list_user)

users = users.set_index('UserID')
users

pop = pd.read_csv ("popular_jobs.csv")
pop = pop.loc[pop['UserId'] < n_user]
pop

arr_job = pop.to_numpy()
pop_job = []
for i in tqdm(range(len(arr_job))):
    if arr_job[i][0] in list_user:
        data_list = str(arr_job[i][1]).split(' ')
        for item in data_list:
            pop_job.append([arr_job[i][0],int(item)])

df_pop_job = pd.DataFrame(pop_job,columns=['id_user','id_job'])
df_pop_job

apps = pd.read_csv ("apps.tsv", sep = '\t')
apps = apps.loc[apps['UserID'] < n_user]
apps = apps[['UserID','JobID']]
apps

arr_apps = apps.to_numpy()

rec_apps = []
as_user = []
for i in tqdm(range(len(arr_apps))):
    if arr_apps[i][0] in list_user:
        raw = users.loc[arr_apps[i][0]].to_list()
        if list(arr_apps[i]) in pop_job:res = 1
        else:res = 0

        raw.append(arr_apps[i][1])
        raw.append(arr_apps[i][0])
        raw.append(res)
        as_user.append(raw)
        rec_apps.append([arr_apps[i][0],arr_apps[i][1],res])

df = pd.DataFrame(as_user)
df

cite = df[[11,10]]
cite.columns = ['source','target']
cite

plt.figure(figsize=(7, 7))
cora_graph = nx.from_pandas_edgelist(cite.sample(n=1000))
subjects = list(df[df[11].isin(list(cora_graph.nodes))][12])

colors = df[12].tolist()
n_nodes = len(cora_graph.nodes)

nx.draw_spring(cora_graph, node_size=15, node_color=colors[:n_nodes])

list_job = df[10].to_list()
print(list_job)

job_con = []
for i in tqdm(range(len(list_job))):
    for j in range(len(arr_job)):
        data_list = str(arr_job[j][1]).split(' ')
        if str(list_job[i]) in data_list:
            for k in range(len(data_list)):
                if int(data_list[k]) in list_job:
                    job_con.append([int(list_job[i]),int(data_list[k])])

df_con = pd.DataFrame(job_con,columns=['target','source'])
df_con

df = df.drop(columns=[11])
df

job_old = df[10]

df_idx = {name: idx for idx, name in enumerate(sorted(df[10].unique()))}
df[10] = df[10].apply(lambda name: df_idx[name])
df

job_new = df[10]

df_con["source"] = df_con["source"].apply(lambda name: df_idx[name])
df_con["target"] = df_con["target"].apply(lambda name: df_idx[name])
df_con

30//3

random_indices = np.random.permutation(range(df.shape[0]))

train_data = df.iloc[random_indices[: len(random_indices) // 3]]
test_data = df.iloc[random_indices[len(random_indices) // 3 : 2 * (len(random_indices) // 3)]]
val_data = df.iloc[random_indices[2 * (len(random_indices) // 3) :]]

class_values = sorted(df[12].unique())

train_indices = train_data[10].to_numpy()
test_indices = test_data[10].to_numpy()
val_indices = val_data[10].to_numpy()

# Obtain ground truth labels corresponding to each paper_id
train_labels = train_data[12].to_numpy()
test_labels = test_data[12].to_numpy()
val_labels = val_data[12].to_numpy()

# Define graph, namely an edge tensor and a node feature tensor
edges = tf.convert_to_tensor(df_con[["target","source"]])
node_states = tf.convert_to_tensor(df.sort_values(10).iloc[:, 1:-1])

# Print shapes of the graph
print("Edges shape:\t\t", edges.shape)
print("Node features shape:", node_states.shape)

class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out

class sageGNN(layers.Layer):
    def __init__(self, units, num_task=2, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_task = num_task
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_task)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)

class GraphNNSage(keras.Model):
    def __init__(
        self,
        node_states,
        edges,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            sageGNN(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs

    def train_step(self, data):
        indices, labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self([self.node_states, self.edges])
            # Compute loss
            loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Compute gradients
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients (update weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute probabilities
        return tf.nn.softmax(tf.gather(outputs, indices))

    def test_step(self, data):
        indices, labels = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute loss
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}

HIDDEN_UNITS = 10
NUM_HEADS = 8
NUM_LAYERS = 3
OUTPUT_DIM = len(class_values)

NUM_EPOCHS = 20
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 3e-1
MOMENTUM = 0.9

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM)
accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_acc", min_delta=1e-7, patience=6, restore_best_weights=True
)

# Build model
gat_model = GraphNNSage(
    node_states, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM
)

# Compile model
gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

history = gat_model.fit(
              x=train_indices,
              y=train_labels,
              validation_data =(val_indices, val_labels),
              validation_split=VALIDATION_SPLIT,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              callbacks=[early_stopping],
              verbose=2,
          )

df_res = pd.DataFrame(history.history)
df_res

df_res[['acc','val_acc']].plot(figsize=(6, 3))
plt.grid(True)
plt.gca().set_ylim(.5, 1)
plt.show()

df_res[['loss','val_loss']].iloc[1:,:].plot(figsize=(6, 3))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

gat_model.summary()

apps

df_test_job = pd.DataFrame([job_old,job_new])
df_test_job = df_test_job.T
df_test_job.columns = ['old','new']
df_test_job

df_enc_job = df_test_job.groupby('old').min()
df_enc_job = df_enc_job.reset_index()
df_enc_job

def recommend(id_user):
    df_get_user = apps.loc[apps['UserID'] == id_user]
    arr_get_user = df_get_user['JobID'].to_numpy()
    if(len(arr_get_user) > 0):
        return arr_get_user
    else:
        print('No data get')

rec = recommend(72)
rec

rec_enc = []
for i in range(len(rec)):
    raw_enc = df_enc_job.loc[df_enc_job['old'] == rec[i]]['new'].to_numpy()
    if len(raw_enc) > 0:
        rec_enc.append(raw_enc[0])

rec_enc

pred = gat_model.predict(np.array(rec_enc))

for i in range(len(pred)):
    if pred[i][0] > pred[i][1]:print('Job',rec_enc[i],'direkomendasikan')
    else:print('Job',rec_enc[i],'tidak direkomendasikan')

rec = recommend(47)
rec
rec_enc = []
for i in range(len(rec)):
    raw_enc = df_enc_job.loc[df_enc_job['old'] == rec[i]]['new'].to_numpy()
    if len(raw_enc) > 0:
        rec_enc.append(raw_enc[0])

rec_enc
pred = gat_model.predict(np.array(rec_enc))

for i in range(len(pred)):
    if pred[i][0] > pred[i][1]:print('Job',rec_enc[i],'direkomendasikan')
    else:print('Job',rec_enc[i],'tidak direkomendasikan')

rec = recommend(80)
rec
rec_enc = []
for i in range(len(rec)):
    raw_enc = df_enc_job.loc[df_enc_job['old'] == rec[i]]['new'].to_numpy()
    if len(raw_enc) > 0:
        rec_enc.append(raw_enc[0])

rec_enc
pred = gat_model.predict(np.array(rec_enc))

for i in range(len(pred)):
    if pred[i][0] > pred[i][1]:print('Job',rec_enc[i],'direkomendasikan')
    else:print('Job',rec_enc[i],'tidak direkomendasikan')

rec = recommend(123)
rec
rec_enc = []
for i in range(len(rec)):
    raw_enc = df_enc_job.loc[df_enc_job['old'] == rec[i]]['new'].to_numpy()
    if len(raw_enc) > 0:
        rec_enc.append(raw_enc[0])

rec_enc
pred = gat_model.predict(np.array(rec_enc))

for i in range(len(pred)):
    if pred[i][0] > pred[i][1]:print('Job',rec_enc[i],'direkomendasikan')
    else:print('Job',rec_enc[i],'tidak direkomendasikan')

rec = recommend(767)
rec
rec_enc = []
for i in range(len(rec)):
    raw_enc = df_enc_job.loc[df_enc_job['old'] == rec[i]]['new'].to_numpy()
    if len(raw_enc) > 0:
        rec_enc.append(raw_enc[0])

rec_enc
pred = gat_model.predict(np.array(rec_enc))

for i in range(len(pred)):
    if pred[i][0] > pred[i][1]:print('Job',rec_enc[i],'direkomendasikan')
    else:print('Job',rec_enc[i],'tidak direkomendasikan')
