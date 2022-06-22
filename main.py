import chesspy
import GameGenerator
#from GameGenerator import serialize_example
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import keras
import numpy as np
import time
import os
from google.protobuf import text_format

tnp.experimental_enable_numpy_behavior()

batch_size=4096
model=''
while True:
    if model=='':
        if os.path.exists("TrainModel"):
            if not os.listdir("TrainModel"):
                model=chesspy.NetTower()
            else:
                model=tf.keras.load_model("TrainModel")
        else:
            model=chesspy.NetTower()

    optimizer = keras.optimizers.SGD(learning_rate=0.2,momentum=0.9)

    @tf.function
    def train_step(pi, mask, input,z):
        with tf.GradientTape() as tape:
            policy,value = model(input, training=True)
            rawP=tnp.array([tnp.array(policy)[0,loc[2],loc[1],loc[0]] for loc in mask])
            P=rawP/tnp.sum(rawP)
            hidden_weights=model.weights[0]

            loss_value = tf.pow(tf.math.add(z,tf.math.negative(value)),2)+tnp.dot(pi,tnp.log(P))+ tf.nn.l2_loss(hidden_weights)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss_value



    gamedump,z=GameGenerator.TrainGame()

    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        '''dataset=tf.data.TFRecordDataset(filenames=["gamedata/playdata.tfrecord"])
        raw_example = next(iter(dataset))
        parsed = tf.train.Example.FromString(raw_example.numpy())

        #parsed=text_format.Parse(tf.train.Example())
        parsed=tf.io.parse_example(parsed.SerializeToString(),features={
            'pi':tf.io.FixedLenFeature([], tf.string),
            'mask':tf.io.FixedLenFeature([], tf.string),
            'inputstack': tf.io.FixedLenFeature([], tf.string),
            'z':tf.io.VarLenFeature(tf.int64)})
        parsed["pi"]=tf.io.decode_raw(parsed["pi"], tf.int8)
        parsed["mask"]=tf.io.decode_raw(parsed["mask"], tf.int8)
        parsed["inputstack"]=tf.io.decode_raw(parsed["inputstack"], tf.int8)
        #parsed["mask"]=tf.reshape(tf.io.decode_raw(parsed["mask"], tf.uint8),[-1, 3])
        #parsed["inputstack"]=tf.reshape(tf.io.decode_raw(parsed["inputstack"], tf.uint8),[119, 8, 8])'''



        # Iterate over the batches of the dataset.
        for step, item in enumerate(gamedump):
            loss_value = train_step(item[0],item[1],item[2],z)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))



        
        print("Time taken: %.2fs" % (time.time() - start_time))