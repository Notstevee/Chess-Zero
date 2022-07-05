import json
import os
import sys
import tensorflow as tf

strategy = tf.distribute.MultiWorkerMirroredStrategy()

'''os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
  sys.path.insert(0, '.')
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"]
    },
   "task": {"type": "worker", "index": 1}
})'''

import chesspy
import GameGenerator
#from GameGenerator import serialize_example

import tensorflow.experimental.numpy as tnp
import keras
import numpy as np
import time
import os
from multiprocessing import util,Pool,cpu_count,freeze_support,Lock,Process,Queue

tnp.experimental_enable_numpy_behavior()


# Checkpoint saving and restoring
def _is_chief(task_type, task_id, cluster_spec):
  return (task_type is None
          or task_type == 'chief'
          or (task_type == 'worker'
              and task_id == 0
              and 'chief' not in cluster_spec.as_dict()))

def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir

def write_filepath(filepath, task_type, task_id, cluster_spec):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id, cluster_spec):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)

checkpoint_dir = ".\models"#os.path.join(util.get_temp_dir(), 'ckpt')

num_epochs = 500000
batch_size=4096
min_factor=2
max_factor=16
cpu=3
#cpu=int(cpu_count()*0.2//1)

learn_schedule=tf.keras.optimizers.schedules.PiecewiseConstantDecay([100000,300000,500000],[0.2,0.02,0.002,0.0002])


# Define Strategy
#strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    # Model building/compiling need to be within `tf.distribute.Strategy.scope`.
    model =chesspy.NetTower()

    #multi_worker_dataset = strategy.experimental_distribute_datasets_from_function(data)
    optimizer = keras.optimizers.SGD(learning_rate=learn_schedule,momentum=0.9)



@tf.function
def train_step(iterator):
  """Training step function."""

  def step_fn(inputs):
    """Per-Replica step function."""
    pi, mask, input,z = inputs

    with tf.GradientTape() as tape:
            policy,value = model(input, training=True)
            rawP=[tnp.array(policy)[0,i[2],i[1],i[0]] for i in mask]
            P=rawP/tnp.sum(rawP)
            hidden_weights=model.weights[0]

            loss_value = tf.pow(tf.math.add(z,tf.math.negative(value)),2)+tnp.dot(pi,tnp.log(P))+ tf.nn.l2_loss(hidden_weights)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value

  per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
  return strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

epoch = tf.Variable(
    initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
step_in_epoch = tf.Variable(
    initial_value=tf.constant(0, dtype=tf.dtypes.int64),
    name='step_in_epoch')

task_type, task_id, cluster_spec = (strategy.cluster_resolver.task_type,
                                    strategy.cluster_resolver.task_id,
                                    strategy.cluster_resolver.cluster_spec())

checkpoint = tf.train.Checkpoint(
    model=model, epoch=epoch, step_in_epoch=step_in_epoch)

write_checkpoint_dir = '.\models'#write_filepath(checkpoint_dir, task_type, task_id,
                                      #cluster_spec)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory=write_checkpoint_dir, max_to_keep=None)

# Restoring the checkpoint
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
  checkpoint.restore(latest_checkpoint)


multi_worker_dataset=[]


def log(result):
    for i in result:
        multi_worker_dataset.append(i)



if __name__ == '__main__':

    ite=1

    with Pool(processes=cpu) as pool:

        
        

        #pool=Pool(processes=cpu)
        lst=[pool.apply_async(GameGenerator.TrainGame,callback=log) for i in range(100000)]

    # Resume our CTL training
        while epoch.numpy() < num_epochs:
            if len(multi_worker_dataset)<ite*batch_size*min_factor:
                timee=time.time()
                while len(multi_worker_dataset)<ite*batch_size*min_factor:
                    print(f"\n\n\n\n\nGame steps simulated ({len(multi_worker_dataset)}) less than required minimum ({ite*batch_size*min_factor}). simulation running for {time.time()-timee} seconds.\n\n\n\n\n")
                    time.sleep(10)

                print("\n\n\n\n\nStarting epoch.\n\n\n\n\n")
            
            iterator = iter(multi_worker_dataset[i] for i in np.random.choice(len(multi_worker_dataset),batch_size,replace=False))
            total_loss = 0.0
            num_batches = 0
            step_in_epoch.assign(0)

            while step_in_epoch.numpy() < batch_size:
                total_loss += train_step(iterator)
                num_batches += 1
                step_in_epoch.assign_add(1)

            train_loss = total_loss / num_batches
            print('Epoch: %d, train_loss: %f.'
                            %(epoch.numpy(),  train_loss))



            checkpoint_manager.save()
            if not _is_chief(task_type, task_id, cluster_spec):
                tf.io.gfile.rmtree(write_checkpoint_dir)

            epoch.assign_add(1)
            step_in_epoch.assign(0)
            ite+=1
            if len(multi_worker_dataset)>batch_size*max_factor:
                multi_worker_dataset=multi_worker_dataset[batch_size*max_factor:]

        for i in lst:
            i.get()
        pool.close()
        pool.join()


'''batch_size=4096
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

    strategy = tf.distribute.MirroredStrategy()

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

        dataset=tf.data.TFRecordDataset(filenames=["gamedata/playdata.tfrecord"])
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
        #parsed["inputstack"]=tf.reshape(tf.io.decode_raw(parsed["inputstack"], tf.uint8),[119, 8, 8])



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



        
        print("Time taken: %.2fs" % (time.time() - start_time))'''