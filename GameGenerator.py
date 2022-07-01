import MCTSchess
import FEN
import chesspy
import chess
import numpy as np
import time
import tensorflow as tf
import os

model =chesspy.NetTower()
checkpoint=tf.train.Checkpoint(model)
from multiprocessing import util
checkpoint_dir = os.path.join(util.get_temp_dir(), 'ckpt')
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
  checkpoint.restore(latest_checkpoint)

class Trainer():#MCTSchess.tree):
    @staticmethod
    def maxdepth():
        return 100
        
    def __init__(self,move='',totalval=0,visit=0,leafnodes=[],depth=0,P=0):
        self.totalval=totalval
        self.visit=visit
        self.P=P
        if type(move)==list:
            self.move=move[0]
            if len(move)>1:
                self.leafnodes=[Trainer(move[1:])]
                self.depth=-1
            else:
                self.leafnodes=leafnodes
                self.depth=0
        else:
            self.move=move
            self.leafnodes=leafnodes
            self.depth=depth

    def PUCT(self,tree,parentvisit):
            if tree.visit==0:
                return 0.25*self.P*np.sqrt(tree.visit)/(1+tree.visit)
            return tree.totalval/tree.visit+0.25*self.P*np.sqrt(tree.visit)/(1+tree.visit)
    
    def SelfChooseMove(self,offset,history=[]):
        if offset!=self.depth:
            ret=self.leafnodes[0].SelfChooseMove(offset,history=history+[self.move])
            return ret
        else:
            pi=[]
            mask=[]
            for i in self.leafnodes:
                pi.append(i.visit)
                mask.append(FEN.ReturnArrLoc(i.move))

            inputstack,gamestate,state=FEN.InputFeature(history+[self.move])
            output=[pi,mask,inputstack]
            


            #file


            k=[self.leafnodes[pi.index(max(pi))]]
            del self.leafnodes
            self.leafnodes=k
            print(k[0].move)
            return output,gamestate,state

    def expand(self,history):


            #model=chesspy.NetTower()
            stack,fen,state=FEN.InputFeature(history)#,fen[1])
            

            tempboard=chess.Board(fen)
            self.leafnodes=list(tempboard.legal_moves)
            if self.leafnodes!=[]:

                policy,value=model.predict(stack)
                policy=np.transpose(policy[0],(2,1,0))
                PolicyVal=[]
                fen=fen.split()
                for i in self.leafnodes:
                    PolicyVal.append(policy[FEN.ReturnArrLoc(str(i))])


                PolicyVal=PolicyVal/np.sum(PolicyVal)

            
            

                #print(list(tempboard.legal_moves))
                for i in range(len(self.leafnodes)):
                    self.leafnodes[i]=Trainer(str(self.leafnodes[i]),depth=self.depth+1,P=PolicyVal[i])
                return self.leafnodes
                #history=self.history+[str(self.leafnodes[i])],
            else: return []

    def roll(self,lastmoves):
        #model=chesspy.NetTower()
        policy,value=model.predict(FEN.InputFeature(lastmoves)[0])

        return value

    def traverse(self,lastmoves=[],offset=0):
        if self.leafnodes==[] and self.depth<self.maxdepth()+offset:
            if not self.visit:
                self.expand(lastmoves+[self.move])
                return 1e-20
          


        else:
            if not self.visit:
                m=self.roll(lastmoves+[self.move])
                self.visit+=1
                self.totalval+=m
                return m
            else:
                
                
                if self.depth<self.maxdepth()+offset:
                    temp=[]
                    for i in self.leafnodes:
                        temp.append(self.PUCT(tree=i,parentvisit=self.visit))
                    if all(temp):
                        m=self.leafnodes[temp.index(max(temp))].traverse(lastmoves+[self.move],offset=offset)
                    else:
                        m=self.leafnodes[np.random.randint(len(temp))].traverse(lastmoves+[self.move],offset=offset)
                    if not m==0:
                        self.totalval=self.totalval+m
                        self.visit+=1
                    return m
                else:
                    
                    m=self.roll(lastmoves)
                    #print(self.visit,"here")
                    self.visit+=1
                    self.totalval+=m
                    return m

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _serialize_array(value):
    return tf.io.serialize_tensor(value)


def serialize_example(pi,mask,inputstack,z):
    feature={
        'pi':_bytes_feature(_serialize_array(pi)),
        'mask':_bytes_feature(_serialize_array(mask)),
        'inputstack':_bytes_feature(_serialize_array(inputstack)),
        'z':_int64_feature(z)
    }

    proto=tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()




def TrainGame(model):
    game=True
    curr=Trainer()
    offset=0
    i=1
    gamedump=[]
    #pi,mask,input=[],[],[]
    while game:
        while curr.visit<10*i:
            mm=time.time()
            curr.traverse(offset=offset)
            print(time.time()-mm,offset)

        k=curr.SelfChooseMove(offset)
        #print(k)
        offset+=1
        i+=1
        z=None
        if  k[-1]!=None: 
            res=k[-1].result()
            if res=="1/2-1/2":
                z=0.0
            elif res=="1-0":
                z=1.0
            elif res=="0-1":
                z=-1.0
            game=False
            pass
        #pi.append(k[0][0])
        #mask.append(k[0][1])
        #input.append(k[0][2])
        gamedump.append([k[0][0],k[0][1],k[0][2]])
        if offset>512 or k[0][2][0,7,7,118]>100 or (k[0][2][0,7,7,110] and k[0][2][0,7,7,111]):
            z=0.0
            game=False
        if z!=None:
            for i in gamedump:
                i.append(z)
            return gamedump
            #zlen=len(pi)
            #return (tf.ragged.constant(pi),tf.ragged.constant(mask),input,z*np.ones([zlen]))
            '''with tf.io.TFRecordWriter("gamedata/playdata.tfrecord") as writer:
                for i in gamedump:
                    #print(tf.cast(i[1],tf.int8),tf.cast(i[1],tf.int8).numpy())
                    writer.write(serialize_example(tf.cast(i[0],tf.int8).numpy(),tf.cast(i[1],tf.int8).numpy(),tf.cast(i[2],tf.int8).numpy(),z))'''

            print(z)
        


#TrainGame()


