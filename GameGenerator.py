import MCTSchess
import FEN
import chesspy
import chess
import numpy as np
import time
import tensorflow as tf


model=chesspy.NetTower()
class Trainer(MCTSchess.tree):
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
        'z':_float_feature(z)
    }

    proto=tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()




def TrainGame():
    game=True
    curr=Trainer()
    offset=0
    i=1
    gamedump=[]
    while game:
        while curr.visit<5*i:
            mm=time.time()
            curr.traverse(offset=offset)
            print(time.time()-mm,offset)

        k=curr.SelfChooseMove(offset)
        print(k)
        offset+=1
        i+=1
        z=None
        if  k[-1]!=None: 
            res=k[-1].result()
            if res=="1/2-1/2":
                z=0
            elif res=="1-0":
                z=1
            elif res=="0-1":
                z=-1
            game=False
            pass
        gamedump.append(k[0])
        if offset>512 or k[0][2][0,7,7,118]>100 or (k[0][2][0,7,7,110] and k[0][2][0,7,7,111]):
            z=0
            game=False
        if z!=None:
            
            with tf.io.TFRecordWriter("playdata.tfrecord") as writer:
                for i in gamedump:
                    i[1]
                    writer.write(serialize_example(i[0],i[1],i[2],z))

            print(z)
        


TrainGame()


