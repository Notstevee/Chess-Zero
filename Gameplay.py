from queue import Queue
import MCTSchess
import FEN
import chesspy
import chess
import numpy as np
import time
import tensorflow as tf
import os
import threading

model =chesspy.NetTower()
checkpoint = tf.train.Checkpoint(model)
from multiprocessing import util
checkpoint.restore(".\models\ckpt-2.index").expect_partial()

Ponder=False



class tree:
    @staticmethod
    def maxdepth():
        return 100

    def __init__(self,move='',totalval=0,visit=0,leafnodes=[],depth=0,P=0,history=[]):
        self.totalval=totalval
        self.visit=visit
        self.P=P
        self.leafnodes=leafnodes
        if type(move)==list:
            self.move=move[-1]
            self.history=move[:-1]
            self.depth=0
        else:
            self.move=move
            self.history=history
            self.depth=depth


    def PUCT(self,tree,parentvisit):
        if tree.visit==0:
            return 0.25*self.P*np.sqrt(tree.visit)/(1+tree.visit)
        return tree.totalval/tree.visit+0.25*self.P*np.sqrt(tree.visit)/(1+tree.visit)#2*np.sqrt(np.log(parentvisit)/tree.visit)

    def expand(self,history):

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


                PolicyVal=np.multiply(1-0.25,PolicyVal/np.sum(PolicyVal))+np.multiply(0.25,np.random.dirichlet([0.3],len(PolicyVal)))
            
            

                #print(list(tempboard.legal_moves))
                for i in range(len(self.leafnodes)):
                    self.leafnodes[i]=tree(str(self.leafnodes[i]),depth=self.depth+1,P=PolicyVal[i])
                return self.leafnodes
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
                m=self.roll(self.history+lastmoves+[self.move])
                self.visit+=1
                self.totalval+=m
                return m
            else:
                
                
                if self.depth<self.maxdepth()+offset:
                    temp=[]
                    for i in self.leafnodes:
                        temp.append(self.PUCT(tree=i,parentvisit=self.visit))
                    m=self.leafnodes[np.random.randint(len(temp))].traverse(self.history+lastmoves+[self.move],offset=offset)
                    if not m==0:
                        self.totalval=self.totalval+m
                        self.visit+=1
                    return m
                else:
                    
                    m=self.roll(self.history+lastmoves)
                    #print(self.visit,"here")
                    self.visit+=1
                    self.totalval+=m
                    return m

    def SelfChooseMove(self,offset):
        if offset!=self.depth:
            self.leafnodes[0].SelfChooseMove(offset)
        else:
            pi=[]
            for i in self.leafnodes:
                pi.append(i.visit)
            
            k=[self.leafnodes[pi.index(max(pi))]]
            del self.leafnodes
            self.leafnodes=k
            print(f"\n\n{k[0].move}\n\n")
            return self.leafnodes



    def OtherChooseMove(self,offset,move):
        if offset!=self.depth:
            self.leafnodes[0].OtherChooseMove(offset,move)
        else:
            temp=[]
            for i in self.leafnodes:
                temp.append(i.move)
            k=[self.leafnodes[temp.index(move)]]
            del self.leafnodes
            self.leafnodes=k
            



def inp(que):
    que.put(input())
    flag.set()

def traver(curr,offset):
    while not flag.is_set:
        
        curr.traverse(offset=offset)



flag=threading.Event()

def Play():
    game=True
    curr=tree()
    offset=0
    i=1
    trav=0

    while game:
        while trav<800*i:   
            mm=time.time()
            curr.traverse(offset=offset)
            print(time.time()-mm,offset)      
            trav+=1

        k=curr.SelfChooseMove(offset)

        offset+=1
        if Ponder:
            flag.clear()
            que=Queue()
            inpt=threading.Thread(target=inp,args=(que,))
            trave=threading.Thread(target=traver,args=(curr,offset))
            inpt.start()
            trave.start()
            inpt.join()
            move=que.get()

            
        else:
            move=input()
        k=curr.OtherChooseMove(offset,move)

        offset+=1

        i+=1
        '''z=None
        if  k[2]!=None: 
            res=k[2].result()
            if res=="1/2-1/2":
                z=0.0
            elif res=="1-0":
                z=1.0
            elif res=="0-1":
                z=-1.0
            game=False
            pass
        curr=k[3]
        if offset>512 or k[0][2][0,7,7,118]>100 or (k[0][2][0,7,7,110] and k[0][2][0,7,7,111]):
            z=0.0
            game=False
        if z!=None:

            print(z)'''
        

if __name__=="__main__":
    Play()
