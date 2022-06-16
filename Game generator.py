import MCTSchess
import FEN
import chesspy
import chess
import numpy as np
import time


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


def TrainGame():
    game=True
    curr=Trainer()
    offset=0
    i=1
    while game:
        while curr.visit<10*i:
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
        if offset>512 or k[0][2][0,7,7,118]>100 or (k[0][2][0,7,7,110] and k[0][2][0,7,7,111]):
            z=0
            game=False
        if z!=None:

            print(z)
            break

TrainGame()


