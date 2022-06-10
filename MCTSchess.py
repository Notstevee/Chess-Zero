
import chess
import chess.pgn
import numpy as np
import gc
import chesspy 
import FEN
import tensorflow as tf



class tree:
    def __init__(self,move='',totalval=0,visit=0,leafnodes=[],depth=0,P=0):
        self.totalval=totalval
        self.visit=visit
        self.P=P
        if type(move)==list:
            self.move=move[0]
            if len(move)>1:
                self.leafnodes=[tree(move[1:])]
                self.depth=-1
            else:
                self.leafnodes=leafnodes
                self.depth=0
        else:
            self.move=move
            self.leafnodes=leafnodes
            self.depth=depth
        #self.history=history

    def PUCT(self,tree,parentvisit):
        if tree.visit==0:
            return 0.25*self.P*np.sqrt(tree.visit)/(1+tree.visit)
        return tree.totalval/tree.visit+0.25*self.P*np.sqrt(tree.visit)/(1+tree.visit)#2*np.sqrt(np.log(parentvisit)/tree.visit)

    def expand(self,history):
        tempboard=chess.Board()
        #print(lastmoves)
        for i in history:
            if i=='':
                continue
        
            tempboard.push_uci(i)
        fen=str(tempboard.fen)[34:].rstrip("')>").split()
        
        self.leafnodes=list(tempboard.legal_moves)
        model=chesspy.NetTower()
        stack=FEN.InputFeature(history,fen[1])
        policy,value=model.predict(stack)
        policy=np.transpose(policy[0],(2,1,0))

        
        PolicyVal=[]
        for i in self.leafnodes:
            PolicyVal.append(policy[FEN.ReturnArrLoc(str(i),stack,fen[1])])


        PolicyVal=PolicyVal/np.sum(PolicyVal)

        
        

        #print(list(tempboard.legal_moves))
        for i in range(len(self.leafnodes)):
            self.leafnodes[i]=tree(str(self.leafnodes[i]),depth=self.depth+1,P=PolicyVal[i])
        return self.leafnodes
        #history=self.history+[str(self.leafnodes[i])],
    
    def roll(self,lastmoves):
        model=chesspy.NetTower()
        policy,value=model.predict(FEN.InputFeature(lastmoves))

        return value
        """rollboard=chess.Board()
        rollboard.reset()
        #print(lastmoves,rollboard)

        for i in lastmoves:
            if i=='':
                continue
            move = chess.Move.from_uci(i)
            rollboard.push(move)
        m=0
        
        while m<500 and not rollboard.outcome()=='None' and not rollboard.is_insufficient_material():
            te=list(rollboard.legal_moves)
            if rollboard.legal_moves.count()<=0:
                break
            move = chess.Move.from_uci(str(te[np.random.randint(rollboard.legal_moves.count())]))
            rollboard.push(move)
            m+=1
            
        #print(rollboard,rollboard.is_insufficient_material(),rollboard.fullmove_number)

        res=rollboard.outcome()
        try:
            if res.result()=="0-1":
                return -1
            elif res.result()=="1-0":
                return 1
            else:
                return 0
        except:
            return 0"""

    def traverse(self,lastmoves=[],offset=0):
        if self.leafnodes==[] and self.depth<7+offset:
            if not self.visit:
                self.expand(lastmoves+[self.move])
                temp=[]
                return 0
          


        else:
            if not self.visit:
                m=self.roll(lastmoves+[self.move])
                self.visit+=1
                self.totalval+=m
                return m
            else:
                
                
                if self.depth<7+offset:
                    temp=[]
                    for i in self.leafnodes:
                        temp.append(self.PUCT(tree=i,parentvisit=self.visit))
                    m=self.leafnodes[temp.index(max(temp))].traverse(lastmoves+[self.move],offset=offset)
                    if not m==0:
                        self.totalval+=m
                        self.visit+=1
                    return m
                else:
                    
                    m=self.roll(lastmoves)
                    print(self.visit,"here")
                    self.visit+=1
                    self.totalval+=m
                    return m

    def SelfChooseMove(self,offset):
        if offset!=self.depth:
            self.leafnodes[0].SelfChooseMove(offset)
        else:
            temp=[]
            for i in self.leafnodes:
                temp.append(i.visit)
            
            k=[self.leafnodes[temp.index(max(temp))]]
            del self.leafnodes
            self.leafnodes=k
            return self.leafnodes[0]

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
        

               
    """def ChooseBranch(self,ChooseMove):
        
            if self.depth<0:
                self.depth-=1
                self.leafnodes[].ChooseBranch(ChooseMove)
            if not self.depth:
                self.depth-=1
                temp=[]
                for i in self.leafnodes:
                    temp.append(i.move)
                self.leafnodes=[self.leafnodes[temp.index(ChooseMove)]]
                self.leafnodes[].ChooseBranch(ChooseMove)"""
            
            
        



    def __str__(self):
        print(self.move,self.totalval,self.visit,self.leafnodes,self.depth)
        for i in self.leafnodes:
            print(i)

"""p=chess.pgn.Game()
p.push(chess.Move.from_uci("e2e4"))
node=p.add_variation(chess.Move.from_uci("e7e6"))
node=node.add_variation(chess.Move.from_uci("e7e5"))
print(node.starts_variation())
print(p.variations)
print(p)
print(p.board())"""

a=tree()

offset=0
i=1
while True:

    while a.visit<1600*i:
        a.traverse(offset=offset)
        print(a.visit)

    a.SelfChooseMove(offset)
    offset+=1
    a.SelfChooseMove(offset)
    offset+=1
    gc.collect()
    i+=1

  

print(a)