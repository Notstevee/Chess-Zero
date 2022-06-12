import chess
import numpy as np


def ReturnArrLoc(uci,stack,colour):
    stack=stack[:,::-1,::-1]
    StrUCI=str(uci)
    OC,OR,NC,NR=ord(StrUCI[0])-97,8-int(StrUCI[1]),ord(StrUCI[2])-97,8-int(StrUCI[3])
    diffC=NC-OC
    diffR=NR-OR
    layer,row,col=0,OR,OC
    if colour=="b":
        stack=stack[:,::-1,::-1]
    if len(StrUCI)>4 and not StrUCI[4]=='q':
        Prom=StrUCI[4]
        piece='rbn'
        layer=67+diffC*3+piece.index(Prom)
    else:
        
        if (abs(diffC)-abs(diffR))==0 or not diffC or not diffR:
            if diffC*diffR==1:
                if diffC>0:
                    layer=8-1+diffC
                else:
                    layer=8*5-1-diffC
            elif diffC*diffR==-1:
                if diffC>0:
                    layer=8*3-1+diffC
                else:
                    layer=8*7-1-diffC

            elif diffC==0:
                if diffC>0:
                    layer=8*2-1+diffC
                else:
                    layer=8*6-1-diffC
            else:
                if diffR>0:
                    layer=8*4-1+diffR
                else:
                    layer=-1-diffR
        if (abs(diffC)==2 and abs(diffR)==1) or (abs(diffC)==1 and abs(diffR)==2):
            layer=56+(0 if diffC==2 else 1)+((0 if diffC>0 else 1)+(0 if diffR>0 else 2))*2
    
    return (layer,row,col)




def InputFeature(history):
    layord="PBNRQKpbnrqk"
    if history[0]=='':
        history.pop(0)
    curr=chess.Board()


    if len(history)<15:
        if len(history)==0:
            stack=np.zeros((14*8,8,8))
        else:
            stack=np.zeros((14*(8-(len(history)+1)//2),8,8))
    else:
        stack=np.array(())
    fenlist=[]
    for i in history:

        curr.push_uci(i)
        fen=str(curr.fen)[34:].rstrip("')>").split()
        fenlist.append(fen[:4])
    last8=(fenlist[:-16:-2])[::-1]
    lenlast8=len(last8)
    for fen in last8:

        
        if fen in fenlist[:-(lenlast8*2-1)]:
            repicount=fenlist[:-(lenlast8*2-1)].count(fen)
            #print(repicount)
        else:
            repicount=0
        fenlist.append(fen[:4])
        tempstack=np.zeros((12,8,8))
        rows=fen[0].split("/")
        rowc=0
        for row in rows:
            colc=0
            for col in row:
               if col<='9' and col>='0':
                   colc+=int(col)
               else:
                    tempstack[layord.index(col),rowc,colc]=1
                    colc+=1
            rowc+=1     
        #if fen[1]=="b":
            #tempstack=tempstack[:,::-1,::-1]
            #x,y=np.vsplit(tempstack,2)
            #tempstack=np.vstack((y,x))
        if repicount==0:
            tempstack=np.vstack((tempstack,np.zeros((2,8,8))))
        if repicount==1:
            tempstack=np.vstack((tempstack,np.vstack((np.zeros((1,8,8)),np.ones((1,8,8))))))
        if repicount==2:
            tempstack=np.vstack((tempstack,np.vstack((np.ones((1,8,8)),np.zeros((1,8,8))))))
        if repicount==3:
            tempstack=np.vstack((tempstack,np.vstack((np.ones((1,8,8)),np.ones((1,8,8))))))
        if not stack.size>0:
            stack=tempstack
        else:
            stack=np.vstack((stack,tempstack))
        lenlast8-=1
    fen=str(curr.fen)[34:].rstrip("')>").split()    
    if fen[1]=='w':
        stack=np.vstack((stack,np.zeros((1,8,8))))
        
    else:
        stack=np.vstack((stack,np.ones((1,8,8))))
    stack=np.vstack((stack,np.ones((1,8,8))*int(fen[5])))
        
    
    if "K" in fen[2]:
        tempstack=np.ones((1,8,8))
    else:
        tempstack=np.zeros((1,8,8))
    if "Q" in fen[2]:
        tempstack=np.vstack((tempstack,np.ones((1,8,8))))
    else:
        tempstack=np.vstack((tempstack,np.zeros((1,8,8))))
    if "k" in fen[2]:
        tempstack=np.vstack((tempstack,np.ones((1,8,8))))
    else:
        tempstack=np.vstack((tempstack,np.zeros((1,8,8))))
    if "q" in fen[2]:
        tempstack=np.vstack((tempstack,np.ones((1,8,8))))
    else:
        tempstack=np.vstack((tempstack,np.zeros((1,8,8))))
    if fen[1]=="b":
            tempstack=tempstack[:,::-1,::-1]
            x,y=np.vsplit(tempstack,2)
            tempstack=np.vstack((y,x))    
    stack=np.vstack((stack,tempstack))
    stack=np.vstack((stack,np.ones((1,8,8))*int(fen[4])))
    if fen[1]=="b":
            stack=stack[:,::-1,::-1]
    stack=np.transpose(stack,(2,1,0))            
    stack=np.expand_dims(stack,axis=0)
    

    return stack,str(curr.fen)[34:].rstrip("')>")
       

