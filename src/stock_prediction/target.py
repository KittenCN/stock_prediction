# -*- coding: utf-8 -*-
import math
import pandas as pd


try:
    from .init import *
except ImportError:

    import sys
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.init import *

np.seterr(divide='ignore', invalid='ignore')


def RD(N,D=3):   return np.round(N,D)
def RET(S,N=1):  return np.array(S)[-N]
def ABS(S):      return np.abs(S)
def LN(S):       return np.log(S)
def POW(S,N):    return np.power(S,N)
def SQRT(S):     return np.sqrt(S)
def MAX(S1,S2):  return np.maximum(S1,S2)
def MIN(S1,S2):  return np.minimum(S1,S2)
def IF(S,A,B):   return np.where(S,A,B)


def REF(S, N=1):
    return pd.Series(S).shift(N).values    

def DIFF(S, N=1):
    return pd.Series(S).diff(N).values

def STD(S,N):
    return  pd.Series(S).rolling(N).std(ddof=0).values     

def SUM(S, N):
    return pd.Series(S).rolling(N).sum().values if N>0 else pd.Series(S).cumsum().values  

def CONST(S):
    return np.full(len(S),S[-1])
  
def HHV(S,N):
    return pd.Series(S).rolling(N).max().values     

def LLV(S,N):             # LLV(C, 5) corresponds to the lowest close in the most recent 5 days     
    return pd.Series(S).rolling(N).min().values    
    
def HHVBARS(S,N):
    return pd.Series(S).rolling(N).apply(lambda x: np.argmax(x[::-1]),raw=True).values 

def LLVBARS(S,N):
    return pd.Series(S).rolling(N).apply(lambda x: np.argmin(x[::-1]),raw=True).values    
  
def MA(S,N):
    return pd.Series(S).rolling(N).mean().values  
  
def EMA(S,N):
    return pd.Series(S).ewm(span=N, adjust=False).mean().values     

def SMA(S, N, M=1):
    return pd.Series(S).ewm(alpha=M/N,adjust=False).mean().values           #com=N-M/M

def WMA(S, N):
    return pd.Series(S).rolling(N).apply(lambda x:x[::-1].cumsum().sum()*2/N/(N+1),raw=True).values 

def DMA(S, A):
    if isinstance(A,(int,float)):  return pd.Series(S).ewm(alpha=A,adjust=False).mean().values    
    A=np.array(A);   A[np.isnan(A)]=1.0;   Y= np.zeros(len(S));   Y[0]=S[0]     
    for i in range(1,len(S)): Y[i]=A[i]*S[i]+(1-A[i])*Y[i-1]
    return Y             
  
def AVEDEV(S, N):
    return pd.Series(S).rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean()).values 

def SLOPE(S, N):
    return pd.Series(S).rolling(N).apply(lambda x: np.polyfit(range(N),x,deg=1)[0],raw=True).values

def FORCAST(S, N):
    return pd.Series(S).rolling(N).apply(lambda x:np.polyval(np.polyfit(range(N),x,deg=1),N-1),raw=True).values  

def LAST(S, A, B):
    return np.array(pd.Series(S).rolling(A+1).apply(lambda x:np.all(x[::-1][B:]),raw=True),dtype=bool)
  

def COUNT(S, N):
    return SUM(S,N)    

def EVERY(S, N):
    return  IF(SUM(S,N)==N,True,False)                    
  
def EXIST(S, N):
    return IF(SUM(S,N)>0,True,False)

def FILTER(S, N):
    for i in range(len(S)): S[i+1:i+1+N]=0  if S[i] else S[i+1:i+1+N]        
    return S
  
def BARSLAST(S):
    M=np.concatenate(([0],np.where(S,1,0)))  
    for i in range(1, len(M)):  M[i]=0 if M[i] else M[i-1]+1    
    return M[1:]                       

def BARSLASTCOUNT(S):
    rt = np.zeros(len(S)+1)
    for i in range(len(S)): rt[i+1]=rt[i]+1  if S[i] else rt[i+1]
    return rt[1:]
  
def BARSSINCEN(S, N):
    return pd.Series(S).rolling(N).apply(lambda x:N-1-np.argmax(x) if np.argmax(x) or x[0] else 0,raw=True).fillna(0).values.astype(int)
  
def CROSS(S1, S2):
    return np.concatenate(([False], np.logical_not((S1>S2)[:-1]) & (S1>S2)[1:]))
    
def LONGCROSS(S1,S2,N):
    return  np.array(np.logical_and(LAST(S1<S2,N,1),(S1>S2)),dtype=bool)
    
def VALUEWHEN(S, X):
    return pd.Series(np.where(S,X,np.nan)).ffill().values  

def BETWEEN(S, A, B):
    return ((A<S) & (S<B)) | ((A>S) & (S>B))  

def TOPRANGE(S):
    rt = np.zeros(len(S))
    for i in range(1,len(S)):  rt[i] = np.argmin(np.flipud(S[:i]<S[i]))
    return rt.astype('int')

def LOWRANGE(S):
    rt = np.zeros(len(S))
    for i in range(1,len(S)):  rt[i] = np.argmin(np.flipud(S[:i]>S[i]))
    return rt.astype('int')
  

def MACD(CLOSE,SHORT=12,LONG=26,M=9):
    DIF = EMA(CLOSE,SHORT)-EMA(CLOSE,LONG);  
    DEA = EMA(DIF,M);      MACD=(DIF-DEA)*2
    return RD(DIF),RD(DEA),RD(MACD)

def KDJ(CLOSE,HIGH,LOW, N=9,M1=3,M2=3):
    RSV = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    K = EMA(RSV, (M1*2-1));    D = EMA(K,(M2*2-1));        J=K*3-D*2
    return K, D, J

def RSI(CLOSE, N=24):
    DIF = CLOSE-REF(CLOSE,1) 
    return RD(SMA(MAX(DIF,0), N) / SMA(ABS(DIF), N) * 100)  

def WR(CLOSE, HIGH, LOW, N=10, N1=6):
    WR = (HHV(HIGH, N) - CLOSE) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    WR1 = (HHV(HIGH, N1) - CLOSE) / (HHV(HIGH, N1) - LLV(LOW, N1)) * 100
    return RD(WR), RD(WR1)

def BIAS(CLOSE,L1=6, L2=12, L3=24):
    BIAS1 = (CLOSE - MA(CLOSE, L1)) / MA(CLOSE, L1) * 100
    BIAS2 = (CLOSE - MA(CLOSE, L2)) / MA(CLOSE, L2) * 100
    BIAS3 = (CLOSE - MA(CLOSE, L3)) / MA(CLOSE, L3) * 100
    return RD(BIAS1), RD(BIAS2), RD(BIAS3)

def BOLL(CLOSE,N=20, P=2):
    MID = MA(CLOSE, N); 
    UPPER = MID + STD(CLOSE, N) * P
    LOWER = MID - STD(CLOSE, N) * P
    return RD(UPPER), RD(MID), RD(LOWER)    

def PSY(CLOSE,N=12, M=6):  
    PSY=COUNT(CLOSE>REF(CLOSE,1),N)/N*100
    PSYMA=MA(PSY,M)
    return RD(PSY),RD(PSYMA)

def CCI(CLOSE,HIGH,LOW,N=14):  
    TP=(HIGH+LOW+CLOSE)/3
    return (TP-MA(TP,N))/(0.015*AVEDEV(TP,N))
        
def ATR(CLOSE,HIGH,LOW, N=20):
    TR = MAX(MAX((HIGH - LOW), ABS(REF(CLOSE, 1) - HIGH)), ABS(REF(CLOSE, 1) - LOW))
    return MA(TR, N)

def BBI(CLOSE,M1=3,M2=6,M3=12,M4=20):
    return (MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4    

def DMI(CLOSE,HIGH,LOW,M1=14,M2=6):
    TR = SUM(MAX(MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1))), ABS(LOW - REF(CLOSE, 1))), M1)
    HD = HIGH - REF(HIGH, 1);     LD = REF(LOW, 1) - LOW
    DMP = SUM(IF((HD > 0) & (HD > LD), HD, 0), M1)
    DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), M1)
    PDI = DMP * 100 / TR;         MDI = DMM * 100 / TR
    ADX = MA(ABS(MDI - PDI) / (PDI + MDI) * 100, M2)
    ADXR = (ADX + REF(ADX, M2)) / 2
    return PDI, MDI, ADX, ADXR  

def TAQ(HIGH,LOW,N):
    UP=HHV(HIGH,N);    DOWN=LLV(LOW,N);    MID=(UP+DOWN)/2
    return UP,MID,DOWN

def KTN(CLOSE,HIGH,LOW,N=20,M=10):
    MID=EMA((HIGH+LOW+CLOSE)/3,N)
    ATRN=ATR(CLOSE,HIGH,LOW,M)
    UPPER=MID+2*ATRN;   LOWER=MID-2*ATRN
    return UPPER,MID,LOWER       
  
def TRIX(CLOSE,M1=12, M2=20):
    TR = EMA(EMA(EMA(CLOSE, M1), M1), M1)
    TRIX = (TR - REF(TR, 1)) / REF(TR, 1) * 100
    TRMA = MA(TRIX, M2)
    return TRIX, TRMA

def VR(CLOSE,VOL,M1=26):
    LC = REF(CLOSE, 1)
    return SUM(IF(CLOSE > LC, VOL, 0), M1) / SUM(IF(CLOSE <= LC, VOL, 0), M1) * 100

def EMV(HIGH,LOW,VOL,N=14,M=9):
    VOLUME=MA(VOL,N)/VOL;       MID=100*(HIGH+LOW-REF(HIGH+LOW,1))/(HIGH+LOW)
    EMV=MA(MID*VOLUME*(HIGH-LOW)/MA(HIGH-LOW,N),N);    MAEMV=MA(EMV,M)
    return EMV,MAEMV

def DPO(CLOSE,M1=20, M2=10, M3=6):
    DPO = CLOSE - REF(MA(CLOSE, M1), M2);    MADPO = MA(DPO, M3)
    return DPO, MADPO

def BRAR(OPEN,CLOSE,HIGH,LOW,M1=26):
    AR = SUM(HIGH - OPEN, M1) / SUM(OPEN - LOW, M1) * 100
    BR = SUM(MAX(0, HIGH - REF(CLOSE, 1)), M1) / SUM(MAX(0, REF(CLOSE, 1) - LOW), M1) * 100
    return AR, BR

def DFMA(CLOSE,N1=10,N2=50,M=10):
    DIF=MA(CLOSE,N1)-MA(CLOSE,N2); DIFMA=MA(DIF,M)
    return DIF,DIFMA

def MTM(CLOSE,N=12,M=6):
    MTM=CLOSE-REF(CLOSE,N);         MTMMA=MA(MTM,M)
    return MTM,MTMMA

def MASS(HIGH,LOW,N1=9,N2=25,M=6):
    MASS=SUM(MA(HIGH-LOW,N1)/MA(MA(HIGH-LOW,N1),N1),N2)
    MA_MASS=MA(MASS,M)
    return MASS,MA_MASS
  
def ROC(CLOSE,N=12,M=6):
    ROC=100*(CLOSE-REF(CLOSE,N))/REF(CLOSE,N);    MAROC=MA(ROC,M)
    return ROC,MAROC  

def EXPMA(CLOSE,N1=12,N2=50):
    return EMA(CLOSE,N1),EMA(CLOSE,N2);

def OBV(CLOSE,VOL):
    return SUM(IF(CLOSE>REF(CLOSE,1),VOL,IF(CLOSE<REF(CLOSE,1),-VOL,0)),0)/10000

def MFI(CLOSE,HIGH,LOW,VOL,N=14):
    TYP = (HIGH + LOW + CLOSE)/3
    V1=SUM(IF(TYP>REF(TYP,1),TYP*VOL,0),N)/SUM(IF(TYP<REF(TYP,1),TYP*VOL,0),N)  
    return 100-(100/(1+V1))     
  
def ASI(OPEN,CLOSE,HIGH,LOW,M1=26,M2=10):
    LC=REF(CLOSE,1);      AA=ABS(HIGH-LC);     BB=ABS(LOW-LC);
    CC=ABS(HIGH-REF(LOW,1));   DD=ABS(LC-REF(OPEN,1));
    R=IF( (AA>BB) & (AA>CC),AA+BB/2+DD/4,IF( (BB>CC) & (BB>AA),BB+AA/2+DD/4,CC+DD/4));
    X=(CLOSE-LC+(CLOSE-OPEN)/2+LC-REF(OPEN,1));
    SI=16*X/R*MAX(AA,BB);   ASI=SUM(SI,M1);   ASIT=MA(ASI,M2);
    return ASI,ASIT   
    
def XSII(CLOSE, HIGH, LOW, N=102, M=7):
    AA  = MA((2*CLOSE + HIGH + LOW)/4, 5)
    TD1 = AA*N/100;   TD2 = AA*(200-N) / 100
    CC =  ABS((2*CLOSE + HIGH + LOW)/4 - MA(CLOSE,20))/MA(CLOSE,20)
    DD =  DMA(CLOSE,CC);    TD3=(1+M/100)*DD;      TD4=(1-M/100)*DD
    return TD1, TD2, TD3, TD4  
    
def LLV(S, N):
    if isinstance(N, (int, float)):
        return pd.Series(S).rolling(N).min().values
    else:
        res = np.repeat(np.nan, len(S))
        for i in range(len(S)):
            if (not np.isnan(N[i])) and N[i] <= i + 1:
                res[i] = S[i + 1 - N[i]:i + 1].min()
        return res


def DSMA(X, N):
    a1 = math.exp(- 1.414 * math.pi * 2 / N)
    b1 = 2 * a1 * math.cos(1.414 * math.pi * 2 / N)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3        
    Zeros = np.pad(X[2:] - X[:-2],(2,0),'constant')          
    Filt = np.zeros(len(X))
    for i in range(len(X)):
        Filt[i] = c1 * (Zeros[i] + Zeros[i-1]) / 2 + c2 * Filt[i-1] + c3 * Filt[i-2]    
    RMS = np.sqrt(SUM(np.square(Filt), N) / N)
    ScaledFilt = Filt / RMS
    alpha1 = np.abs(ScaledFilt) * 5 / N    
    return DMA(X, alpha1)    

def SUMBARSFAST(X, A): 
    if any(X<=0):   raise ValueError('all elements in X must be greater than zero')
    X = np.flipud(X)
    length = len(X)
    if isinstance(A * 1.0, float):  A = np.repeat(A, length)
    A = np.flipud(A)
    sumbars = np.zeros(length)
    Sigma = np.insert(np.cumsum(X), 0, 0.0)
    for i in range(length):
        k = np.searchsorted(Sigma[i + 1:], A[i] + Sigma[i])
        if k < length - i:
            sumbars[length - i - 1] = k + 1
    return sumbars.astype(int)  
