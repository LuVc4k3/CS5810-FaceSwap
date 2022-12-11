import cv2
import numpy as np

def frame_matcher(a,b):
    num_a=len(a)
    num_b=len(b)
    indexes=[0]
    a_m_b=[]
    if num_a==num_b:
        return a,b
    if num_a>num_b:
        temp=b
        b=a
        a=b
        
        
    a_m_b=[0 for i in range(num_b)]
    a_m_b[0]=a[0].astype("uint8")
    a_m_b[num_b-1]=a[num_a-1].astype("uint8")
    for i in range(1,num_a-1):
        index=int(round(((i/num_a)*num_b),0))+1
        a_m_b[index]=a[i].astype("uint8")
        indexes.append(index)
    indexes.append(num_b-1)
    index=1
    print("indexes",indexes)
    for i in range(1,len(indexes)):
        batch_range=indexes[i]-indexes[i-1]
        index=indexes[i-1]+1
        for j in range(index,index+batch_range):
            half=batch_range/2
            if j<=half:
                a_m_b[j]=a_m_b[indexes[i-1]].astype("uint8")
            else:
                a_m_b[j]=a_m_b[indexes[i]].astype("uint8")
            

    
     
    if num_a>num_b:
        return b,a_m_b
    else:
        return a_m_b,b
    
                
            
        
        
