def frame_matcher(a,b):
    #Call with b>a
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
    a_m_b[0]=a[0]
    a_m_b[num_b-1]=a[num_a-1]
    for i in range(1,num_a-1):
        index=int(round(((i/num_a)*num_b),0))+1
     #   print("index",index)
        a_m_b[index]=a[i]
        indexes.append(index)
    indexes.append(num_b-1)
    index=1
    print("a",a)
    print("a_m_b",a_m_b)
    print("b",b)
    print("indexes",indexes)
    for i in range(1,len(indexes)):
        batch_range=indexes[i]-indexes[i-1]
        #print(indexes[i],indexes[i-1])
        margin=1/(batch_range)
        for j in range(batch_range):
            a_m_b[index]=(margin*a_m_b[i-1]+((1-margin)*a_m_b[i])).astype("uint8")
            #print(a_m_b[index].dtype)
            #print(b[index].dtype)
            index+=1
    #print("a_m_b",a_m_b)
    #print("b",b)
    #print("indexes",indexes)
    if num_a>num_b:
        return b,a_m_b
    else:
        return a_m_b,b
    
                
            
        
        
