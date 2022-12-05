def frame_matcher(a,b):
    #Call with b>a
    num_a=len(a)
    num_b=len(b)
    indexes=[0]
    if num_b>num_a:
        a_m_b=[i for i in range(num_b)]
        a_m_b[0]=a[0]
        a_m_b[num_b-1]=a[num_a-1]
        for i in range(num_a):
            index=((i/num_a)*10)+1
            a_m_b[index]=a[i]
            indexes.append(index)
        indexes.append(num_b-1)
        index=1
        for i in range(1,len(indexes)):
            batch_range=indexes[i]-indexes[i-1]
            margin=1/(batch_range)
            for j in range(batch_range):
                a_m_b[index]=margin*a_m_b[i-1]+((1-margin)*a_m_b[i])
                index+=1
    return a_m_b,b
                
            
        
        
