# Message Passing Markov Chains

This is pseudocode for the semantics of a send.
Our data structures are Markov Chains with a single channel
```
// from A[x==0] state to B[x==1] state
// drop message in buffered channel, or immediate rendevous
A:  B.c ! A.v
...
// ultimately, drop message into B.v.
B:  B.c ? B.v

// we sit in a loop and randomly pick A or B to be scheduled
// the variable changes are what get them into new states.
//

// all objects have variables:
// - the name of the channel from the notation: c 
// - capacity of queue: qlen
// - population of queue: q
// - values in queue: V[]
// - moved value (sent or recvd): v
// - transition probabilities: T[]
// variables like x,y are specific to the actor.
// R is a dice roll per step
// 
// implementation semantics
R = rand()

A: if A.x==0 # if this state... is a state to send from
     block = 1
     if B.qlen == 0 # rendevous
       if B.x==1   # rcvr is in recv state
         B.v = A.v # delivered
         block = 0
     else
       # if we can async deliver, then do so
       if B.q < B.qlen
         B.V[B.q] = A.v
         B.q++
         block = 0
       else
         pass # block until we can deliver
     if block == 1
       pass
     else
       if 0 <= R < A.T[0]
         A.x++
       if A.T[0] <= R < A.T[1]
         ...
       ...

B: if B.x==1 # if we are in recv state
     block = 1
     if B.qlen > 0
       if B.q > 0
         B.v = B.V[0] # delivered!
         for i = 0; i < B.qlen - 1; i++
           B.V[i] = B.V[i+1]
           B.q--
           block = 0
     else
       block = 0 # this was already rendevous delivered
     if block == 1
       pass
     else
       if 0 <= R < A.T[0]
         B.x++
       ...
              
```




