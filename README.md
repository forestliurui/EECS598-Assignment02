# EECS598-Assignment02
### Q1
469s

### Q2
521s
Comparing with Q1, checkpointing cost some time..

### Q3
453s
Comparing with Q1, this experiment ran a little faster. I think it is because they have the same amount of data but Q3 only have 48 batches, which saved some time in applying the gradients.

### Q5
#### 1
3857s

#### 2
3963s

#### 3
3811s

### Q6
285s
We kept 96 batches and modified batch size in each worker to 64. Thus, a batch have the same amount of data to train. In this case, it is faster than Q1. However, as there are some overhead about spliting data and computing the average gradients, it can not be 2 times faster.

### Q7
185s
We kept 96 batches and modified batch size in each worker to 32. It is also faster than Q1. For similar reason, it can not be 4 times faster.

### Q8
1053s
We use similar parameter with Q7. For similar reason, it is faster than Q5 but it is not as fast as 4 times.
