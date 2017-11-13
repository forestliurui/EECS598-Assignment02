# EECS598-Assignment02
### Q1
469s
![architecture](./plots/cpu_q1.png)
![architecture](./plots/mem_q1.png)

### Q2
521s
Comparing with Q1, checkpointing cost some time..
![architecture](./plots/cpu_q2)
![architecture](./plots/mem_q2)

### Q3
453s
Comparing with Q1, this experiment ran a little faster. I think it is because they have the same amount of data but Q3 only have 48 batches, which is smaller than Q1 and saved some time in applying the gradients. At the same time, the size of each batch increased, which resulted a longer time for each batch.
![architecture](./plots/cpu_q3)
![architecture](./plots/mem_q3)

### Q4
![architecture](./fig/tensorboard_q4)

### Q5
#### 1
3857s
![architecture](./plots/cpu_vgg_q1)
![architecture](./plots/mem_vgg_q1)

#### 2
3963s
![architecture](./plots/cpu_vgg_q2)
![architecture](./plots/mem_vgg_q2)

#### 3
3811s
![architecture](./plots/cpu_vgg_q3)
![architecture](./plots/mem_vgg_q3)

#### 4
![architecture](./fig/tensorboard_q5)

### Q6
285s
We kept 96 batches and modified batch size in each worker to 64. Thus, a batch have the same amount of data to train. In this case, it is faster than Q1. However, as there are some overhead about spliting data and computing the average gradients, it can not be 2 times faster.
![architecture](./plots/cpu_q6)
![architecture](./plots/mem_q6)
![architecture](./fig/tensorboard_q6)

### Q7
185s
We kept 96 batches and modified batch size in each worker to 32. It is also faster than Q1. For similar reason, it can not be 4 times faster.
![architecture](./plots/cpu_q7)
![architecture](./plots/mem_q7)
![architecture](./fig/tensorboard_q7)

### Q8
1053s
We use similar parameter with Q7. For similar reason, it is faster than Q5 but it is not as fast as 4 times.
![architecture](./plots/cpu_q8)
![architecture](./plots/mem_q8)
![architecture](./fig/tensorboard_q8)

### Q9
About 100 hours.
![architecture](./plots/cpu_q9)
![architecture](./plots/mem_q9)
![architecture](./fig/tensorboard_q9)
