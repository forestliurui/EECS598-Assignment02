# EECS598-Assignment02
### Q1
469s
![architecture](./plots/cpu_q1.png)
![architecture](./plots/mem_q1.png)

### Q2
521s
Comparing with Q1, checkpointing cost some time..
![architecture](./plots/cpu_q2.png)
![architecture](./plots/mem_q2.png)

### Q3
453s

Comparing with Q1, this experiment ran a little faster. I think it is because they have the same amount of data but Q3 only have 48 batches, which is smaller than Q1 and saved some time in applying the gradients. At the same time, the size of each batch increased, which resulted a longer time for each batch.
![architecture](./plots/cpu_q3.png)
![architecture](./plots/mem_q3.png)

### Q4
![architecture](./fig/tensorboard_q4.png)

### Q5
#### Repeating Q1
3857s
![architecture](./plots/cpu_vgg_q1.png)
![architecture](./plots/mem_vgg_q1.png)

#### Repeating Q2
3963s
![architecture](./plots/cpu_vgg_q2.png)
![architecture](./plots/mem_vgg_q2.png)

#### Repeating Q3
3811s
![architecture](./plots/cpu_vgg_q3.png)
![architecture](./plots/mem_vgg_q3.png)

#### Repeating Q4
![architecture](./fig/tensorboard_q5.png)

### Q6
285s

We kept 96 batches and modified batch size in each worker to 64. Thus, a batch have the same amount of data to train. In this case, it is faster than Q1. However, as there are some overhead about spliting data and computing the average gradients, it can not be 2 times faster.
![architecture](./plots/cpu_q6.png)
![architecture](./plots/mem_q6.png)
![architecture](./fig/tensorboard_q6.png)

### Q7
185s

We kept 96 batches and modified batch size in each worker to 32. It is also faster than Q1. For similar reason, it can not be 4 times faster.
![architecture](./plots/cpu_q7.png)
![architecture](./plots/mem_q7.png)
![architecture](./fig/tensorboard_q7.png)

### Q8
1053s

We use similar parameter with Q7. For similar reason, it is faster than Q5 but it is not as fast as 4 times.
![architecture](./plots/cpu_q8.png)
![architecture](./plots/mem_q8.png)
![architecture](./fig/tensorboard_q8.png)

### Q9
About 100 hours.
![architecture](./plots/cpu_q9.png)
![architecture](./plots/mem_q9.png)
![architecture](./fig/tensorboard_q9.png)
