# Report


## Kết quả

> Kết quả train GANomaly với config:
>- number of epochs : 100
>- random seed :  42
>- size of image after crop : 64
>- size of latent vector  : 100
>- default: lr : 0.0002, b1 : 0.5, b2 : 0.999
>- w_adv = 1, w_con = 40, w_enc = 1
>- optimizer : Adam


|Class name      |F1 score       |Accuracy      | AUC          |
|----------------|--------------|---------------|--------------|
|metal_nut       |0.894         |0.8            |0.399         |
|grid            |0.868         |0.769          |0.751         |
|hazelnut        |0.778         |0.627          |0.666         |


## Experiments

### metal_nut
![Input](./Images/Experiments/metal_nut/input_test.png)

![Output](./Images/Experiments/metal_nut/output_test.png)

### hazelnut
![Input](./Images/Experiments/hazelnut/input_test.png)

![Output](./Images/Experiments/hazelnut/output_test.png)

### grid
![Input](./Images/Experiments/grid/input_test.png)

![Output](./Images/Experiments/grid/output_test.png)