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
|grid            |0.868         |0.769          |0.751        |
|hazelnut        |0.111         |0.1111         |0.1111        |


## Experiments

### Metal nut


