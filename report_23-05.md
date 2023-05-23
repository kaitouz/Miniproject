# Report 23-05

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
> Đây là kết quả ảnh sinh được x_hat (x_hat = Generator(x)) 

### metal_nut
![Input](./Images/Experiments/metal_nut/input_test.png "Input")

![Output](./Images/Experiments/metal_nut/output_test.png "Output")

### hazelnut
![Input](./Images/Experiments/hazelnut/input_test.png "Input")

![Output](./Images/Experiments/hazelnut/output_test.png "Output")

### grid
![Input](./Images/Experiments/grid/input_test.png "Input")

![Output](./Images/Experiments/grid/output_test.png "Output")


## Training process
> Đây là kết quả ảnh sinh được qua các epoch 

### metal_nut

|Input                                              |              Output                                 |
|:-------------------------------------------------:|:---------------------------------------------------:|
|![](./Images/Experiments/metal_nut/Train/inputs_0.png)  |  ![](./Images/Experiments/metal_nut/Train/outputs_0.png) |
|![](./Images/Experiments/metal_nut/Train/inputs_20.png)  |  ![](./Images/Experiments/metal_nut/Train/outputs_20.png) |
|![](./Images/Experiments/metal_nut/Train/inputs_40.png)  |  ![](./Images/Experiments/metal_nut/Train/outputs_40.png) |
|![](./Images/Experiments/metal_nut/Train/inputs_60.png)  |  ![](./Images/Experiments/metal_nut/Train/outputs_60.png) |
|![](./Images/Experiments/metal_nut/Train/inputs_80.png)  |  ![](./Images/Experiments/metal_nut/Train/outputs_80.png) |
|![](./Images/Experiments/metal_nut/Train/inputs_100.png)  |  ![](./Images/Experiments/metal_nut/Train/outputs_100.png) |
|![](./Images/Experiments/metal_nut/Train/inputs_120.png)  |  ![](./Images/Experiments/metal_nut/Train/outputs_120.png) |
|![](./Images/Experiments/metal_nut/Train/inputs_140.png)  |  ![](./Images/Experiments/metal_nut/Train/outputs_140.png) |
|![](./Images/Experiments/metal_nut/Train/inputs_160.png)  |  ![](./Images/Experiments/metal_nut/Train/outputs_160.png) |
|![](./Images/Experiments/metal_nut/Train/inputs_180.png)  |  ![](./Images/Experiments/metal_nut/Train/outputs_180.png) |


### hazelnut

|Input                                              |              Output                                 |
|:-------------------------------------------------:|:---------------------------------------------------:|
|![](./Images/Experiments/hazelnut/Train/inputs_0.png)  |  ![](./Images/Experiments/hazelnut/Train/outputs_0.png) |
|![](./Images/Experiments/hazelnut/Train/inputs_40.png)  |  ![](./Images/Experiments/hazelnut/Train/outputs_40.png) |
|![](./Images/Experiments/hazelnut/Train/inputs_80.png)  |  ![](./Images/Experiments/hazelnut/Train/outputs_80.png) |
|![](./Images/Experiments/hazelnut/Train/inputs_120.png)  |  ![](./Images/Experiments/hazelnut/Train/outputs_120.png) |
|![](./Images/Experiments/hazelnut/Train/inputs_160.png)  |  ![](./Images/Experiments/hazelnut/Train/outputs_160.png) |
|![](./Images/Experiments/hazelnut/Train/inputs_200.png)  |  ![](./Images/Experiments/hazelnut/Train/outputs_200.png) |
|![](./Images/Experiments/hazelnut/Train/inputs_240.png)  |  ![](./Images/Experiments/hazelnut/Train/outputs_240.png) |
|![](./Images/Experiments/hazelnut/Train/inputs_280.png)  |  ![](./Images/Experiments/hazelnut/Train/outputs_280.png) |


