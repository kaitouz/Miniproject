# Report 23-05

## A Summary of Completed Tasks

### Implementation and evaluation STFPM Model on MVTEC Dataset

>Config:
>- number of epochs : 500
>- random seed :  42
>- size of image after transform : 256x256
>- size of loss image : 64x64
>- Teacher/Student model : Resnet18 
>- Teacher pretrained weight : IMAGENET1k_v1
>- optimizer : SGD (lr=0.4, momentum=0.9, weight_decay=1e-4)


|Class name      |F1 score       |Accuracy      | AUC          |
|----------------|--------------|---------------|--------------|
|metal_nut       |0.894         |0.8            |0.399         |
|grid            |0.9912        |0.9744         |0.9958        |
|hazelnut        |0.778         |0.627          |0.666         |

### Examples
![Input](./Images/Experiments_30-05/STFPM_MVTEC/example_grid.png "Input")
![Ouput](./Images/Experiments_30-05/STFPM_MVTEC/example_grid_output.png "Ouput")

Score Histogram: Normal (Orange) vs. Anomalous (Blue) Images 

![Distribution](./Images/Experiments_30-05/STFPM_MVTEC/distribution_grid.png "Ouput")


### Implemetation and evaluation GANomaly Model on Station Dataset


### Implemetation and evaluation STFPM Model on Station Dataset
