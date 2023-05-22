---
layout: page
title: Masked Face recognition
description: Masked Face recogntioon problem solved using Barlow Twins technique
img: assets/img/bt_masked.png
importance: 4
category: work
---

# Project Description
When Covid-19 appeared, the most immediate defensive tool for contrasting the spread was the use of face-mask. However this introduction made more evident one of the weaknesses of our facial recognition systems, they indeed were unable to correctly classify/recognize the identity of a person wearing a face mask. We tried to overcome this problem proposing a solution applicable to the standard systems made by a convolutional neural network as feature extractor plus a final classifier such as SVM or $$k$$-NN. Our method exploits the Barlow Twins technique for learning the invariance to the presence of the face mask. We used MLFW dataset, a new synthetic dataset created starting from CALF, so our network not only learned the invariance to the face mask but also to the age. Our results aren't comparable to state-of-the-art systems however seem promising for successive studies.  
The code can be found on [GitHub](https://github.com/TiaBerte/masked-face).


# Table of Contents
1. [Problem definition](#section1)
2. [Proposed Solution](#section2)
3. [Barlow Twins](#section3)
4. [Dataset](#section4)
5. [Distributed optimization](#section5)
6. [Experiments and results](#section6)
7. [Ethical issues](#section7)
8. [Conclusions](#section8)



# Problem definition  <a name="section1"></a>
When Covid-19 disease started to spread all over the world, face-mask become one of the first defensive strategy to counteract its diffusion. Face recognition is one of the most common biometric authentication methods,  however masked face recognition is a highly challenging task since the mask occludes partially the face making impossible extracting some informative features. Previous systems indeed showed a drop in performances up to $$20\%$$ when they had to deal with a masked face.   
The most widespread solutions for facial recognition/identification were based on a features extraction phase, usually made by a convolutional neural network trained in a supervised way, plus a final classifier such as SVM or $$k$$-NN.  


# Proposed Solution  <a name="section2"></a>
We tried to propose our solution exploiting a self-supervised learning technique called [Barlow Twins](https://arxiv.org/pdf/2103.03230.pdf). The idea is pretty simple, they proposed to pass through a twins network architecture two distorted versions of the same image obtained by applying some randomly selected transformation from a set of predefined ones, and then with the help of a defined ad hoc loss, the system learns to produce the same embedding for the distorted images. It helps to produce very similar embedding for all the images of the same class since the network becomes invariant to all these transformations.  
Starting from this idea we thought to consider wearing a face mask as a transformation and to train the network to learn the invariance to the presence of the mask.  
At each step, the network receives two images of the same person, one with the mask and one without the mask. We found a tool for generating realistic masked-face starting from unmasked ones, however, to decrease the computational costs we used the pre-built dataset obtained using this tool. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/maskedface.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Face-mask application pipeline.
</div>

# Barlow Twins  <a name="section3"></a>
This section contains a brief theoretical review of the Barlow Twins' loss.  
This technique produces two distorted views for all images of a batch sampled from a dataset. The distorted views are obtained via a distribution of data augmentations. The two batches of distorted views $$Y^A$$ and $$Y^B$$ are then fed to a neural network which generates embedding $$Z^A$$ and $$Z^B$$ respectively.  
They defined the following new loss.

$$
\mathcal{L}_{BT} = \sum_i (1 - \mathcal{C}_{ii})^2 + \lambda \sum_i \sum_{j \neq i} C_{ij}^2
$$

where $$\lambda$$ is a positive constant trading off the importance of the first and second terms of the loss, and where $\mathcal{C}$ is the cross-correlation matrix computed between the outputs of the two identical networks along the batch dimension:

$$
   \mathcal{C}_{ij} = \frac{\sum_b z_{b,i}^A z_{b,j}^B}{\sqrt{\sum_b (z_{b,i}^A)^2} \sqrt{\sum_b (z_{b,i}^B)^2}} 
$$

where $$b$$ indexes batch samples and $$i, j$$ index the vector dimension of the networks’ outputs. $$\mathcal{C}$$ is a square matrix with the size the dimensionality of the network’s output, and with values comprised between $$-1$$ (i.e. perfect anti-correlation) and $$1$$ (i.e. perfect correlation).  
Intuitively, the invariance term of the objective, by trying to equate the diagonal elements of the cross-correlation matrix to $$1$$, makes the embedding invariant to the distortions applied. The redundancy reduction term, by trying to equate the off-diagonal elements of the cross-correlation matrix to $0$, decorrelates the different vector components of the embedding. This decorrelation reduces the redundancy between output units so that the output units contain non-redundant information about the sample.


# Dataset  <a name="section4"></a>
As explained in Section 2, the dataset we used is a synthetic one, called [MLFW (Masked LFW)](https://arxiv.org/pdf/2109.05804.pdf). They started from [CALFW (Cross-Aged LFW)](https://arxiv.org/pdf/1708.08197.pdf) dataset which is a variation of the widespread dataset [LFW (Labelled Face in the Wild)](http://vis-www.cs.umass.edu/lfw/lfw.pdf). CALFW contains labeled faces of famous people in different scenarios, moreover, people are present at different ages making the classification harder.  
This fact made our task more complex since the network did not need just to learn the invariance to the face mask but also to the changes due to the age, making in a lot of cases the comparison between colored and gray-scale images. The dataset required also some cleaning due to duplicated images, similar masks applied to the same image, or bad-quality images.
Starting from these images, they created a tool for generating realistic masked face through a complex algorithm that detects some key points and then applies the mask. To make more general as possible, researchers applied different mask templates to mimic the most common face masks.  
Barlow Twins requires big batch sizes to perform well, however large batch sizes are known to lead to a drop in performance. To overcome this problem, they used [LARS](https://arxiv.org/pdf/1708.03888.pdf) as an optimizer which allows the use of large batch size thanks to different adaptive learning rates for each layer. 
<div class="row">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/Frank_Abagnale_Jr_0002_0000.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/Roseanne_Barr_0004_0000.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="row">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/Tatiana_Panova_0004_0000.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/Paula_Dobriansky_0001_0000.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Example of low-quality images.
</div>

# Distributed optimization <a name="section5"></a>
To speed up this computationally expensive training, we split the work on 8 Quadro RTX 6000.  
Each GPU processes a different subset of the batch and computes the gradient; then all the gradients are averaged together to obtain the final update direction. In particular, we used a batch size of $$512$$, thus every GPU process $$64$$ samples. To speed up the communication of the gradient, we tried [PowerSGD](https://arxiv.org/pdf/1905.13727.pdf), a gradient compression algorithm. PowerSGD compresses the gradient using a low-rank approximation. For speed purposes, the low-rank approximation is estimated using the power iteration algorithm.
    
# Experiments and results <a name="section6"></a>
We finetuned a ResNet50 pre-trained on [MS-Celeb-1M](https://arxiv.org/pdf/1607.08221.pdf) and already finetuned on [VGG2](https://arxiv.org/pdf/1710.08092.pdf) dataset (weights can be found on {GitHub](https://github.com/cydonia999/VGGFace2-pytorch).   
We tested the network on identities not present in the train set. We split the test set into 2 parts, one for training the $$k$$-NN and one for testing it.  
We generated the embedding for the $k$-NN using the backbone. We selected $$k = 1$$, which means that the predicted identity is one of the closest embedding in the projected space, it was also due to the fact that many identities present just a few images, so using $$k=1$$ treat all the identities in the same way without favoring the ones with many images.  
Since SGD presented the best results, we tried to exploit at most all the images at our disposal. Instead of sampling only 2 images for each celebrities, we sampled all $$n/2$$ couples for each identity where $$n$$ is the number of image for that id. Doing so the number of couples for each epochs almost doubled.  
In Table are presented the results of our experiments.

| **Technique** | **Accuracy** |   
| --- | --- |  
| SGD | 79.72 |  
| PowerSGD | 74.72 |  
| SGD + new sampler | 76.67 |  


# Ethical issues <a name="section7"></a>
Face identification presents always some issues related to the privacy of data and to the possibility of discrimination due to the different capabilities of the system to identify persons of different ethnicity. This is usually related to the unbalanced distribution of the training set.  
We tried to derive the ethnicity of the people using an automatic method, such as [skin tone estimation](https://ieeexplore.ieee.org/document/8266229). This algorithm (repo on [GitHub](https://github.com/colin-yao/simple-skin-detection))consists of two main parts :  

* Foreground and background separation using Otsu's Binarization;  
* Pixel-wise skin classifier based on HSV and YCrCb colorspaces.  

Even if the algorithm performed very well, it was difficult to cluster the images, indeed the tone varies really gradually and it is impossible to estimate the ethnicity depending only on this feature. Moreover, the skin tone was too dependent on the light condition.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/BGR.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/final_skin.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/color.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Example of skin tone estimation.
</div>

We decided to analyze the performances of our system in each different group. We identified $5$ main groups: White/Caucasian, African American, Asiatic, Arab, and Hispanic.  
We labeled each person manually, however, this has not to be intended as a classification aim to create a new dataset but just to provide an idea of the dataset distribution. We are conscious of the possibility of making mistakes in this labeling process, our purpose was just for performance analysis and it wasn't intended to offend any of the people on the list.  

\begin{table}[h!]
    \centering
    \begin{tabular}{c|c|c}
\hline
      \multirow{2}{*}{\textbf{Ethnicity}} & \textbf{Train set} & \textbf{Test set}\\
     & \textbf{percentage} & \textbf{percentage}\\
     \hline
     African American & 8.04 & 5.22\\
     Arab & 4.06 & 6.71\\
     Asiatic & 5.49 & 2.24\\
     Hispanic & 5.11 & 6.72\\
     Caucasian & 77.30 & 79.11\\
     \hline
    \end{tabular}
    \caption{Ethnicity distribution of a dataset.}
    \label{tab:train_ethnicity }
\end{table}
Once the dataset was labeled, we checked the performances of the single ethnicity to prevent biases. The performances are reported in Table.

\begin{table}[h!]
    \centering
    \begin{tabular}{c|ccc}
    \hline
    \multirow{2}{*}{\textbf{Ethnicity}} & \multicolumn{3}{c}{\textbf{Accuracy}}\\
    \cline{2-4}
         &  \multirow{2}{*}{SGD} & Power & SGD +\\
         & & SGD & new sampler \\
    \hline
     African American & 94.44 & 94.44 & 88.89\\
     Arab & 75.00 & 85.72 & 78.57\\
     Asiatic & 83.33 & 83.33 & 83.33\\
     Hispanic & 80.00 & 70.00 & 70.00\\
     Caucasian & 79.02 & 72.72 & 75.87\\
     \hline
    \end{tabular}
    \caption{Accuracy results for each ethnicity.}
    \label{tab:ethnicity_results}
\end{table}

Unexpectedly even if the Caucasian ethnicity is the most present, it isn't the one that presents the best accuracy. It means that this approach isn't particularly biased toward some specific ethnicity or it's unable to work with a specific minority.

# Conclusions <a name="section6"></a>
We weren't interested in reaching state-of-the-art performances in masked face recognition, however, we proved the validity of our idea by exploiting a self-supervised technique combined with a metric learning approach. Our proposed solution first of all showed the effectiveness of the Barlow Twins technique moreover, we understood that also face masks and aging can be considered as simple transformations as it is for geometric or color transformations.  
One of the biggest weaknesses of our solution is related to the dataset, it's not built so well for our purpose. It required some manual cleaning and a lot of images was really similar since produced from the same original unmasked face with the addition of the same face mask but with different colors. It slows down the learning process and the generalization capability. This can be also the reason behind the decreasing in performances when we increased the dataset size. Sampling all the possible couple increased a lot the probability of sampling two really similar images which causes bad training.  
Even if PowerSGD seemed a promising technique for speeding up the training, we noticed little to no speedup in using it. We believe this is because the GPUs were located on the same node, thus they had a very high bandwidth.  
In the end, a positive aspect we think is worth noticing is the behavior related to the different ethnicity. Of course, we can notice different performances for each subgroup, however, the majority of them show better performances than the Caucasian ethnicity. We cannot conclude for sure that this approach is free from biases however it seems a promising point for successive studies.

\begin{figure}[h!]
        \subfloat[Michael Jordan face mask template 1]{%
            \includegraphics[width=.48\linewidth]{images/Michael_Jordan_0001_0000.jpg}%
       }\hfill
       \subfloat[Michael Jordan face mask template 3]{%
            \includegraphics[width=.48\linewidth]{images/Michael_Jordan_0001_0003.jpg}%
        }\\
        
        \caption{Example of different samples produced from the same original image but with different face masks.}
        \label{fig:same_images}
    \end{figure
    

