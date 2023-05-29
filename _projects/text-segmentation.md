---
layout: page
title: Text segmentation using DETR
description: Text segmentation problem solved using DETR architecture
img: assets/img/detr.jpg
importance: 1
category: work
---

# Project description
Less than a third of high school seniors in the U.S. are proficient writers, according to the National Assessment of Educational Progress1. Low-income, Black, and Hispanic students fare even worse, with less than 15 percent demonstrating writing proficiency. Since good writing skills are important for success, unbiased support for writing could have a positive impact on society. One way to achieve this is via automated feedback tools, which often require a pipeline of tools to be deployed. We decided to contribute to the cause, building and releasing a model for text segmentation and argumentative element classification. Our model is an end-to-end neural network based on Transformers, capable of identifying the argumentative elements of a text and classifying them. We address the problem, proposing a new architecture inspired by successful ones for object detection in Computer Vision. Regarding the dataset, we used the one provided by the Feedback Prize competition on Kaggle.
The code can be found on [GitHub](https://github.com/TiaBerte/text-segmentation).


# Table of Contents
1. [Background](#section1)
2. [Proposed Solution](#section2)
3. [Barlow Twins](#section3)
4. [Dataset](#section4)
5. [Distributed optimization](#section5)
6. [Experiments and results](#section6)
7. [Ethical issues](#section7)
8. [Conclusions](#section8)

## Background  <a name="section1"></a>
### Text Segmentation 
 Text Segmentation is an underexplored area of research. The goal of this task is to identify argumentative elements from a document. Closely related is the \textit{Argument Mining} task, from which we take inspiration. In the latter, the aim is also to extract a structure of argumentative elements, while in our task this is not necessary (even if an understanding of the structure of the text can help achieve better performances). One notable reference is [(Ruiz et al., 2021)](https://arxiv.org/pdf/2011.13187.pdf) which shows the effectiveness of transformer-based architectures for the Argument Mining tasks. [(Galassi et al., 2021)](https://arxiv.org/ftp/arxiv/papers/2102/2102.12227.pdf) proposed {\sc ResAttArg}, a state-of-the-art architecture for Augmentative Mining. The main limitation is that it doesn't scale up well with large documents. We aim to overcome this issue using [Longformer](https://arxiv.org/pdf/2004.05150.pdf).  

### Longformer 
Longformers are a particular type of transformer proposed to scale up for long documents. To do so, they use two types of attention, a local and a global one. The local attention is standard sliding-window attention, while the global one is a sparse pattern to attend to all the documents. With this trick, the scales linearly with the size of the sequence to process, making it suitable for our task. However, choosing the global attention pattern is a new implementation choice that must be tuned.  

### DETR
[DETR](https://arxiv.org/pdf/2005.12872.pdf) is a state-of-the-art architecture for object detection proposed by FAIR. In a nutshell, it involves a backbone, typically a ResNet50 or a ResNet101 , that extracts the important feature of the image. These features are passed to an encoder-decoder transformer that extracts the box predictions. In its simplicity, DETR doesn't require a complex and hand-crafted pipeline for box extraction and it is highly parallelizable. However, it requires setting a maximum number of box predictions. For training, DETR uses two different losses. The first one is the cross-entropy loss for the classification of the boxes. The second one is a linear combination of the $$L_1$$ norm between the predicted center and length of the bounding boxes and the ground truth plus the Generalized Intersection-Over-Union between the true and the predicted boxes.   
To match the boxes with the ground truth, DETR uses the Hungarian algorithm. So the overall loss is:
$$
    \mathcal{L}_{Hungarian}(y, \hat{y}) = \sum^N_{i=0}\big[ -\lambda_{CE} \log p_{\hat{\sigma}(i)(c_i)} + 1_{\{c_i \neq \emptyset\}} \mathcal{L}_{box}(b_i, \hat{b}_{\hat{\sigma}(i)}) \big]
$$

where $$\lambda_{CE}$$ is a hyperparameter and $$\hat{\sigma}$$ is the optimal assignment computed as:
$$
    \hat{\sigma} = \argmin_{\sigma \in \Theta} \sum_i^N \mathcal{L}_{match}(y_i, \hat{y}_{\sigma_i})
$$

and $$\mathcal{L}_{box}$$ is:

$$
    \mathcal{L}_{box}(b_i, \hat{b}_{\sigma_i}) = \lambda_{iou}\mathcal{L}_{iou}(b_i, \hat{b}_{\sigma_i}) + \lambda_{L1}||(b_i, \hat{b}_{\sigma_i})||_1
$$

with $$\lambda_{iou}$$ and $$\lambda_{L1}$$ hyperparameters.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/detr2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
     DETR architecture representation. The model predicts a fixed number of boxes and then classifies them with a special class to discard boxes.
</div>


## System description <a name="section1"></a>

### Dataset
The dataset contains more than 15k argumentative essays written by U.S students in grades 6-12. The essays were annotated by expert raters for elements commonly found in argumentative writing.  
As always in kaggle competition, the test set is private, you have access to a certain amount of evaluations of your model, however we decided to create also the test set from the whole dataset. Doing so we could analyze the error of our model on the test set, which wouldn't be possible otherwise.  
As shown in figure, the documents have a quite spread distribution of lengths, with the majority having less than 1000 words. Instead, the number of argumentative units in each document follows a Gaussian distribution. 

<div class="row">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/documents_length_his.pdf" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.html path="assets/img/arg_units_per_doc_hist.pdf" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Number of words in each document and number of argumentative units for each document.
</div>

### Annotations
The annotations report the following information:
* _discourse\_start_ - character position where discourse element begins in the essay response;   
* _discourse\_end_ - character position where discourse element ends in the essay response;  
* _discourse\_type_ - classification of discourse element;

Those kinds of annotation are similar to the ones commonly found in object detection tasks. In fact, we can make a parallelism between pixel position in the images of bounding boxes, and character position in the document of the discourse units for details. The major difference is that here the segmentation boxes are 1-dimensional, instead of the 2-D bounding boxes in image object detection. So we augment the annotations by including two new features:  
* _box\_center_ - character position of the center of the discourse;  
* _box\_length_ - number of characters in the discourse;  

Those are the features that our model will train on and predict.  


### Classes
The discourses are classified using the following categorization:  

 * **Lead** - an introduction that begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis;  
* **Position** - an opinion or conclusion on the main question;  
* **Claim** - a claim that supports the position;  
* **Counterclaim** - a claim that refutes another claim or gives an opposing reason to the position;  
* **Rebuttal** - a claim that refutes a counterclaim;  
* **Evidence** - ideas or examples that support claims, counterclaims, or rebuttals;    
* **Concluding Statement** - a concluding statement that restates the claims;

Some parts of the essays will be unannotated (i.e., they do not fit into one of the classifications above).

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/class_freqs.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
     Distribution of the classes.
</div>


The classes _Claim_ and  _Evidence_ are by far the most common in the dataset.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/discourse_rel_pos.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
     Average start position, end position and length (in characters) normalized with respect to the document length (with standard deviations)
</div>


In figure  we can see that on average the documents respect the structure of the argumentative essay. They start with the lead section, to introduce the thesis, followed by positions, claims and evidence to support the thesis. Then comes the counterclaims and the rebuttals, that are needed to give more credibility to the thesis, and then they end with the concluding statements.   
The central sections have a large standard deviation, as they can be mixed and even repeated multiple times in an essay. As we can expect, the larger sections are the Lead, the Concluding statements and the Evidence because they are the main parts of an argumentative essay, instead, the other sections are just needed to introduce the discussion.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/word_class_correlation.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
     Correlation between words and classes.
</div>


We also evaluated how the words correlate with the classes. As correlation metrics, we take inspiration from the Tf-idf statistic, which is intended to reflect how important a word is to a document in a collection or corpus. We evaluated the term frequency in the class as   
$$\mathrm {tf_{c, t}} = \frac{f_{c, t}}{\sum_t' f_{c, t'}}$$  

with $$f_{c, t}$$ as the counts of the occurrence of the term $$t$$ in the class $$c$$. Then instead of the inverse-document-frequency term, we weighted the frequencies with an inverse-discourse-frequency in as

$$\mathrm {idf_t} = \log\frac{|D|}{|\{d \in D : t \in d\}|}$$

with $$D$$ the set of all discourses in the dataset.  
Some interesting correlations emerge. For example, in the _Counterclaim_ class, the most correlated are _argue, say, although, may, might_, which are all words that introduce a hypothetical opposite opinion. In _Position_ we can find words like _believe, idea, agree_, instead the top correlation for _Rebuttal_ and _Concluding Statement_ are respectively _however_ and _conclusion_, as one might expect. Instead _Claim_, _Evidence_ and _Lead_ correlate with more generic words, maybe because in those classes there is more freedom in the content.


## Architecture
Starting from the DETR proposal, we tried to adapt this architecture to our task. Even if CNNs still represent the SoTA in some computer vision tasks, they don't work so well in NLP tasks, where transformers became the dominant paradigm in the last years. Following these premises, we decide to replace the ResNet feature extractor in favour of an advanced version of transformers called Longformer. Actually, the Longformer architecture is nothing different from the BERT model, the innovative idea is related to the attention mechanism. The original Transformer model has a self-attention component with $$O(n^2)$$ time and memory complexity where $n$ is the input sequence length, the Longformer model instead tries to combine local attention with a sliding windows approach, whose complexity is $$O(n)$$, plus global attention related to preselected input locations in order to provide inductive bias to the model, since the number of these inputs is small related to the length of the text, the model complexity is still $$O(n)$$. Since our task requires dealing with long texts, in which each sentence has to be analyzed with respect to the whole text, Longformer is a good trade-off.
Following the same intuition, since the Longformer is present in the encoder-decoder variant, we decided to use it also for the second part of the architecture.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/attention.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
     Comparison between Longformer attention mechanism and previous proposal.
</div>


## From Object Detection to Text Segmentation
As said before we have made parallelism between object detection and text segmentation. The main difference is that in object detection the boxes are 2-D, instead in text segmentation they are 1-D. The regression head of DETR predicts 4 parameters that are the center and the dimensions of the bounding boxes, normalized in the range $$(0, 1)$$ with respect to the dimensions of the image. We changed it to predict only 2 parameters, the center and the length of the "segmentation boxes". We choose to consider those values normalized with respect to the position of the words in the document tokenized by the LongormerTokenizer that we used during training.  
With this in mind, we adapted all the operations that should have involved the 2-D bounding boxes, in order to make them work with our 1-dimensional "boxes". The main function to modify is the $$\mathcal{L}_{iou}$$ which evaluates the Intersection Over Union between bounding boxes, and it's used both in the Hungarian Matcher and in the Loss function. In particular, we used the [Generalized Intersection Over Union](https://arxiv.org/pdf/1902.09630.pdf), that overcomes the issues of the plateau that IoU has in the case of nonoverlapping bounding boxes. GIoU in the general case is defined as follows: for two arbitrary convex shapes (volumes) $$A, B \subseteq \mathbb{S} \in \mathbb{R}^n$$, find the smallest convex shapes $$C \subseteq \mathbb{S} \in \mathbb{R}^n$$ enclosing both $$A$$ and $$B$$. Then calculate a ratio between the volume (area) occupied by $$C$$ excluding $$A$$ and $$B$$ and divide by the total volume (area) occupied by $$C$$. This represents a normalized measure that focuses on the empty volume (area) between $$A$$ and $$B$$. Finally, GIoU is attained by subtracting this ratio from the IoU value. In summary:   

$$GIoU = \frac{|A \cap B|}{|A \cup B|} - \frac{|C \setminus A \cup B|}{|C|}$$

$$GIoU$$ is both a metric and a loss, with $$\mathcal{L}_{GIoU}(A, B) = 1 - GIoU(A, B)$$. It shows a consistent improvement in performance measures on popular object detection benchmarks by incorporating this generalized IoU as a loss into the state-of-the-art object detection frameworks.   
To adapt this loss to our task, we considered to have boxes in the start-end format $$b_i = (s_i, e_i)$$ obtained from the predictions in format center-length with the equations $$s_i = c_i - \frac{l_i}{2}$$ and $$e_i = c_i + \frac{l_i}{2}$$ so it's guaranteed that $$s_i \leq e_i$$. We evaluate the $$GIoU$$ as follows:  
* $$|b_i| = l_i = s_i - e_i$$ - The area of the boxes is their length;  
* $$|b_i \cap b_j| = \max(0, \max(s_i, s_j) - \min(e_i, e_j))$$ - The intersection of boxes is the length of the segment they have in common;  
* $$|b_i \cup b_j| = |b_i| + |b_j| - |b_i \cap b_j|$$;
* $$|C| = \min(s_i, s_j) - \max(e_i, e_j)$$;
* $$|C \setminus b_i \cup b_j| = |C| - |b_i \cup b_j|$$ - The length of the segment that, if any, separates $b_i$ and $b_j$. With perfect overlap, this factor is equal to 0.  

Since every token can belong to at most one argumentative part, we also experiment with a new loss $$\mathcal{L}_{Overlap}$$ that penalizes overlapping between the boxes issued by the network. Thus this loss is simply the mean of the \textit{GIoU} between every couple of boxes issued by the network. 

## Evaluation metrics

To evaluate our model we used the metric described in the competition web page. These metrics evaluate the overlap between ground truth and predicted word indices.  
For each sample, all ground truths and predictions for a given class are compared.  
If the overlap between the ground truth and prediction is $$\geq 0.5$$, and the overlap between the prediction and the ground truth $$\geq 0.5$$, the prediction is a match and is considered a true positive. If multiple matches exist, the match with the highest pair of overlaps is taken.  
Any unmatched ground truths are false negatives and any unmatched predictions are false positives.  
The final score is arrived at by calculating TP/FP/FN for each class, then taking the macro F1 score across all classes.  
Annotations and predictions must be in word indices, that is calculated by using Python's \verb|.split()| function and taking the indices of the words of the discourse unit in the resulting list. The two overlaps are calculated by taking the  \verb|set()| of each list of indices in a ground truth/prediction pair and calculating the intersection between the two sets divided by the length of each set.  
The idea is similar to the Mean Average Precision (mAP) used commonly in object detection tasks, for example to evaluate the accuracy in the COCO dataset, where a threshold on the Intersection Over Union (IOU) is used to evaluate the matches between ground truth and predicted bounding boxes.  
For example, if for the class _Claim_ in a document there are ground truth annotations $$[(1, 2, 3, 4, 5), (6, 7, 8), (21, 22, 23, 24, 25)]$$ and the model predicts $$[(1, 2), (6, 7, 8)]$$ then:  

* The first prediction would not have $$\geq 0.5$$ overlap with either ground truth and would be a false positive.
* The second prediction would overlap perfectly with the second ground truth and be a true positive.
* The third ground truth would be unmatched, and would be a false negative.



## Experimental setup

We trained our networks on a Tesla V100. The training takes about 30 min per epochs. We trained around 50 networks for tuning purposes, with an average of 2 epochs each, resulting in roughly 50 hours of training.     
We experimented multiple times varying the setup to understand the best way for training this model. For tackling the class imbalance problem, we implemented the focal loss technique. Then we tried also different initialization of weights and bias for the final layers and also for the encoder-decoder model when non-pretrained weights were used. We experimented also with fine-tuning of the features extractor backbone and with it frozen. As optimizer, we used \textit{AdamW} and we dropped the learning rates for the second epochs by a factor of ten.
We report the best hyperparameters that we found:


* Hidden dimension = 2048;
* Heads lr = $$10^{-4}$$;   
* Transf. lr = $$10^{-5}$$;   
* Number of queries = 40;     
* $$\lambda_{CE}$$ = 1;  
* $$\lambda_{L1}$$ = 1;      
* $$\lambda_{GIoU}$$ = 0.5; 
* $$\lambda_{Overlap}$$ = 0.5; 
* $$\gamma_{FL}$$ = 2.


where $$\lambda$$s are the coefficient of the diverse components of the loss as indicated by the subscript and $$\gamma_{FL}$$ is the modulating term of the cross entropy (focal loss).
For the best model, we also set the depth for the classification head and the bounding box head to 3 and 5 layers respectively. We also initialize the bias of the last classification layer in order to match the prior distribution of classes.  
Instead, we don't apply weight decay nor dropout, since our model doesn't overfit the training data. We didn't see any benefit in applying or not gradient clipping.  
The model has 235M parameters and was trained for 2 epochs since training for more time doesn't improve its performance.    
In order to exploit the information obtained from the data set analysis, we decided to define a new strategy for the global attention. For each class we detected the most frequent word and associated with them the token of the global attention, to impose the network a certain focus on particularly relevant words.

\begin{table}[ht]
\centering
\begin{tabular*}{.45\linewidth}{@{\extracolsep{\fill}}cc}
\hline
\textbf{Class} & \textbf{Most freq. word}\\
\hline
Claim & Reason \\
Concluding statement & Conclusion\\
Counterclaim & Argue\\
Evidence & Electors\\
\hline
\end{tabular*}
\quad
\begin{tabular*}{.45\linewidth}{@{\extracolsep{\fill}}cc}
\hline
\textbf{Class} & \textbf{Most freq. word}\\
\hline
Lead & Name\\
Position & Believe\\
Rebuttal & However\\
 & \\
\hline
\end{tabular*}
\caption{Each class with its correspondent most frequent word.}
\label{table:class_words}
\end{table}


As can be seen, in some cases we can notice a relation between the meaning of a word and its classes, while in other cases we have words whose presence is strictly dependent on our data set (Lead/Name, Evidence/Electors). However for sake of fairness, we selected one word for each class. 

## Results

As baseline we thought to use a simple detector which predict always the same class, 'Claim' which is the most frequent and detect an argumentative element every 151 words which is the median length. It can give a qualitative information on the hardness of the task and it is useful to show how we were able to improve this simplest classifier. For a standard classifier defining a simple baseline it's easy, it's not the same for an object detection problem, indeed the baseline we proposed isn't really informative.

\begin{table}[ht]
\centering
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Class} & \textbf{Precision}  & \textbf{Recall}  & \textbf{F1-score}\\
\hline
Lead                 &  0.000 & 0.000 & 0.000\\
Position             &  0.000 & 0.000 & 0.000\\
Evidence             &  0.000 & 0.000 & 0.000\\
Claim                &  0.001 & 0.001 & 0.001\\
Concluding Statement &  0.000 & 0.000 & 0.000\\
Counterclaim         &  0.000 & 0.000 & 0.000\\
Rebuttal             &  0.000 & 0.000 & 0.000\\
\hline
\textbf{Macro Avg}   &  0.0001 & 0.0002 & 0.0002\\
\hline
\end{tabular}
\caption{Scores of the baseline model.}
\end{table}

\begin{table}[ht]
\centering
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Class} & \textbf{Precision}  & \textbf{Recall}  & \textbf{F1-score}\\
\hline
Lead                 &  0.418 & 0.705 & 0.525\\
Position             &  0.219 & 0.222 & 0.221\\
Evidence             &  0.559 & 0.567 & 0.563\\
Claim                &  0.137 & 0.342 & 0.196\\
Concluding Statement &  0.213 & 0.840 & 0.340\\
Counterclaim         &  0.019 & 0.165 & 0.035\\
Rebuttal             &  0.019 & 0.036 & 0.025\\
\hline
\textbf{Macro Avg}   &  0.226 & 0.411 & 0.272\\
\hline
\end{tabular}
\caption{Scores of the best model without global attention on most frequent words for each class.}
\end{table}


\begin{table}[ht]
\centering
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Class} & \textbf{Precision}  & \textbf{Recall}  & \textbf{F1-score}\\
\hline
Lead                 &  0.438 & 0.715 & 0.543\\
Position             &  0.179 & 0.389 & 0.246\\
Evidence             &  0.371 & 0.661 & 0.476\\
Claim                &  0.119 & 0.256 & 0.163\\
Concluding Statement &  0.535 & 0.615 & 0.572\\
Counterclaim         &  0.015 & 0.068 & 0.025\\
Rebuttal             &  0.010 & 0.066 & 0.017\\
\hline
\textbf{Macro Avg}   &  0.238 & 0.396 & 0.292\\
\hline
\end{tabular}
\caption{Scores of the best model with global attention on most frequent words for each class.}
\end{table}

Adding the global attention to particular words present a general improvement $$(+6\%$$ in f1-score). However we can notice a drop in performances related to the recall while an increasing in the precision. This probably means that the network lose a bit in the capability of classification but improve its performances in the identification of the position of the argumentative units. This technique could be in general a good idea, however some of the words we used don't present any real correlation with their class. So it is highly probable that in a different scenario using this words could lead to a poor results. The way in which we selected them is too task dependant, another possibility would be to use a list of words provided by a team of experts in order to have a more general capability.

## Error analysis
One of the most common error and easiest to notice is the incapability of the model to understand which are the most common words or symbols that define the beginning and the end of a sentence, such as periods, commas or conjunction which usually are very helpful also for humans for understanding which kind of argumentative unit we can expect. A possible way to influence the model towards this direction, could be to use the global attention on all the conjunction. On one hand, we have that this approach would be very informative for the model since it would mimick also the way in which human annotators establish the groundtruth, on the other hand there would be the risk of increasing too much the computational complexity.
Another common error is the overlapping between different detection, in some cases it is related just to consecutive detection, in other cases instead it is due to multiple detection of the same unit, this problem is pretty common in object detection task. A possible way to overcome this problem, would be to adopt technique such as non-maxima suppression, which consist in discarding all the detection which overlap more than a threshold and keeping just the one with the greatest score. Clearly also this approach doesn't come for free, it increases the computational complexity and moreover it takes in account only the best score, without any consideration for the rest. Below you can see an example the perfectly summarize the errors described above.  
$$
{\color{magenta}
participate seagoing cowboys once. People might not have many reasons to support there statement about 'Why you [Claim]
}
$$
$$
{\color{blue}
participate seagoing cowboys once. People might not have many reasons to support there statement about 'Why you shouldn't be a Seagoing Cowboy', but I have at least some reasons as why you should join [Evidence]
}
$$

In the end, another big difference between the object detection task and ours, is related to the proportion foreground/background and Argumentative/Non-Argumentative units. In the first case is common to have small number of object with a lot of background while in pour case, working with argumentative essay, more or less all the proposition have to be classified.

## Discussion

Even before starting, we were conscious that trying to tackle such a complex task as Argumentative Mining using a completely different approach, in particular one derived from object detection, would have been an hard challenge. However we were curious to explore new path and propose a different point of view for this kind of problems. It gave us the opportunity to work and understand not only the way in which transformers work, but also how they can be optimized for handling long texts, which by the way can be very useful for argumentative essays.
This problem was very stimulating and challenging. Even if it's not new as described in section \ref{background}, the fact that doesn't exist a common solution which is able to perform well in any situation, urge us to analyze in deep the data set to find different way to improve our initial solution. Of course we wouldn't expect to tie the results of the top teams of the competition, we were more interested in proposing something new, or at least not so spread as more standard approaches. We hope it could offer a new perspective to Argumentative task and maybe to find a direction to merge CV and NLP with model capable of both tasks at the same time.
