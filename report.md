# ECE 276A Project 1 Report

## Introduction

Object detection has tremendously wide application in computer vision, robotics, security, and many other fields. It's important for computer algorithm to detect and localize objects in images and videos. In this project we will present gaussian discriminatory analysis based approach to accurately classify colors and detect blue recycle bins in images. The joint distribution between colors and labels can be modeled using multivariable gaussians. Hence allowing us to compute the probability of a given data sample belong to a perticular class.

## Problem Formulation

### Color Classification

Let $f(\textbf{x};\theta): \mathbb{R^3}  \rightarrow (1,2,3)$  that maps a pixel to its color label ($\text{red}=1,\text{green}=2,\text{blue}=3$).

_**Problem:**_ Find the best parameter $\theta$ such that $\theta^{*} = \underset{\theta}{\operatorname{argmin}} \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\{f(\mathbf{x}_{i})\neq y_i\}$, which in term minimizing the number of incorrect predictions.

### Bin Detection

Let $f(\textbf{X};\theta): \mathbb{R^{m \times n}}  \rightarrow \mathbb{R^{2 \times 2}}$ that takes an input image and produces bounding box for blue recycling bins.

_**Problem:**_  Find the best parameter $\theta$ such that $\theta^{*} = \underset{\theta}{\operatorname{argmax}} \frac{1}{n}\sum_{i=1}^{n} IoU(f(\textbf{X}_i),\mathbf{Y}_i)$, where IoU is the intersection over union between the proposed bounding box and the goundtruth bounding box.

## Technical Approach

### Color Classification

#### Modeling Distributions

The joint distribution between pixel and its color and be factored using bayes rule, $p(x,y)=p(x|y)p(y).$ The class conditional probability $p(x|y)$ can be modeled using multivariable gaussian s.t. $p(x|y)\sim \mathcal{N}(x,\mu,\Sigma)$, and $p(y)$ can be modeled by a single parameter $\theta$ that denotes the probability of $y$. 

#### Parameter Estimation

The complete distribution of i.i.d. samples $\mathcal{D}\{x_{i},y_{i}\}_{i=1}^{N}$ can be expressed as:
$$
\begin{align}
p(\mathbf{X},\mathbf{y}) &= \prod_{i=1}^{N}p(x_i|y_i)p(y_i) \\
                                     &= \prod_{i=1}^{N} \prod_{k=1}^{K} \left\{\mathcal{N}(x_i,\mu_{k},\Sigma_{k}) \theta_k \right\}^{\mathbf{1}\{y_i = k\}}                         
\end{align}
$$

where $x_i \in \R^{3}, y_i \in \{1,2,3\dots K\}$

$\mathbf{X} = x_1, x_2,\dots,x_N$ and $\mathbf{y} = y_1,y_2,\dots, y_N$.

Taking the log of joint distribution, we have:
$$
\log p(\mathbf{X},\mathbf{y}) =\sum_{i=1}^{N}\sum_{k=1}^{K} \mathbf{1}\{y_i=k\}(\log \theta_k+ \log \mathcal{N}(x_i,\mu_{k},\Sigma_k)) \\

\begin{align*} 
\theta_k^{*} &=  \underset{\boldsymbol \theta}{\operatorname{argmax}} \log p(\mathbf{X},\mathbf{y})\text{, subject to} \sum_{k=1}^{K} \theta_{k} = 1 \\

\mu_k^{*} &=  \underset{\boldsymbol \mu}{\operatorname{argmax}} \log p(\mathbf{X},\mathbf{y})\\
\Sigma_k^{*} &=  \underset{\boldsymbol \Sigma}{\operatorname{argmax}} \log p(\mathbf{X},\mathbf{y})
\end{align*}
$$
All of the above three optimization problems can be solved by taking the gradient of $\log p(\mathbf{X},\mathbf{y})$ with respect to the variable we are trying to optimize and set the gradient to zero. We can then obtain the following solutions:
$$
\begin{align*} 
\theta_k^{*} &= \frac{1}{N}\sum_{i}^{N} \mathbf{1}\{y_i=k\} \\

\mu_k^{*} &=  \frac{\sum_{i=1}^{N}\mathbf{1}\{y_i =k\}x_i}{\sum_{i=1}^{N}\mathbf{1}\{y_i =k\}}\\
\Sigma_k^{*} &=  \frac{\sum_{i=1}^{N}(x_i - \mu_k)(x_i - \mu_k)^\top\mathbf{1}\{y_i =k\}}{\sum_{i=1}^{N}\mathbf{1}\{y_i =k\}}
\end{align*}
$$

#### Inference

After we have obtained parameters, we can start to use our model to make predictions. For a given pixel sample $x$, we first compute the joint distribution against all classes, i.e. $p(x,y=1)$, $p(x,y=2)$, $p(x,y=3)$. The prediction $\hat{y} = \underset{\boldsymbol c}{\operatorname{argmax}} p(x,y=c)$.

### Bin Detection

Here we employee a two fold approach: first segment the image with only pixels of the same color as blue recycling bins, then draw bounding box on pixel regions that are mostly to be blue recycling bins.

#### Mask Segmentation

Based on the fact that blue recycling bin has a distinctive color, we can build a color classifier using approach outlined in part 1. The objective is simple: is the given pixel recycling bin blue or not. 

##### Training Data

First we label the recycling bin region in the image, this will be the training data for recycling bin class. The rest of the image will be training data for the none-recycle bin class. Additionally, sky blue are also sampled to create sky blue class to increase performance and avoid confusion between sky and recycling bin.

##### Color Space

Lighting conditions can affect RGB values dramatically. Therefore picking a different colorspace can make the classifier more robust and accurate in various conditions. Here we experiment with following colorspaces `["HSV","HLS","LAB","RGB","YUV"]`. We then measure the distance between $\mu_{\text{bin}}$ and $\mu_{\text{not_bin}}$. The distance between the mean vector of each class indicates how far they are apart under that particular colorspace. When mean vectors are far apart, classifier is less likely to mislabel one class for another. 

|   Colorspace   | $\|\mu_{\text{bin}} - \mu_{\text{not bin}}\|$   |
| ---- | ---- |
|   HSV   |  79.23    |
|   HLS   |   64.43   |
|   LAB   |   35.76   |
|   RGB   |   52.50   |
|   YUV   |   32.11   |

From this table, we can clearly see HSV is the best colorspace to use.

#### Image Denoising 




