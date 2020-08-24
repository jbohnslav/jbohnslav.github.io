---
layout: post
title:  "A brief introduction to active learning"
date:   2020-08-21 16:48:00
categories: active learning, deep learning, computer vision
---

# Introduction
Supervised machine learning requires two fundamental components: data and labels. The goal of supervised learning is to learn some function $f$ that approximates the true labels given your input data. If you're working with existing datasets, like [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) or ImageNet, you accept all data and labels as given. However, if for your research project or startup you actually have to collect data, you have to think very carefully about how to most efficiently use your resources to assign labels to data. *Active learning* is the subfield of machine learning that tries to identify the most useful unlabeled data points to label.

# Overall approach and definitions
Let's assume we have the following situation:

* A large dataset $ \mathbf{X} = \\{ \mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2... \mathbf{x}_N  \\}$.
* Two subsets of $\mathbf{X}$: the unlabeled set $U$ and the labeled set $\mathcal{L}$. *The size of $U > > \mathcal{L}$, otherwise we wouldn't need active learning*. The labels $\mathcal{Y}$ can be anything, from sentiments in NLP to pixel-wise object labels in the case of semantic segmentation.
* Some machine learning model that learns with pairs of $(\mathbf{x}_i, \mathbf{y}_i)$
* An *oracle*, which is some entity that generates the true labels $\mathbf{y}_i$ for each given $\mathbf{x}_i$. In practice, this is a cool name for human annotators.

The goal of active learning is to intelligently pick which unlabeled examples in $U$ we submit to our "oracle" for labeling.

### Active learning scenarios
There are a few main types of active learning frameworks. The first, **membership query synthesis**, is a scenario in which the model itself generates unlabeled examples. It doesn't make a lot of sense in the image classification domain for a model to generate images--this is a totally different, difficult task. To read about scenarios in which MQS might be useful, see Settles^[@settlesActiveLearningLiterature2009].

The next active learning framework is **stream-based**, in which your model is presented with an unlabeled example one at a time. The model will decide for each one whether or not to submit it for labeling. Stream-based active learning might be useful if you're already processing data one at a time. If you were a large image-sharing website working on models to automatically detect prohibited content, you might want to submit any image with a reasonable probability of containing prohibited content to human labeling. The downside of stream-based active learning is immediately obvious--it doesn't particularly matter if the stream-based strategy is optimal if it drastically exceeds your labeling budget.

The framework we'll focus on is **pool-based** active learning, in which you evaluate your model on your whole unlabeled dataset, pick a labeling budget, and label the top-K most "useful" unlabeled examples.

With this setup, we will iterate through the *active learning loop*:
1. Train your model(s) on your labeled training set $\mathcal{L}$.
2. Run your model on your unlabeled set $U$.
3. Use your model(s) to *rank* unlabeled examples from most useful to least useful.
4. Send the top K most useful unlabeled examples to the oracle.




## Should you use active learning?
Before we get started describing the overall framework of active learning, and its different variants, let's discuss whether or not you *should* use active learning at all.

{% include image.html url="/assets/posts/active_learning/scientists_should.gif" description="Figure 1: Listen to Jeff Goldblum. Consider whether or not you *should* first!" %}

<figure>
  <p align="center">
    <img src="{{site.url}}/assets/posts/active_learning/scientists_should.gif" alt="Should you actually implement active learning,">
    <figcaption align="center">Figure 1: Listen to Jeff Goldblum. Consider whether or not you "should" first!</figcaption>
  </p>
</figure>

As we saw above, the baseline against which we compare active learning algorithms is random sampling from our unlabeled data $U$. For active learning to be worthwhile, the *total cost of implementing an active learning pipeline must not exceed randomly sampling more data*. To see why this is, let's do a little math.

Let's say you're a junior data scientist at US-based company. To keep the math easy, let's say you make \\$100,000 per year and work 50 weeks per year. To implement your active learning pipeline, you did a day or two of reading and background research; a few days of coding; a few days of experiments to find the best methods and hyperparameters for your company's data; and a day or two to write up your results and present to your colleagues. This is two weeks of (very productive) work: at your salary, this cost your company \\$4,000. According to the Minimum Cost Active Labeling paper^[@qiuMinimumCostActive2020], image classification labels cost about $0.04 cents per label. That means your salary alone for the duration of this project cost the company as much as *randomly labeling 100,000 new datapoints!* This doesn't even include the cost of model training or evaluation on your unlabeled pool.

Therefore, to beat our null strategy of *random sample even more data*, one of the following scenarios must apply:
* Your labels are expensive. If you need 21 board-certified dermatologists^[@estevaDermatologistlevelClassificationSkin2017] to label your training examples, you will spend a lot more than 4 cents per image.
* You are working *at scale*. Lots of applications will probably be just fine with hundreds of thousands of randomly picked training examples. However, if you're someone like [Tesla AI](https://www.youtube.com/watch?v=hx7BXih7zx8), randomly annotating your 10 million-th stop sign will probably not improve your model much. You need some active learning strategy to find the difficult, edge cases: stop signs occluded by a tree branch, for example.
<figure>
  <img align="center" src="{{site.url}}/assets/posts/active_learning/scaledml.png" alt="Screenshot from Karpathy talk">
  <figcaption align="center">Figure N: Occluded stop sign from TeslaAI</figcaption>
</figure>


Sections
* Problem statement
* Definition of terms
* Pool vs streaming
* Acquisition functions
* Discussion of focus on classification problems
* Drawbacks
  * Engineering + thought time when you should’ve just randomly labeled more



Points to hit
* Useful when dataset is large, most samples are easy, imbalanced classes, labeling is expensive
  * Expensive: human doctors

Questions
* Why are models re-trained from scratch?
  *E.g. https://openaccess.thecvf.com/content_cvpr_2018/html/Beluch_The_Power_of_CVPR_2018_paper.html

References
* Blogs
  * https://jacobgil.github.io/deeplearning/activelearning
  * https://blog.scaleway.com/2020/active-learning-some-datapoints-are-more-equal-than-others/

```python
def testing(args, kwargs):
    return args, kwargs

```

$$ mean = \frac{1}{N}\sum_{i=1}^{n} x_{i} $$
