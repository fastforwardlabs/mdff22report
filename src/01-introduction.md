## Introduction

After iterations of development and testing, deploying a well-fit machine learning model often feels like the final hurdle for an eager data science team. In practice however, a trained model is never final, and this milestone marks just the beginning of the perpetual maintenance race that is production machine learning. This is because most machine learning models are static, but the world we live in is dynamic. More specifically, the ability of a trained model to generalize relies on an important assumption of stationarity - meaning the data upon which a model is trained and tested are *independent and identically distributed (i.i.d)*. In real-world environments, this assumption is often violated as human behavior and consequently the systems we aim to model are dynamically changing all time.^[[On the Reliable Detection of Concept Drift from Streaming Unlabeled Data](https://arxiv.org/pdf/1704.00023.pdf)]

![Figure 1: Examples of machine learning tasks where the effects of concept drift are prominent](figures/FF22-01.png)

Take for instance the impact of the recent COVID-19 pandemic on algorithm driven businesses like inventory management. Instacart’s model for forecasting in-store product availability [dropped from 93% to 61% accuracy](https://fortune.com/2020/06/09/instacart-coronavirus-artificial-intelligence/) due to the drastic change in shopping behavior as consumers stockpiled what previously were infrequently purchased goods. The model was forced to adapt to this transitory shift in it’s prior understanding of the world.

Not all changes are this sudden though. Consider the task of maintaining an email spam filtering service. The core technology consists of a text classification model that picks up on keywords in email content to block spammers. Over time, users will begin to manually report more messages as spam that are not caught by the filter. In this adversarial environment, spammers are continuously adjusting terminology to outwit the deployed spam filters so models must relearn what language constitutes the evolving concept of spam to remain effective.

Or we can look at the job of forecasting energy consumption where historical demand is just one piece of the puzzle. In practice, future demand is driven by a slew of non-stationary forces like climate fluctuations, population growth, or disruptive clean energy tech that necessitate both gradual and sudden domain adaptation.

::: info
***Domain Adaptation***

Domain adaptation (a subcategory of transfer learning) is the ability to apply an algorithm trained in one or more “source domains” to a different, but related “target domain”. In domain adaptation, the source and target domains share the same feature space, but different distributions.^[[Domain adaptation](https://en.wikipedia.org/wiki/Domain_adaptation)]
:::

Changes in environmental conditions like these are referred to as _concept drift_ and will cause the predictive performance of a model to degrade over time, making it obsolete for the task it was initially intended to solve.

![Figure 2: Production model performance will decay over time without adapting to drifting concepts](figures/FF22-02.png)

To combat this divergence between static models and dynamic environments, teams often adopt an adaptive learning strategy that is triggered by the detection of a drifting concept. Supervised drift detection is generally achieved by monitoring a performance metric of interest (such as accuracy) and alerting a retraining pipeline when the metric falls below some designated threshold.

While this strategy proves to be effective, there are several limitations that often prevent its use in practice. Namely, it requires immediate access to an abundance of labels at inference time to quantify a change in system performance - a requirement that may be cost prohibitive or outright impossible in many real-world machine learning applications.

In this report, we explore broadly applicable approaches for dealing with concept drift when labeled data is _not_ readily accessible. We’ll start by defining what we mean by concept drift and frame the limitations of supervised methods for detecting it. Then, we’ll discuss why true unsupervised concept drift detection is not possible, and explore several alternative methods for dealing with it. Finally, we’ll share our experimental results supporting the proposed methods, and wrap up with a discussion of considerations and limitations.