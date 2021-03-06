## Framing the problem

With these definitions in mind, we see that real concept drift in a data stream (Source 2) poses the main concern for production models, since it directly impacts model performance. The most effective solution to address this issue involves detecting when the learned relationship between features and targets is no longer appropriate for incoming data, and then training a new model to learn the novel concept. An adaptive workflow like this is shared among common supervised methods like Drift Detection Method (DDM), Early Drift Detection Method (EDDM), and ADaptive WINdowing (ADWIN). We describe this workflow in Figure 6, below.

![Figure 6: General workflow of supervised drift detection methods that use significant changes in performance metrics to signal concept drift.](figures/FF22-06.png)

In general, these techniques monitor a task-dependent performance metric like accuracy, F-score, or precision/recall. If the metric of interest deviates from an acceptable level (as determined during training evaluation on the reference window), a drift is signaled. 

![Figure 7: Impact of supervised concept drift detection on machine learning system performance over time.](figures/FF22-07.png)

The cumulative effect of this approach over the lifetime of a machine learning system is highlighted in Figure 7. Initially, the system celebrates strong performance because the model has learned from recent data. After some time, accuracy declines as concepts evolve, until ultimately a metric threshold is crossed, and drift is detected. System performance then realizes an immediate boost after retraining, as the new concept is absorbed.

Despite the ample research and proven effectiveness of these supervised methods, they all suffer from a shared, impractical assumption － that true labels are instantaneously available after inference. In most use cases, the immediate availability of true labels is infeasible for several reasons.

First, annotating data is expensive, in both cost and labor, as it often requires hired domain expertise. The issue is described succinctly by the authors of [On the Reliable Detection of Concept Drift from Streaming Unlabeled Data](https://arxiv.org/pdf/1704.00023.pdf):

> *"To highlight the problem of label dependence, consider the task of detecting hate speech from live tweets, using a classification system facing the Twitter stream (estimated at 500M daily tweets). If 0.5% of the tweets are requested to be labeled, using crowdsourcing websites such as Amazon’s Mechanical Turk2, this would imply a daily expenditure of $50K (each worker paid $1 for 50 tweets), and a continuous availability of 350 crowdsourced workers (assuming each can label 10 tweets per minute, and work for 12 hours/day), every single day, for this particular task alone. The scale and velocity of modern day data applications makes such dependence on labeled data a practical and economic limitation."*

Second, in addition to label scarcity, _verification latency_ - or the period between the availability of an unlabeled test instance and the availability of its true label - is application-dependent and often variable.^[[Fast Unsupervised Online Drift Detection Using Incremental Kolmogorov-Smirnov Test](https://www.kdd.org/kdd2016/papers/files/rpp0427-dos-reisA.pdf)] For example, it can take several months for an act of credit card fraud to be reported (i.e., ground truth) from the time the fraudulent transaction occurred. If F1 score is the only metric being used to track model performance (and thus detect concept drift), there may be several months of higher than normal fraudulent activity without any signal that something is wrong.

Finally, some use cases operate in an extreme case of _infinite verification latency_, where ground truth labels are impossible to ever obtain. Consider a bank which uses machine learning to power its lending decisions. If a model predicts a loan will default for a given applicant, the loan is never granted; therefore, it can never be determined if the loan would have actually defaulted or been repaid. Use cases like this demand an alternative solution.
