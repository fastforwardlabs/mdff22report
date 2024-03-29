## Methods for inferring concept drift

In comparison to recent research on supervised drift detectors, much less attention has been paid to unsupervised methods.^[[An overview of unsupervised drift detection methods](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1381)] However, detecting shifts in data distributions is a well-explored field of data mining, with solutions ranging from multiple hypothesis testing and novelty detection to discriminative distance and algorithm-specific techniques. In our exploration, we focused on methods that are model-agnostic and truly unsupervised, to ensure broad applicability in practice. In this section, we present four methods for inferring concept drift without labels, and use a binary classification task to frame the discussion.

### 1. Statistical test for change in feature space

The ultimate goal of feature drift detection is to determine if two distributions are different. Therefore, the first and most basic approach to infer concept drift applies a hypothesis test to flag if a statistically significant change has occurred between the reference and detection windows for each feature in a given data stream.

For continuous features, we use a two-sample [Kolmogorov-Smirnov (KS)](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) test, which is a non-parametric hypothesis test used to check whether two samples originate from the same distribution. For categorical features, we make use of a [Chi-Squared](https://en.wikipedia.org/wiki/Chi-squared_test) goodness of fit test.

![Figure 8: Hypothesis tests are performed feature-wise when dealing with multivariate tabular data. Correction is applied to the tests to arrive at overall determination of significance.](figures/FF22-10.png)

In the case of multivariate tabular data, we can test each feature independently (while accounting for multiple tests) to arrive at an overall signal of drift or no-drift, as seen in Figure 8. The [Bonferroni test](https://en.wikipedia.org/wiki/Bonferroni_correction) is a notable approach for correcting multiple hypothesis tests - while making conservative assumptions about the (in)dependence among them - to arrive at a final result.

In contrast to this feature-wise approach, we could also apply a multivariate two-sample hypothesis test like the kernel-based technique called [Maximum Mean Discrepancy (MMD)](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf). MMD allows us to distinguish between two probability distributions, based on the mean embeddings of those distributions. While this method does side-step the need for multiple tests, the choice of kernel is critical to ensuring its correctness, and a linear time complexity imposes a potential bottleneck in streaming applications.^[[Optimal kernel choice for large-scale two-sample tests](http://www.stat.cmu.edu/~siva/Papers/MMD12.pdf)]

::: info
**_High Dimensional Data_**

When it comes to high dimensional data, (e.g., images, text) best practices for detecting drift with two sample tests are an area of active research. Recent work proposes combining dimensionality reduction techniques (e.g., PCA, randomly initialized auto-encoders) with subsequent two-sample testing.^[[Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)] The overall idea is that these dimensionality reduction techniques yield either a uni- or multi-dimensional representation of the data. We can then choose a suitable statistical test to apply to the reduced data stream to detect drift.
:::

While feature-wise and multivariate hypothesis testing is broadly applicable across machine learning use cases, it has several limitations as a real concept drift inference tool. Because these methods consider drift in each feature to be equally important (despite p-value correction across tests), they are prone to false positive detections. Imagine the case where several features in a datastream exhibit drift, but none of them are of high importance to a classifier’s decision-making process. It's likely that the present drift will not actually impact the learned decision boundary, despite the hypothesis test’s ringing alarm.

This limitation arises because we’ve excluded the classifier from the detection process and are making decisions solely on the distribution characteristics of incoming features - resulting in increased sensitivity to change and a high number of false alarms.

### 2. Statistical test for change in response variable

Unlike the previous method, where only the feature space is analyzed, our second approach infers concept drift by tacitly involving the classifier in the detection process, making change detection relevant to the prediction task at hand. To do so, we apply a model that's been trained on the reference window to generate predicted class probabilities (a response distribution) for observations in the detection window. Then, we use a _k_-fold procedure to obtain probability estimates for the reference window.

::: info
**_K-fold Procedure_**

In the _k_-fold procedure, the entire dataset is sequentially divided into _k_ bands of samples. In the first iteration, the first _k-1_ bands serve as the training set, to learn a model that is used to generate predictions over the _kth_ band of observations. This process is repeated _k_ times where each band functions as the test set exactly once, yielding a response distribution for the entire reference window.^[[On the Reliable Detection of Concept Drift from Streaming Unlabeled Data](https://arxiv.org/pdf/1704.00023.pdf)]
:::

With our two populations in hand, we can apply a Kolmogorov-Smirnov hypothesis test to see if the response distributions between reference and detection windows differ significantly. In effect, the trained model serves as a dimensionality-reducing preprocessing step. It leverages its learned relationship between features and targets (i.e., \\(P(y|X)\\)) to generate a response distribution that is sensitive to feature space changes that will _likely_ affect the performance of the model in question. If important features in the detection window have drifted from those the model learned on, we would expect the classifier to produce significantly different response distributions, as depicted in Figure 9.

![Figure 9: Example response distributions between reference and detection windows for a binary classification task. The plot on the left shows nearly identical distributions resulting from a case where feature drift is not present, while the plot on the right depicts divergent distributions.](figures/FF22-08.png)

Although it's a step in the right direction, this method is still overly sensitive. That’s because, by design, a KS test is responsive to changes across the _entire_ response distribution. But do we really care about changes in regions of high confidence? For example, if the density shape between 0 and 0.25 confidence level changes a bit, it doesn’t impact the classification outcome of those points, because they’re still well below the 0.5 decision threshold. This leads us to the next approach.

### 3. Statistical test for change in margin density of response variable

Rather than test for changes across the entire cumulative response distribution, we can instead focus on just the regions of uncertainty around our decision threshold, where slight variations in confidence lead to different classification outcomes.

To do so, we must introduce a parameter that specifies a desired _margin width_ around the decision boundary, to define a region of uncertainty. Margin here is the portion of the prediction space which is most vulnerable to misclassification.^[[On the Reliable Detection of Concept Drift from Streaming Unlabeled Data](https://arxiv.org/pdf/1704.00023.pdf)] Then, for both windows, we classify each observation as _in-margin_ or _out-of-margin_, based on its predicted confidence score. We compare these categorical populations between windows, using a Chi-square goodness of fit test to check for significant changes in the margin density. The underlying assumption here is that a significant change in the number of samples in the margin is indicative of a drifting concept.

![Figure 10: Response distributions that diverge only at tail ends do not impact classification results (left), whereas changes of distribution within the margin do (right). The decision boundary here corresponds to a confidence of 0.5.](figures/FF22-09.png)

The impact of this approach is highlighted in Figure 10, above. On the left, we see the case where a divergence exists towards the tail ends of the distribution, but the rest of the probability space remains congruent. This example would fail the KS test (described in Method 2), signaling a feature drift, and consequently request costly new labels for retraining. However, because the divergence exists far from the decision boundary, it would _likely_ not have impacted the classification results, making it a false positive detection. In contrast, the margin density approach would tolerate this inconsequential change. Only when a statistically significant divergence occurs _inside_ the margin will Method 3 raise an alarm, as shown in Figure 10 (on the right).

Introducing a margin of uncertainty to desensitize feature drift detection does help reduce the number of false positive detections. However, there is still room for improvement. Each method we have discussed so far relies on hypothesis testing to signal drift. Unfortunately, the mere falsity of a null hypothesis doesn’t say much about our window samples, other than that they don’t come from an _identical_ population. But do we really care if the populations are identical?

If our goal is to reduce the sensitivity of feature drift detections, we probably care more about quantifying how different two populations are, which is something that a statistical test cannot provide. A quantitative measure of similarity affords us the flexibility to set our own threshold, depending on our tolerance for error. In essence, we need a way to distinguish what level of change is _statistically significant_ from what is _practically significant_.

### 4. Detect change in margin density of response distribution using a learned threshold

Our final method uses a learned threshold to detect change in the margin density of a response distribution. Building upon the previous two methods, we first obtain a response distribution for each window, and introduce a margin to classify predictions as in or out of the region of uncertainty. However, rather than applying a Chi-square test (as in Method 3), we establish an expected value for margin density based on the reference window.

This is accomplished during the _k_-fold procedure by calculating the percentage of instances falling in margin, relative to the total instances in the window (i.e., margin density) for each fold. The cross-validation procedure produces a population of \\(k\\) margin density values from which we can calculate a mean (\\(MD_{ref}\\)) and standard deviation (\\(\sigma_{ref}\\)), providing a strong estimate of the expected margin density value and an acceptable deviation.

These values are then used to signal change in the detection window based on a desired level of sensitivity, \\(S\\). A practically significant change is signaled when the margin density of the detection window differs by more than \\(S\\) standard deviations from the expected margin density value, as seen in the equation below.

$$ |MD_{det} - MD_{ref}| > S \times \sigma\_{ref} $$

By setting the expected margin density value from the population observed in the reference window, we establish a baseline specific to the problem at hand. Adding the sensitivity parameter offers control over the detection process. Larger values of \\(S\\) will reduce the number of false positive detections, but possibly at the cost of increasing false negatives - a decision that might make sense when the cost/impact of a false negative is low. Inversely, lowering \\(S\\) might be a good idea for critical applications, where the cost of a real drift could be harmful if undetected.^[[On the Reliable Detection of Concept Drift from Streaming Unlabeled Data](https://arxiv.org/pdf/1704.00023.pdf)]
