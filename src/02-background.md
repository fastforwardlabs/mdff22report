## Background

### What is concept drift?

Most machine learning systems today operate in a batch paradigm where they probe a historical data set to develop a model that reflects the world as it was at the time of training. But as we’ve seen, the world is always changing and the complex relationships that a model abstracts are also likely to change over time causing model performance to deteriorate if not accounted for. This phenomenon in which the statistical properties of a target domain change over time is considered concept drift.^[[Learning under Concept Drift: A Review](https://arxiv.org/pdf/2004.05785.pdf)]

Formally, concept drift between time \\(t\\) and \\(t+1\\) can be defined as:

$$P_{t}(X,y) \not= P_{t+1}(X,y)$$

where \\(P_t\\) denotes the joint probability distribution at time \\(t\\) between the set of input variables \\(X\\) and the target variable \\(y\\). Since the joint probability can be decomposed as the product of the probability of \\(X\\) and the conditional probability of \\(y\\) given \\(X\\) changes in a data stream can therefore be characterized by changes in the components of this relationship according to the equation below.

$$P_t(X,y) = P_t(X) \times P_t(y|X)$$

This decomposition yields two underlying sources of drift.

##### Source 1: Feature Drift

Feature drift (also referred to as covariate shift, feature change, input drift) characterizes the scenario where the distribution of one or more input variables change over time (i.e. \\(P(X)\\) changes).
![Figure 3: Forms of feature drift. The classification boundary depicted at time \\(t+1\\) represents the _previously learned relationship_ between features and targets at time \\(t\\). Colors represent ground truth classes of the data points at the specified time step.](figures/FF22-03.png)
This is seen in both Figure 3a & b above, where the distribution of features has changed from time t. In Figure 3.a, the feature drift has occurred in a region that directly affects the outcome of the learned classification boundary, causing model performance to decrease (and thus making it classified as both Source 1 and Source 2 drift). However, feature drift can also occur where \\(P(X)\\) changes over time, but the changes do not affect the learned decision boundary. This describes a specific type of feature drift called _virtual drift_, as seen in Figure 3.b. This is an important distinction because as we see here, only the changes in \\(P(X)\\) _that affect the prediction decision_ actually warrant a model adaptation.

For example, consider a clothing brand that is looking to recommend items for a given customer as relevant or not relevant. Suppose this customer lives in a tropical climate, so lightweight, breathable clothing items are relevant to them while heavy, cold weather apparel is not. In this scenario, the independent features \\(X\\) are both the customer's preferences (e.g. age, size, location) and the brand’s product line. The dependent variable, \\(y\\), is the relevance of a clothing item to the customer. 

If the brand’s lead designer quits and is replaced, the brand's design style will naturally change as a result (i.e. change in \\(P(X)\\)). However, warm weather clothing still remains relevant for this customer despite the stylistic differences (i.e. no change in \\(P(y|X)\\)). This scenario corresponds to a virtual drift.^[[A Survey on Concept Drift Adaptation](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/82cb2dbe-86a2-43d0-8ac2-fb2892295b48/A_Survey_on_Concept_Drift_Adaptation_2014.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210804%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210804T123525Z&X-Amz-Expires=86400&X-Amz-Signature=0e75f52b14a5448db13ddefc0c1825b9df11a4fab6577f10d632ae64bf317373&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22A%2520Survey%2520on%2520Concept%2520Drift%2520Adaptation%25202014.pdf%22)]

Suppose now that due to a shift in brand strategy, the company alters their product focus to sell mostly cold weather gear and less warm weather items, but the designer (and style) stay the same. This scenario also corresponds to a feature drift (Source 1), however it’s one that does impact the decision boundary (i.e. \\(P(y|X)\\)). Therefore this scenario is also categorized as Source 2 drift.

##### Source 2: Real Concept Drift

The second source of drift, called real concept drift (also commonly referred to as actual drift, concept shift, conditional change), refers to changes in \\(P(y|X)\\) and signals that a previously learned relationship between features and targets no longer holds true. Unlike feature drift, this type of drift will _always_ cause a drop in model performance, like we noticed in the previous example.
![Figure 4: Forms of real concept drift. The classification boundary depicted at time \\(t+1\\) represents the _newly learned relationship_ between features and targets at time \\(t+1\\). Colors represent ground truth classes of the data points at the specified time step.](figures/FF22-04.png)
It’s important to note that real concept drift can happen either with or without a change in \\(P(X)\\).^[[A Survey on Concept Drift Adaptation](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/82cb2dbe-86a2-43d0-8ac2-fb2892295b48/A_Survey_on_Concept_Drift_Adaptation_2014.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210804%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210804T123525Z&X-Amz-Expires=86400&X-Amz-Signature=0e75f52b14a5448db13ddefc0c1825b9df11a4fab6577f10d632ae64bf317373&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22A%2520Survey%2520on%2520Concept%2520Drift%2520Adaptation%25202014.pdf%22)] This nuance is shown in Figure 4.a where both the input feature distributions and the learned decision boundary have changed in the new time step. In contrast, Figure 4.b demonstrates the scenario where input distributions remain constant, while the ground truth class labels have actually evolved.

Continuing our example, now suppose that the customer moves from their tropical paradise to the Alaskan tundra, while the clothing brand makes no changes to their offerings or staff. In this case, the very meaning of “relevance” flips, making cold weather gear relevant and warm weather clothing irrelevant. This describes another example of real concept drift, but with no change in \\(P(X)\\).

However, the real world is rarely ever this clean cut, and oftentimes both sources of drift are at play simultaneously. Let’s now imagine the case where the customer moves to a temperate climate with cold nights and warm days, and the company slightly alters their product mix towards cold weather gear. Here, we observe changes in both \\(P(X)\\) and \\(P(y|X)\\) making it difficult to attribute concept drift to any single source.

::: info
***Additional Classifications***

Both feature drift and real drift can be further classified based on the rate at which the concept evolves. For instance, the drift could occur abruptly resulting in a quick change in the distribution, think of drifts induced by a sensor or an equipment failure. Such cases are considered _sudden_ concept drift. There are other instances where the drift occurs slowly over time, like the drift induced by rising temperatures in the atmosphere. These are deemed _gradual_ concept drift. In addition, there are also _recurring_ concept drifts, which are patterns or trends that tend to repeat themselves at intervals and are commonly found in seasonal data.
:::

### What is a data stream?

Before moving on, let’s take a step back and define some terminology that will be mentioned throughout this report. To discuss the idea of concept drift in production, we must consider dynamic data environments. For this reason, we reference concept drift detection and adaptation with regard to _data streams_ where instances arrive continuously and sequentially over time. Often streaming data is generated on the fly, potentially at a fast and variable rate, and with infinite range, making it a prime candidate for evolving data distributions.

Despite this, concept drift is not exclusive to data streams. And in order to frame a discussion involving both stream and batch contexts, concept drift detection methods commonly employ the notion of _sliding windows_, or groups of sequentially ordered observations. 
![Figure 5: Data streams are decomposed into windows of observations to establish context upon which concept drift occurs.](figures/FF22-05.png)
In general, one window contains the instances belonging to the most recent known concept which were used to train or update the deployed model, and one window contains instances which may have suffered a concept drift. We refer to these windows as the _reference window_ and _detection window_, respectively.^[[An overview of unsupervised drift detection methods](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1381)]