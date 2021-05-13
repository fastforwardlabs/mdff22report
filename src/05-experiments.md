## Experiments

Let’s put this into practice, and see how to specifically use word2vec for next event prediction (NEP) in order to generate product recommendations. In keeping with our e-commerce example above (Rhonda’s online shopping), we’ve chosen an open source e-commerce dataset that lends itself well to this task. The discussion below details our strategy, experiments, and results (the code for which can be found on [our GitHub repo](https://github.com/fastforwardlabs/recommendations)).

### Data
We chose an open domain e-commerce dataset^[[https://www.kaggle.com/vijayuv/onlineretail](https://www.kaggle.com/vijayuv/onlineretail)] from a UK-based online boutique selling specialty gifts. This dataset was collected between 12/01/2010 and 12/09/2011 and contains purchase histories for 4,372 customers and 3,684 unique products. These purchase histories record transactions for each customer and detail the items that were purchased in each transaction. This is a bit different from a browsing history, as it does not contain the order of items clicked while perusing the website; it only includes the items that were eventually purchased in each transaction. However, the transactions are ordered in time, so we can treat a customer’s full transaction history as a session. Instead of predicting recommendations for what a customer might click on next, we’ll be predicting recommendations for what that customer might actually *buy* next. Session definitions are flexible, and care must be taken in order to properly interpret the results (more on this in [Overall Considerations](#overall-considerations).  

In this case, we define a session as a customer’s full purchase history (all items purchased in each transaction) over the life of the dataset. Below, we show a boxplot of the session lengths (how many items were purchased by each customer). The median customer purchased 44 products over the course of the dataset, while the average customer purchased 96 products. 

![Figure 10: Session length in the Online Retail Data Set](figures/session_lengths.png)

Another thing to note is the popularity of individual products. Below, we show the log counts of how often each product was purchased. Most products are not very popular and are only purchased a handful of times. On the other hand, a few products are wildly popular and purchased thousands of times.

![Figure 11: Log counts of each product in the Online Retail Data Set](figures/product_counts_arrow.png)

This dataset has already been preprocessed (e.g., personally identifying information has already been removed.) The only additional preprocessing we performed was to remove entries that did not contain a customer ID number (which is how we define a session).  

### Setup
In NEP, we consider a user’s history to recommend items for the future—but, when training models for recommendation, all the data is historical. In order to mimic “real life” behavior, we’ll pretend that we only have access to the user’s first *n*-1 purchased items, and use those to try to predict the *n*th item purchased. 

To visualize this, let’s go back to Rhonda’s historical browsing information, collected while she was using our site. We’ll use the highlighted items as our training set to learn product representations, which will be used to generate recommendations. Recommendations are typically based on the most recent interaction by the user, called the **query item**. In this case, we’ll treat the last item (“cap” in our highlighted set of items below) as the query item, and use that to generate a set of recommendations.

The item outside of the highlighted box (in this case, a “water bottle”) will be the ground truth item, and we’ll then check whether this item is contained within our generated recommendations.

![Figure 12: Rhonda’s session, wherein the first *n*-1 items highlighted in a green box act as part of the training set, while the item outside is used as ground truth for the recommendations generated.](figures/FF19_Artboard_10rev.png)

To put it more concretely: for each customer in the Online Retail Data Set, we construct the training set from the first *n*-1 purchased items. We construct test and validation sets as a series of [query item, ground truth item] pairs. The test and validation sets must be disjoint—that is, each set is composed of pairs with no pairs shared between the two sets (or else we would leak information from our validation into the final test set!).

With this in mind, there is one more preprocessing step that we must apply to our dataset. Namely, we remove sessions that contain fewer than three purchased items. A session with only two, for example, is just a [query item, ground truth item] pair and does not give us any examples for training. 

Once we have our train/test/validation sets constructed, it’s time to train! Training [Gensim](https://radimrehurek.com/gensim/)’s word2vec is a one-liner. (For the uninitiated, Gensim is an open-source natural language processing library for training vector embeddings.) We simply pass it the training set and two very important parameters: `min_count` and `sg`. `min_count` is the minimum number of times a word in the vocabulary must be present for word2vec to create an embedding for it. Because we have some rare products, we set this to 1, so that all product IDs in the training set have an embedding. (`sg` is short for skip-gram, and setting this equal to 1 causes word2vec to use this architecture, as opposed to CBOW.)

Under the hood, word2vec will construct a “vocabulary,” a collection of all unique product IDs, and then learn an embedding for each. Once trained, we can extract the product ID embeddings.

![](figures/code_snippet.png)

Next, we need to generate recommendations. Given a query item, we’ll generate a handful of recommendations that are the most similar to that item, using cosine similarity. This is the same technique we would use if we wanted to find similar words. Instead of semantic similarity between words, we hope we have learned embeddings that capture the semantic similarity between product IDs that users purchased. Thus, we’ll look for other product IDs that are “most similar” to the query item.

![Figure 13:  The query item (the last item in a training sequence) is used to generate K product recommendations](figures/FF19_Artboard_11.png)

Armed with a list of recommendations, we can now score our model by checking whether the corresponding ground truth item is in our list of recommendations.

![Figure 14: The user’s actual next selection (the final item in the user’s sequence) is considered the ground truth item, and we check whether that item is found in our list of generated recommendations.](figures/FF19_Artboard_12.png)

We’ll perform this set of operations for each [query item, ground truth item] pair in our test set, to compute an overall score using Recall@*K* and MRR@*K*. The output of the code snippet above resulted in a Recall@10 of 19.7 and MRR@10 of 0.108. These results tell us that nearly 20% of the time, the user’s true selection was included in the list of recommendations we generated. 

### Hyperparameters Matter
In the previous section, we simply trained word2vec using the default hyperparameters-but hyperparameters matter! In addition to the learning rate or the embedding size (hyperparameters likely familiar to many), word2vec has several others which have considerable impact on the resulting embeddings. Let’s see how. 

#### Context window size
The window size controls the size of the context. Recall our earlier example: “The cat jumped over the puddle.” If the context size were set to 6, the entire sentence would be contained within the **context window**. If we set it to 5, then this sentence would be broken up, and we would instead consider a context like: “The cat jumped over the.” Thus, the context window influences how far apart words can be while still being considered in the same context.

#### Negative sampling exponent
Word2vec’s training objective is to find word representations that are useful for predicting the surrounding words in the context, by maximizing the average log probability over all possible words.^[[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)] This probability is modeled with the softmax function. Computing the softmax scales with the size of the vocabulary, as such, can become computationally expensive. However, we can approximate the softmax through a technique known as “negative sampling.” In this technique, the model must try to discern between a word that is truly in the context from words that are not part of the context.
 
Negative samples are chosen from all the other possible words in the corpus based on the frequency distribution of words in that corpus. This frequency distribution is controlled by a hyperparameter called the **negative sampling exponent**. When this value is 1, common words are more likely to be chosen as negative samples (think stopwords like “the”, “it”, “and”). If the value is 0, then each word is equally likely to be chosen as a negative example (this is a uniform distribution where “the” is just as likely as “cordial”). Finally, if the value is -1, then rare words are more likely to be selected as negative examples. 

So, we will train the model to positively identify words belonging to the same context by presenting it with pairs, like [“jumped”, “the”], [“jumped”, “cat”], etc. These are positive examples because “the” and “cat” both appear in the same context as “jumped.”  In negative sampling, we will also try to train the model to identify words that are not in the context, by presenting examples like [“jumped”, “airplane”] or [“jumped”, “cordial”]. (These are negative examples because “airplane” and “cordial” are not in the same context as “jumped.”)

#### Number of negative samples
In addition to the negative sampling exponent, another hyperparameter determines how many **negative samples** are provided for each positive example. That is, when showing the model [“jumped”, “cat”] (our positive sample), we could include one, five, or maybe even fifty different negative samples. 

The code snippet displayed above uses the default values for each of these hyperparameters. These values were found to produce semantically meaningful representations for words in documents, but we are learning embeddings for products in online sessions. The order of products in online sessions will not have the same structure as words in sentences, so we’ll need to consider adjusting word2vec’s hyperparameters to be more appropriate to the task.

| Hyperparameter | Start Value | End Value | Step Size | Configurations |
| ------ | ------ | ------ | ------ | ------ |
| Context window size | 1 | 19 | 3 | 7 |
| Negative sampling exponent | -1 | 1 | 0.2 | 11 |
| Number of negative samples | 1 | 19 | 3 | 7 |
| **Number of Trials** |  |  |  | 539 |

![Table 1: This table shows the main hyperparameters we tuned over. For each one, we show the starting and ending values we tried, along with the 
step size we used. The total number of trials is computed by multiplying each value in the Configurations column.]

Above, we detail the hyperparameters we considered, the values we allowed these hyperparameters to assume, and the total number of trials necessary to test them all in a sweep. If we want to find the best hyperparameters for our dataset, we’ll need to do quite a bit of training—more than 500 different hyperparameter combinations! We _could_ set up our own code, constructing several nested loops to cover all of the possible parameters—or perhaps we could use sklearn’s GridSearch. However, there’s an even better solution.

### Hyperparameter Tuning with Ray
There are several libraries that make hyperparameter optimization approachable, easy, and scalable (three things you won’t get by rolling your own or using sklearn’s GridSearch). For this cycle, we explored [Ray](https://docs.ray.io/en/master/index.html). 

At its core, Ray is a simple, universal API for building distributed applications. Atop this foundation are a handful of libraries designed to address specific machine learning challenges. [Ray Tune](https://docs.ray.io/en/master/tune/index.html) provides several desirable features, including distributed hyperparameter sweep, checkpointing, and state-of-the-art hyperparameter search algorithms—all while supporting most major ML frameworks, such as PyTorch, Tensorflow, and Keras. 

In our experiments, we tuned over the hyperparameters in the table above for the number of trials specified. Each trial was trained for 100 epochs and was evaluated on the validation set using Recall@10. This resulted in the best hyperparameter configuration for the Online Retail Data Set.  

#### Results
We trained a word2vec model using the best hyperparameters found above, which resulted in a Recall@10 score of 25.18+-0.19 on the validation set, and 25.21+-.26 for the test set. These scores may not seem immediately impressive, but if we consider that there are more than 3600 different products to recommend, this is far better than random chance!

#### Hyperparameter Optimization Results  
For this dataset, there is actually a wide range of values that would work well for this task. In the following figures, we plot the Recall@10 score for each  hyperparameter configuration we tested. Because we tuned over three hyperparameters, we display three figures showing the relationship between pairs of hyperparameters. Essentially, we’ve flattened a 3D space into two dimensions for readability. This means that, for the flattened dimension, we averaged over the Recall@10 scores. For example, in the left-most figure we plot the *number of negative samples* against the *negative sampling exponent*. We average over the Recall@10 scores in the context window dimension, which are the colored points you see in the 2D figure, where yellow indicates a high Recall@10 score and purple is a low score.

![Figure 15: Results from our hyperparameter sweep: Each panel shows the Recall@10 scores (colored points, where yellow is a high score, and purple is a low score) associated with a unique configuration of hyperparameters. The best hyperparameter values for the Online Retail Data Set are denoted by the light blue circle. Word2vec’s default values are shown by the orange star. In all cases, the orange star is nowhere near the light blue circle, indicating that the default values are not optimal for this dataset.](figures/hpsweep_results.png)

In each figure above, the orange star signifies the default word2vec values, and the light blue circle indicates the best hyperparameter configuration we found during our sweep. In all cases, the best hyperparameters are typically in the upper right quadrant; larger values of these hyperparameters perform better than smaller values for this dataset. And in each case, the default word2vec parameters are near, but outside of, the optimal range of values to maximise performance on Recall@10.  

We also find two other items of note. First is the size of the context window. In the middle figure, we see that many values of context window size (y axis) will work pretty well, so long as it’s larger than one. (Note that the bottom row of this figure is quite purple, indicating very low relative scores). When the context window is too small, the model is unable to learn relationships between product IDs, and thus recommendations (and our Recall@10 score) suffer. Context matters!

Second is the number of negative samples. In the figure to the far right, we see that more negative samples lead to much better Recall@10 scores. When we select only a single negative example for each positive example, the model again struggles. It is crucial to provide the model with sufficient negative examples. The model must learn not only what things _should_ be in the same context, but especially what things _should not_ be in the same context.

#### Model Comparisons
Now that we’ve learned the best hyperparameters for our dataset, we can start looking at some model comparisons. We trained models using both our best hyperparameters and the default hyperparameters, each for 100 epochs. The number of epochs is another important parameter, and the default in Gensim’s implementation is five. So we trained another default word2vec model with only five epochs. Finally, we’ve shown the “Association Rules” baseline, a simple heuristic that predicts recommendations based on frequent co-occurrence between items.

![Figure 16: Model comparisons](figures/model_comparison.png)

It should be no surprise that the best model is the one configured with the hyperparameters discovered during the tuning sweep—but it turns out the number of training epochs is just as crucial. 

We can see the relative contribution of this parameter in Figure 16. The blue bar is the Recall@10 score for a word2vec model trained with all default values, including Gensim’s default of only five epochs of training. The orange bar is the same model trained for 100 epochs. This indicates that simply training word2vec for more epochs can give you a big boost in performance, without doing any hyperparameter sweep at all. But if we train for too many epochs, shouldn’t we be worried about overfitting? Not with word2vec. 

With traditional neural networks, we set the number of epochs to minimize the training loss (and increase learning) without overfitting (typically indicated by an increase in the validation loss). However, word2vec is different. We do not care necessarily about the training loss because the goal is to learn embeddings for a _downstream_ task. In this sense, it might not actually matter if the model overfits the training data, so long as the resulting embeddings increase performance on the downstream task, according to whatever metrics we’ve implemented. Therefore, it is almost always beneficial to train word2vec _for as many epochs as resources allow_, or until the downstream task has reached a performance plateau—in which case, additional training does not yield an increase in the downstream metric. 

We can see this effect in the figure below, where we plot word2vec’s training loss in grey and the Recall@10 score on the validation set in blue (since there is no validation loss in this case) as a function of training epochs. The Recall@10 score is completely decoupled from word2vec’s training loss and steadily increases over the course of training. By the end of the 100 epochs, this score has relatively flattened. We tried training for longer (200 epochs), but this didn’t significantly increase performance. 

![Figure 17: Recall@10 score on the validation set as a function of epochs, where the dark line indicates the mean over 5 models trained on the same (best) hyperparameters, and the light shading is one standard deviation.](figures/loss_recall_vs_epochs.png)

Of course, it’s still better to determine the best set of hyperparameters for the specific dataset you are working with; however, these results demonstrate that you can get far by simply increasing the number of training epochs. We estimate that the increase in training epochs accounts for up to 58% of the performance gains for our best word2vec model (compared to the model trained on default values for only 5 epochs). This suggests that having ideal hyperparameters accounts for roughly 42% of the performance gains, which has performance implications of another kind: computational. Let’s look at some of the challenges of this method for session-based recommendations, starting with the computational cost. 

### Challenges
While using word2vec to learn product embeddings for recommendations is relatively new and somewhat intuitive, it does come with some challenges. Some of these challenges we faced directly in our experiment, but others are likely to crop up when using this approach for different types of datasets or use cases.  

#### Computational Resources
Using word2vec for product embeddings in a recommendation system can be a viable approach, but for some datasets, identifying the right hyperparameters to optimize those embeddings is a must. The problem is that hyperparameter searches are expensive, requiring one to train hundreds, or even thousands, of model configurations. While the Gensim library^[[https://radimrehurek.com/gensim/index.html](https://radimrehurek.com/gensim/index.html)] is one of the fastest for training vector embeddings, it only supports CPU. GPU support *is* possible with a Keras wrapper, but benchmarks^[[Gensim word2vec on CPU faster than Word2veckeras on GPU](https://rare-technologies.com/gensim-word2vec-on-cpu-faster-than-word2veckeras-on-gpu-incubator-student-blog/)] indicate it is actually slower than the original Gensim CPU version.^[One promising lead is the implementation by AWS, [BlazingText](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-blazingtext-parallelizing-word2vec-on-multiple-cpus-or-gpus/), which supports multiple CPUs or GPUs. Their benchmarks claim to dramatically decrease training time.]

However, there are some strategies to mitigate this challenge: 
>1. Perform hyperparameter optimization on a subsample of the original dataset. Although we did not perform this experiment, research^[[Tuning Word2vec for Large Scale Recommendation Systems](https://arxiv.org/abs/2009.12192)] has indicated that, for certain datasets, performing hyperparameter optimization on a 10% subsample is sufficient to find good hyperparameters at a fraction of the computational cost. 
>2. Perform a smarter hyperparameter sweep using one of many hyperparameter optimization search algorithms. The idea here is that, overall, fewer trials are needed to find the optimal hyperparameters, thus saving CPU hours.

#### Metrics
Recall@*K* and MRR@*K* are two common metrics for NEP, and provide a way to assess various models on equal footing. However, they don’t directly measure what a business might truly wish to optimize: revenue, watch/listen time, etc.

**Recall@*K***
This metric is most practical for settings in which the absolute order does not matter (e.g., the recommendations are not highlighted and they are all shown on the screen at the same time). This might be the case for a website that displays ten or twenty recommendations at once, allowing the user to explore options. However, it’s a harsh metric because it doesn’t assign a score to other items that might be nearly identical to the ground truth item. Recall@*K* simply assigns a 1 or a 0 depending on whether the user’s true choice was included in the list of recommendations. It does not score any number of similar products in that recommendation list that, in reality, might have been equally acceptable to the user. 

**MRR@*K***
Mean reciprocal rank is a better choice for applications in which the order of the recommendation matters (e.g., lower ranked recommendations are only shown once the user scrolls to the bottom of the page). In this case, having a higher MRR is crucial, indicating the best recommendations are near the top. Again, this metric only assigns a score if the user’s true choice was included in the list of recommendations and does not give “partial credit” to similar items.

While both of these metrics share similarities with real-world use cases, neither of them directly correlates with increased revenue, watch/listen time, or other real-world KPIs. Furthermore, these metrics do not take into account the *quality* of the resulting recommendations (more on this in [Online evaluation](#online-evaluation), below). To assess these real-world outcomes, any session-based recommender must be subjected to live, online A/B testing.

#### The cold start problem
The cold start problem afflicts nearly all recommendation systems, and usually comes in two flavors: new users and new products. For session-based recommendation systems, new users are typically not a problem because these systems do not rely explicitly on user characteristics. As long as a website has **some** historical user base, that data can be used to generate embeddings for existing products based on how users navigate the site. These embeddings can then generate reasonable recommendations for new users.

However, new products still present a challenge. These items do not have any sessions associated with them in the training dataset; hence, there are no embeddings associated with them, either. A possible solution to this could be to look at a few similar items (for instance, similar products based on domain, or similar music style) and then assign an embedding that is the average of the embedding vectors of the similar items to create an initial vector.

#### Embeddings and scalability
A general challenge when using word2vec embeddings is that they can be computationally demanding,^[[Network–Efficient Distributed Word2vec Training System for Large Vocabularies](https://arxiv.org/abs/1606.08495)] especially compared to simpler heuristics (such as co-occurrence-based recommendations). Embedding methods require substantial amounts of training data to be effective. Also, in order to actually recommend an item, one needs to compute the closest items (based on cosine similarity, nearest neighbor, or some other distance metric) to a given item using the embeddings. This could be challenging to compute in real time. One way to get around it would be to pre-compute and store the *k*-closest items to an item for easy lookup when needed. There are also indications that embedding methods^[[Evaluation of Session-based Recommendation Algorithms](https://arxiv.org/abs/1803.09587)] may not perform as well as purely sequential models, such as RNNs, although this is likely use-case dependent.
