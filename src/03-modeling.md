## Modeling Session-based Recommenders

There are many baselines for the next event prediction (NEP) task. The simplest and most common are designed to recommend the item that most frequently co-occurs with the last item in the session. Known as “Association Rules,”^[[Evaluation of Session-based Recommendation Algorithms](https://arxiv.org/abs/1803.09587)] this heuristic is straightforward, but doesn’t capture the complexity of the user’s session history.  

More recently, deep learning approaches have begun to make waves. Variations of graph neural networks^[[Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)] and recurrent neural networks^[[Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)] have been applied to the problem with promising results, and currently represent the state of the art in NEP for several use cases. However, while these algorithms capture complexity, they can also be difficult to understand, unintuitive in their recommendations, and not *always* better than comparably simple algorithms (in terms of prediction accuracy).

There is still another option, though, that sits between simple heuristics and deep learning algorithms. It’s a model that can capture semantic complexity with only a single layer: word2vec.  

### Treating it as a NLP problem
Word2vec? Isn’t that the algorithm that made word embeddings commonplace? Yes! Word2vec uses the co-occurrence of words in a sentence to learn embeddings for each word that capture the semantic meaning of that word. At its core, word2vec is a simple, shallow neural network, with a single hidden layer. In its skip-gram version, it takes as input a word, and tries to predict the context of words around it as the output. For instance, consider this sentence:

“The cat _jumped_ over the puddle.”^[Example taken from Stanford’s popular CS224 NLP lecture series]

Given the central word “jumped,” the model will be able to predict the surrounding words: “The,” “cat,” “over,” “the,” “puddle.”

![Figure 9: Word2vec versions: Skip-Gram vs Continuous Bag of Words](figures/FF19_Artboard_9.png)

Another approach—Continuous Bag of Words (CBOW)—treats the words “The,” “cat,” “over,” “the,” and “puddle” as the context, and predicts the center word: “jumped.” For the rest of this report, we will restrict ourselves to the skip-gram model. The Ws in the above diagram represent the weight matrices that control the weight of the successive transformations we apply to the input to get the output. Training this shallow network means learning the values of these weight matrices, which gives us the output that is closest to the training data. Once trained, the output layer is usually discarded, and the hidden layer (also known as the embeddings) is used for downstream processes. These embeddings are nothing but vector representations of each word, such that similar words have vector representations that are close together in the embedding space. 

### How can we use this for NEP? 
Let’s take another look at Rhonda’s browsing history. We can treat each session as a sentence, with each item or product in the session representing a “word.” A website’s collection of user browser histories (including Rhonda’s) will act as the corpus. Word2vec will crunch over the entire corpus, learning relationships between products in the context of user browsing behavior. The result will be a collection of embeddings: one for each product. The idea is that these learned product embeddings will contain more information than a simple heuristic, and training the word2vec algorithm is typically faster and easier than training more complex, data-hungry deep learning algorithms. 

In fact, word2vec can be invoked as a reasonable approach any time we’re faced with a problem that is sequential in nature, and where the order of the sequences contains information. In this case, casting sessions as an NLP problem makes sense because we do have sequential data (user browser histories are typically ordered by time) and the order likely does matter (capturing the user’s interests as they navigate various products). While the basic algorithm can conceptually be applied to other domains, only recently has research explored this explicitly for the NEP task.^[[Word2Vec applied to Recommendation: Hyperparameters Matter](https://arxiv.org/abs/1804.04212)] ^[[Tuning Word2vec for Large Scale Recommendation Systems](https://arxiv.org/abs/2009.12192)]
