## Defining the Session-based Recommender Problem Space 

Let’s say we own a popular online shopping website for workout accessories. Rhonda, a new customer, has been browsing tops, shoes, and weights. Her browsing history looks like this:

![Figure 4:  Rhonda’s browsing history](figures/FF19_Artboard_4rev.png)

What should we recommend to her next? Good recommendations increase the likelihood that Rhonda will see something she likes, click on it, and make a purchase. Poor recommendations will, at best, lead to no new revenue, but-even worse-could give her a negative customer experience. (You know this feeling: when a website keeps recommending something to you that you have already bought, or something that you’ve never really wanted, your impression of that website diminishes!) 

We’ll consider Rhonda’s recent browsing history as a “session.” Formally, a session is composed of multiple user interactions that happen together in a continuous period of time—for instance, products purchased in a single transaction. Sessions can occur on the same day, or across several days, weeks, or months. 

Our goal is to predict the product within Rhonda’s session that she will like enough to click on. This task is called **next event prediction** (NEP): given a series of events (Rhonda’s browsing history), we want to predict the next event (Rhonda clicking on a product we recommend to her). 

In reality, this means that our model might generate a handful of recommendations based on Rhonda’s browsing history; we want to maximize the likelihood that Rhonda clicks on at least one of them. To train a model for this task, we’ll need to use historical browsing sessions from our other existing users to identify trends between products that will help us learn recommendations. 

![Figure 5: Historical browsing sessions of various lengths](figures/FF19_Artboard_5.png)

### Use Cases

This problem is well-aligned with emerging real-world use cases, in which modeling short-term preferences is highly desirable. Consider the following examples in music, rental, and product spaces.^[Adopted from [Applying word2vec to Recommenders and Advertising](https://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/)]

#### Music recommendations
Recommending additional content that a user might like while they browse through a list of songs can change a user’s experience on a content platform.

The user’s listening queue follows a sequence. For each song the user has listened to in the past, we would want to identify the songs listened to directly before and after it, and use them to teach the machine learning model that those songs somehow belong to the same context. This allows us to find songs that are similar, and provide better recommendations.^[[Using Word2vec for Music Recommendations](https://towardsdatascience.com/using-word2vec-for-music-recommendations-bb9649ac2484)]

![Figure 6: Playlist](figures/FF19_Artboard_6rev.png)

#### Rental recommendations
Another powerful and useful application of session-based recommendation systems occurs in any type of online marketplace. For example, imagine a website that contains millions of diverse rental listings, and a guest exploring them in search of a place to rent for a vacation.^[[Listing Embeddings in Search Ranking](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e)] The machine learning model in such a situation should be able to leverage what the guest views during an ongoing search, and learn from these search sessions the similarities between the listings. The similarities learned by the model could potentially encode listing features-like location, price, amenities, design taste, and architecture.

![Figure 7: Rental listings](figures/FF19_Artboard_7rev.png)

#### Product recommendations
Leveraging emails in the forms of promotions and purchase receipts to recommend the next item to be purchased has also proven to be a strong purchase intent signal.^[[E-commerce in Your Inbox:
Product Recommendations at Scale](https://arxiv.org/pdf/1606.07154.pdf) (PDF)] Again, the idea here is to learn a representation of products from historical sequences of product purchases, under the assumption that products with similar contexts (that is, surrounding purchases) can help recommend more meaningful and diverse suggestions for the next product a user might want to purchase.

![Figure 8: Email purchase receipts](figures/FF19_Artboard_8rev.png)

With these examples in mind, let’s dig deeper into what it takes to design and build a session-based recommendation system for product recommendations, in the context of an online retail website.
