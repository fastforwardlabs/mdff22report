## Introduction

After iterations of development and testing, deploying a well-fit machine learning model often feels like the final hurdle for an eager data science team. In practice however, a trained model is never final, and this milestone marks just the beginning of the perpetual maintenance race that is production machine learning. This is because most machine learning models are static, but the world we live in is dynamic. More specifically, the ability of a trained model to generalize relies on an important assumption of stationarity - meaning the data upon which a model is trained and tested are independent and identically distributed (i.i.d). In real-world environments, this assumption is often violated as human behavior and consequently the systems we aim to model are dynamically changing all time. 

![Figure 1: The sheer amount of items available online make recommendation systems necessary](figures/FF19_Artboard_1.png)

While this report is not comprehensive, we will touch on a variety of approaches to recommendation systems, and dig deep into one approach in particular. We’ll demonstrate how we used that approach to build a recommendation system from the ground up for an e-commerce use case, and showcase our experimental findings. Finally, we’ll also talk about many of the considerations necessary to building thoughtful, information-driven recommendation systems.

### Session-based Recommender Systems
Recommendation systems are not new, and they have already achieved great success over the past ten years through a variety of approaches. These classic recommendation systems can be broadly categorized as content-based, as collaborative filtering-based, or as hybrid approaches that combine aspects of the two.

At a high level, content-based filtering makes recommendations based on user preferences for product features, as identified through either the user's previous actions or explicit feedback. Collaborative filtering, on the other hand, utilizes user-item interactions across a _population_ of users in order to make recommendations for one particular user, based on the preferences of other, very similar users (where similar users are identified by the items they have liked, read, bought, watched, etc.). These systems generally tend to utilize historical user-item interactions (i.e., the items that a user has clicked on in the past) to learn a user’s long-term preferences. 

![Figure 2: Content-based vs Collaborative filtering approaches](figures/FF19_Artboard_2rev.png)

The underlying assumption in both of these systems is that all of the historical interactions are equally important to the user’s current preference—but in reality, this may not be true. A user’s choice of items not only depends on long-term historical preference, but also on short-term and more recent preferences. 

Choices almost always have time-sensitive context; for instance, “recently viewed” or “recently purchased” items may actually be more relevant than others. These short-term preferences are embedded in the user’s most recent interactions, but may account for only a small proportion of historical interactions. In addition, a user’s preference towards certain items can tend to be dynamic rather than static; it often evolves over time. 

![Figure 3: A user’s preference can change over time](figures/FF19_Artboard_3rev.png)

These considerations have prompted the exploration and development of a new class of recommendation algorithms: known as **session-based recommendation algorithms**, these rely heavily on the user’s most recent interactions, rather than on the user’s historical preferences. In addition, this approach is especially advantageous because a user _could_ appear anonymously—that is, a user may not be logged in or may be browsing incognito.  

### Why now?
While nearly unknown as of just a few years ago, session-based recommenders have grown quickly in popularity, and for several reasons. First, this method can be implemented _even in the absence of historical user data_, and doesn’t explicitly rely on user population statistics. This is helpful because, as just noted above, users aren’t always logged in when they browse a website, which makes session-based recommenders highly relevant.^[[Empirical Analysis of Session-Based Recommendation Algorithms](https://arxiv.org/abs/1910.12781)]

Second, a wealth of new, publicly available, session-centric datasets have been released, especially in the e-commerce domain, allowing for model development and research in this area.

Third, session-based recommenders have benefited from the rise of deep learning approaches expressly suited for sequences (more on this in [Modeling Session-based Recommenders](#modeling-session-based-recommenders), below).
