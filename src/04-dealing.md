## Addressing the problem

Due to these limitations, there is a clear need for effective methods that can detect real concept drift (Source 2) in an unsupervised manner. Unfortunately, this proves to be an impossible task, as the only way to confirm a change in \\(P(y|X)\\) with certainty is to have some access to ground truth labels. 

In the absence of labeled data, the best we can do is attempt to _infer_ real concept drift by detecting feature drift (Source 1). That is, we are interested in quantifying visible changes in \\(P(X)\\) and surmising that those changes correspond to meaningful change in the classification boundary \\(P(y|X)\\). Of course, this approach is prone to error because as we’ve seen:
>1. Not all changes in \\(P(y|X)\\) are visible from \\(P(X)\\), resulting in false negative detections where real drift occurs but is not signaled.
>2. Not all changes in \\(P(X)\\) actually affect \\(P(y|X)\\), resulting in overly sensitive detectors that trigger costly false positive detections.

Inferring real concept drift in an unsupervised fashion thus becomes a delicate balancing act, in order to minimize the number of false positive detections (and therefore labels needed) while remaining sensitive enough to pick up on meaningful changes in the feature space that _likely_ contribute to a change in concept.
