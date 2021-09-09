## Conclusion

As we’ve learned, maintaining a static representation of an ever-changing environment is challenging, and requires diligent performance monitoring to signal when a machine learning model is no longer suited for its original task. This issue becomes even more difficult when the cost or availability of ground truth labels make performance-based drift detection methods infeasible － which is often the case in real world applications. 

In this scenario, teams must monitor and detect changes purely from independent variables as a means to infer concept drift. Unfortunately, monitoring changes in input distributions produces many false positive detections, because not all changes in the feature space of a population actually correspond to a meaningful drift in relation to the target variable.

In this report, we presented four ways to infer concept drift in an unsupervised manner, with the goal of reducing false positive drift detections. We reported experimental results comparing and contrasting the nuances of each method, and conclude that the best approach for detecting drift without labels will depend on your specific application’s tolerance for error.

We hope this report has brought to light a few practical challenges associated with production machine learning, and we look forward to continued research in this space!
