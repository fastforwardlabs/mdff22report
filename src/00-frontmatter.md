# Inferring Concept Drift Without Labeled Data

FF22 · _Aug 2021_

![](figures/ff22_cover_splash.png)

*This is an applied research report by [Cloudera Fast Forward](https://www.cloudera.com/products/fast-forward-labs-research.html). We write reports about emerging technologies, 
and conduct experiments to explore what's possible. Read our full report on **Concept Drift** below, or <a href="/FF22-Concept_Drift-Cloudera_Fast_Forward.pdf" target="_blank" id="report-pdf-download">download the PDF</a>, and be sure to check out [our github repo](https://github.com/fastforwardlabs/concept-drift) for the [Experiments](#Experiments) section.*

*After rigorous iterations of development and testing, deploying a well-fit machine learning model often feels like the final hurdle for an eager data science team. In practice however, a trained model is never final, and this milestone marks just the beginning of the perpetual maintenance race that is production machine learning. This is because most machine 
learning models are static, but the world we live in is dynamic. More specifically, the ability of a trained model to generalize relies on an important assumption of stationarity - meaning the data upon which a model is trained and tested are independent and identically distributed (i.i.d). In real-world environments, this assumption is often violated as human behavior and consequently the systems we aim to model are dynamically changing all time.This report explores approaches for dealing with such dynamically changing environments through concept drift when labeled data is not readily accessible.*

[[TOC]]
