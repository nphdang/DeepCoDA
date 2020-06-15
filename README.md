# DeepCoDA: deep learning model for personalized interpretability for compositional health
This is the implementation of the DeepCoDA model in the paper "DeepCoDA: personalized interpretability for compositional health", ICML 2020: https://arxiv.org/pdf/2006.01392.pdf

# Introduction
Interpretability allows the domain-expert to directly evaluate the model's relevance and reliability, a practice that offers assurance and builds trust. In the healthcare setting, interpretable models should implicate relevant biological mechanisms independent of technical factors like data pre-processing. 

We define personalized interpretability as a measure of sample-specific feature attribution, and view it as a minimum requirement for a precision health model to justify its conclusions. Some health data, especially those generated by high-throughput sequencing experiments, have nuances that compromise precision health models and their interpretation. These data are compositional, meaning that each feature is conditionally dependent on all other features. 

We propose the DeepCoDA framework to extend precision health modelling to high-dimensional compositional data, and to provide personalized interpretability through patient-specific weights. Our architecture maintains state-of-the-art performance across 25 real-world data sets, all while producing interpretations that are both personalized and fully coherent for compositional data.

## DeepCoDA model architecture
