# DeepArc: Towards the modularization of neural networks for maintanents

Neural networks are an emerging data-driven programming paradigm widely used in many areas. Unlike traditional software systems consisting of decomposable modules, a neural network is usually delivered as a monolithic package, raising challenges for some maintenance tasks such as model restructure and retraining. we propose DeepArc, a novel modularization method for neural networks, to reduce the cost of model maintenance tasks such as model restructure and re-adaption. Specifically, DeepArc decomposes a neural network into several consecutive modules, each of which encapsulates consecutive layers with similar semantics. The network modularization facilitates practical tasks such as refactoring the model to preserve existing features (e.g., model compression) and enhancing the model with new features (e.g., fitting new samples). The modularization and encapsulation allow us to restructure or retrain the model by only pruning and tuning a few localized neurons and layers. (1) the architectural bad smell of a network model so that we can compress modules for effective model compression and (2) the cost-saving opportunities to boost the model performance by only retraining few module weights. Our experiments show that, (1) DeepArc can boost the runtime efficiency of the state-of-the-art model compression techniques by 14.8%;(2) compared to the traditional model retraining, DeepArc only needs to train less than 20% of the neurons to fit adversarial samples and repair under-performing models, leading to 32.85% faster training performance while achieving similar model prediction performance. 

The more information can be found at https://sites.google.com/view/deep-arc.

The source code and the results can be found at https://github.com/hnurxn/Deep-Arc.
# Contact Lists

If you have any question, please contact the authors:Ren Xiaoning [hnurxn@ustc.mail.edu.cn]

# DOI


[DOI: 10.1109/ICSE48619.2023.00092](https://ieeexplore.ieee.org/document/10172675)


