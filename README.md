# DelayLearning
Implementation of different mechanisms of delay learning, using SNN and applied to event data (produced by DVS camera)

Libraries used: 
- tonic : to handle event based data
- PyNN : to implement spiking neural networks

## Organisation of the repository
> - Nadafian: this directory extends a previous implementation of the work published in [[1]](#1). The first implementation was produced by Hugo Bulzomi during his internship at I3S / CNRS, UCA in Fall 2021 and can be found [on this github repertory](https://github.com/HugoBulzomi/SNN_Delay_Learning). The main improvement in the current implementation is the computation time: 6 minutes to simulate the network with two convolution layers during 1 second, versus 43 minutes previously. 

## References
<a id="1">[1]</a> 
Nadafian, A. and Ganjtabesh, M. (2020). 
Bio-plausible Unsupervised Delay Learning for Extracting Temporal Features in Spiking Neural Networks.
arXiv:2011.09380[cs,q-bio].
