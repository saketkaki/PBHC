"""Visualization of the latent space for encoded observations.
This will be done to examine if there exist trends in the observations for each attack performed by the adversary.

Pull any run and visualize its encoded observations in the latent space.
The encoder for the attack observation will output a 32-dimensional vector which is compressed into a 13-dimensional vector.
The 13-dim vector consists of 8 class logits (one for each attack), 4 phase logits, and one time-to-impact logit.
"""