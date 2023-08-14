# Edmund 08-08-2023

I am preparing a meeting with Edmund to discuss some work that I am doing on singular learning theory (SLT). The project is about looking at the behavior of stochastic gradient descent (SGD) on singular statistical models. 
Here are the following point that I would like to discuss:
Having a better understanding of the problem:
* My understanding of the problem: I think that currently one of the main theoretical obstacle to applying SLT to  is that we do not have a good understanding of how SGD behaves on singular model and in particular if it prefers places of the phase space with lower RLCT. To understand this we would like to see if SGD converges towards region of lower RLCT and in particular if it preferentially stays there. 
* Another picture that I am confused about is the phase transition story. I think in the Bayes picture the phase transition make sense as the Bayes posterior can update directly to one or other region of parameter space that have lower free energy. But in the SGD picture I would expect that this happen continuously, so it would be harder to see phase transitions? Also the potential should change its shape since it is not the same for SGD and Bayes (SGD, the potential has much lower samples)
What I have explored so far:
* I have been wondering about applying SGD on very simple singular models. I considered a polynomial regression and did some analytic calculation. The main insight seems to be that if we do the gaussian approximation, then the covariance of the gradient of the gaussian field is related to the second derivative of the KL divergence. If we put ourselves at some epsilon distance of the singularity then the covariance of the gaussian field is higher if locally there is a bigger singularity. This suggest that SGD will spend more time on the singularity with higher RLCT. Something that still confuses me with this: Strictly speaking we cannot take the gaussian limit since we have small batches (unless we consider batches to be large enough?). Also interestingly: the fluctuations becomes the noise of gradient descent (in fact what we do seems similar to approximate SGD with GD + gaussian noise). I seem to remember that people where thinking the gap was mostly between GD + gaussian noise and SGD rather than GD and GD + noise. Show the calculations.
* Now I want to gain better intuitions by simulating a simple 1D singular models with 2 different singularities. And run SGD on this model. 

Some questions that I want to ask to Edmund:
* Are there other theoretical questions that you think are particularly important that I am missing?
* Do you have some recommendations on other directions to pursue?
* Can you suggest improvements on what I am doing? For example the simulation?
* Are there important aspects of the problem that I am misunderstanding? 

Evaluate my plan for the discussion (this is a 1h discussion) and suggest improvements.


Top level perspective:

* How singularities affect SGD?
* Critical points and their degenaricies are the main organising? 
* SLT is more about the model which has critical points with various deg. SLT has the most result in Bayesian learning. Link the generalisation error to the Bayesian posterior.
* Calibrate the difficulty:
  * SGD in a regular model is not a well studied thing. Most studies assume that the gradient noise (GD + gaussian noise). Well studied: continuous version of this is well studied, close formed solution. Finite step size is difficult.

  Paper: SGD with constant learning rate can converge to local maxima. Continous approximation destroy some behaviour. 

Fokker Planck equation: To do the SGD experiment. Data distribution need to decay more slowly. 

GD is lower near the minima. 

More clearly formulated: Probability of escaping.

First level:
* Global topological level
* Morse smale complex.
* Morse function: all critical point are isolated
* Two questions: where a gradient flow would end up?
* Global question: Where does gradient flow
* Local: What is the chance of escaping ?
* Is continuous approximation good for the singular

Look at the continuous

Fokker Plank?

Send invide to Edmund

Ask person I know

# Nicolas 08-08-2023

* Discussion with Edmund:
  * Similar perspective on the problem
  * Review calculation made with covariance of noise
  * Suggestion to work on explicit calculation with \xi_b
  * Difficulty calibration SGD and GD + noise 
* Progress on the simulation
  * SGD on 1D model
* Look at bibliography on agency

ToDo

* Review of literature of SGD
* Check differential and Fokker plank
* Numerical experiments
* Explicit \xi_b
* Collaboration good

Focaliser sur choses numeriques ou analytiques?

Quel models sont interessant a regarder. 

Why model 2D and not 1D?

Discuss with JC Mourrat

Look at Fokker Plank equation

Next steps:
