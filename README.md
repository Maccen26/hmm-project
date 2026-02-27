# $ \text{CO}^2 $ modelling with Hidden Markov Models 

This project is special couse a Denmarks Technical University which explores how to model $CO^2$ data with Hidden Markov Models. The couse has been conducted in the Spring Semester 2026 and is 5 ETCS points. 

## Scope of project
The projects leads up to a Bachelor Projects, and this course therefore focus on: 

- Designing flexible software for implementation of HMM's 
- Exploring & understanding what HMM's are and how they can used. This could include a lof of notes around the code and in the jupyternotebook files. 
- Practiceing interpretability of HMM and their states
- Testing HMM's 



## Generel about the project
Package manager is uv. 

## AI Disclosure in this project. 
1. Copilot chat completation has been used. 
2. No Agents has written code. 
3. Agents has been used to debug Jax Modules (sometimes). 
4. Claude has been used to find sources and explain concepts.  
5. Claude code has been used to generate documentation about my code.


## Notes

### Software Design 
3 classes: HMM Class, Transition Class, Emmission class. \
The responsibilities of each class are: 
1. Emission: Compute the state distribution given the $y_i$ observation. Base class should take $x$ arguments as covariates
2. Transition: Compute the transition matrix at timestep $i$ given obs $y_i$. Should take $x$ as arguments as covarites. 
3. HMM Model: Compose a transition and a emission class. The goal of this class is to step trough the markov chain using the state distributions and transition matrix. The class should also take a initial state distribution as argument such that we can vary this depending on the state we start in. 





