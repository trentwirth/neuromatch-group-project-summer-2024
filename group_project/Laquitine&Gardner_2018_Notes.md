
Abstract clarification:

I wasn't sure what this sentence meant; "We found that despite excellent agreement of estimates mean and variability with a Basic Bayesian observer model, the estimate distributions were bimodal with unpre- dicted modes near the prior and the likelihood." 

This snippet I think clears it up:
> “The estimate distributions were bimodal with unpredicted modes near the prior and the likelihood”: Despite the model’s success in predicting the general trends of the estimates, the actual distribution of those estimates—how they spread out across possible values—showed a pattern that the model did not predict. Instead of showing a single peak (unimodal distribution) that would suggest a smooth integration of prior knowledge and sensory evidence, the distributions were bimodal. This means there were two peaks in the distributions of estimates. One peak was near the value dictated by the prior knowledge, and another near the sensory evidence itself. This bimodality suggests that rather than integrating the two sources of information, participants might have been switching between relying on prior knowledge and relying directly on the sensory evidence, depending on the trial.

"Switching observer model" -> a model that switches between a Basic Bayesian observer and a model that is driven by the sensory information on a given trial.

I'm curious what their model "that exclusively relied on sensory likelihood" looks like.

How would you be able to demonstrate that the prior set *is not* learned?

If we plot the error from the sensory information vs. the error from the global mean across trials, we might be able to see if participants are learning

# Research Questions

1. Can we recreate their models?
2. I don't personally believe in the strong Bayesian story; so I believe that a better model can be developed. What if we're seeing a constant hysterisis effect occurring - participants reach a certain state basedon the stimuli and go in and out of it, this could perhaps explain the bimodality of the observed data. There might not be a need for hysterisis; we could have competition between the stimulus signal and the "historical mean" of the past N trials. N could be a parameter we fit. The influence of the stimulus mean is weighted by its coherence. This weight is inversely connected to the "historical mean"
2. Question Trent has: A heuristic model that then transitions to a bayesian one might provide the best description of behavior, but it doesn't explain the "why" of the behavioral bimodality. A why model here would be really cool, especially because it might be able to inform a new "how" model...
3. Behavioral research question Trent has: would this experiment work differently if the motion was presented exclusively in the periphery?


# Modeling Tutorial, Walking through the steps...

The phenomenon: Participants perceived a dot-motion stimulus that was manipulated by two experimental conditions: 1. the coherence of the motion (how many dots were moving in the same direction) [high: 0.24, medium: 0.12, low: 0.06] and 2. the "prior block", where every trial within a block has a mean belonging to a distributions of M = 225 Deg with a varying standard devionat [80, 40, 20, 10 STD]. Experimenters found that a a model that switched between utilzing the prior mean and the sensory evidence mean of the motion direction was most effective in predicting subject performance, as a function of the strength of the sensory evidence.

Critical: They claim that the priors are learned within 45 trials. Apparently, this is typical of prior learning - something that is a little strange to me, though, is that the prior strength varies from block to block, but they allow the priors to be learnt exponentially throughout the experiment. They use this as evidence that people really are switching between a prior and sensory evidence.

Let's imagine that we do not buy the Bayesian story. What are the alternatives?
- Dynamic systems model with Hysteresis. 
- Neural Network model with competition between the sensory evidence and the historical mean of the past N trials.

Both of these alternatives take into consideration the trial history. 

We will know if our model performs better if the trial-to-trial error is lower than the predicted estimate generated by the switching mode - and if we can still approximate the distributions of the estimates across conditions.

So, what we'd be modeling:
- Angular estimate
- With inputs:
    - Coherence
    - Trial history

Q: Can some what models tell us anything about the trial history? 

Our approach will be falsified if, by accounting for trial history, nothing is gained - or we do significantly worse than accounting for the "prior" mean.
> Note: Hysterisis might be modulated by the coherence of the previous trials; this wouldn't actually need to be modeled if the input data was simply the stimulus... 

> Is there any effect of the prior mean? For arguments sake, I think we should try and simulate the data without explicit knowledge of the prior mean...

Things that I'm not thinking of modeling (yet)
- Initial paddle (response metric) position
- reaction time
- feedback estimation error

