# Extraction of human preferences 👨→🤖

Developing safe and beneficial AI systems requires making them aware and aligned with human preferences. Since humans have significant control over the environment they operate in, we conjecture that RL agents implicitly learn human preferences.  Our research aim is to first show that these preferences exist in an agent and then extract these preferences. To start, we tackle this problem in a toy grid-like environment where a reinforcement learning (RL) agent is rewarded for collecting apples. Since it has been shown in previous work ([Wichers 2020](https://arxiv.org/abs/2002.06137)) that these implicit preferences exist and can be extracted, our first approach involved applying a variety of modern interpretability techniques to the RL agent trained in this environment to find meaningful portions of its network. We are currently pursuing methods to isolate a subnetwork within the trained RL agent which predicts human preferences.

## Report and results 📝 📉
Our project report has been published on LessWrong which includes a detailed overview of our intermediate results as well:  

## Running experiments 🧪
Please refer to the Colab notebooks and the readme in [/agent](https://github.com/arunraja-hub/Preference_Extraction/tree/master/agent) for details on the agents.

## Team 🧑‍🤝‍🧑
Arun Raja, Mislav Juric, Nevan Wichers, Riccardo Volpato

## Acknowledgements 🙏
We would like to thank Paul Christiano, Evan Hubinger and Christos Dimitrakakis for their research advice during AI Safety Camp 2020.




