# A CNN-based Model to Choose AI Planner

Automated Planning is a field within Artificial Intelligence that deals with composing a plan based on the problem description. This involves searching in a very large space, using a heuristic function. As finding a plan is known to be PSPACE-Hard, no single planner can be expected to work well for all tasks and domains. One approach to efficiently solve various planning problems is to aggregate multiple planners in a Portfolio. In this work, we establish a CNN-based model to act as a portfolio and test its performance which turns out to be have advantages over other approaches.

The class PlannerPortfolioDataset is implemented in pp_dataset.py file.
The models' architectures can be found in architectures.py file.
A simple Jupiter Notebook that demonstartes how we process the data is also attached.

To train a model simply use the bash script in test.sh or run it directly through the terminal. e.g. "python3 main.py net_1 0 -optimizer Adam".



## Contact

Please direct questions to Ronen Nir, ronen.nx@gmail.com.
