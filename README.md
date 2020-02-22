Our GradientDescent function can be found here: [GradientDescent](https://github.com/Alex-Lacy/CS499-Project-1/blob/master/GradientDescent.m)

However, it is easily accessed and used through our Driver function, [ROC](https://github.com/Alex-Lacy/CS499-Project-1/blob/master/ROC.m)

Our algorithm can run on a number of datasets at the same time.  In order to run the code on a particular dataset, follow these steps:

* Change the counter at the top to reflect how many datasets you're working with

* With each dataset in its own "if" block, do the following:
  * import a workspace (only if you have already run the "split" function)
  * load the dataset
  * initialize the dataset so that X is the observations, y is the labels, and m is the number of columns (see prebuilt examples)
  
* if you wish to use one of the prebuilt datasets:
  * please be sure you have the dataset in the correct directory (please see [Stanford's data reposity](https://web.stanford.edu/~hastie/ElemStatLearn/data.html) for actual data files)
  * Un-comment both lines with the splitdata function in [ROC](https://github.com/Alex-Lacy/CS499-Project-1/blob/master/ROC.m)
  * Make sure the number of datasets counter is set to 3 (there are 3 example datasets)

That's it! Run the code and watch the algorithm work!
