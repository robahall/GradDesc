# GradDesc
Various Gradient Descent algorithms

Current goal is create random data sets and investigate 
how the different gradient descent algorithms speed up the process.

First notebook wrapping my head around plotting the error and the learning path is located in notebooks. 


First Batch:

*   Batch Gradient Descent - update after one epoch (run through all the data once and then update coefficents)
*   Stochastic Gradient Descent (SGD)- update coefficents after random sample group of total batch. 
*   Mini Batch Gradient Descent - update coefficients after small sample of total batch
*   Included visualization of learning rates

Project Structure:

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
│
├── data
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-rah-Gradient_Descent_Algorithms`.
│
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
```