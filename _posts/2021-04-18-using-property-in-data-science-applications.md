---
layout: post
title: "Using @property in Data Science Applications"
date: 2021-04-18
desc: "Using the Python decorator @property for classes in Python"
keywords: "DataScience,python"
categories: [data,python]
tags: [python,decorators,classes]
icon: icon-python
---

`@property` is a decorator in Python that is not mentioned in learning materials. I was surprised when I came across it and it took me a while to understand how and when to use it. 

In this notebook, I'll demonstrate the basic usage and a sample case for a Data Science settings. I'll conclude with some pointers on where else it can be useful.

## What is a decorator?

Decorators are functions that operate on functions. They are a neat shorthand for wrapping a function with another function such as: `decorator_func(decorated_func)`. They are commonly used to either operate on output of functions, measure the time, or cache some value. There are also [class decorators](https://realpython.com/primer-on-python-decorators/#decorating-classes) that can change the behaviour of every method in a class.

```python
# when the @ operator is added on top of a function definition
# Python wraps the defined function with the one refered with @
decorator_func(decorated_func)
# is the same as
@decorator_func
def decorated_func():
    pass
```

## What is @property?
`@property` is a decorator for Python class attributes that enables setting special behavior when the value of the attribute is changed. Those special behaviors can be almost anything but common examples are:
* Checking if the type of the new value is valid
* Making sure that the new value is consistent with other attributes
* Applying a transformation if new value if needed

## How to use @property?
```python
class DummyClass:
    def __init__(self):
        self._text = 'this is a dummy'
    
    # by declaring a class method as property
    # Python will now enable accessing this property as
    # DummyClass().text
    @property
    def text(self):
        return self._text
 
    # By using @text.setter, the special behavior is 
    # enabled when changing the value of this attribute
    # as DummyClass().text = 'new_value'
    @text.setter
    def text(self, new_text):
        if not isinstance(new_text, str):
            raise TypeError('text must be a string')
        self._text = new_text
```


```python
# when we try to change the value of the class
# the setter will come into effect and it'll raise
# the TypeError as shown above
dummy = DummyClass()
dummy.text = 5
```

    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-3-e0742d3b980a> in <module>()
          3 # the TypeError as shown above
          4 dummy = DummyClass()
    ----> 5 dummy.text = 5
    

    <ipython-input-2-72e6b56c07f1> in text(self, new_text)
         16     def text(self, new_text):
         17         if not isinstance(new_text, str):
    ---> 18             raise TypeError('text must be a string')
         19         self._text = new_text


    TypeError: text must be a string


In the case above, `@property` allows us to check the type of the value of the attribute before we can set it.

## The Case
Think of a case where you're working for a company that operates a service that provides house price estimates to customers. Recently, the company has acquired an enriched dataset to provide better estimates to a portion of the customers. However, this new dataset works only for a portion of the area that the services cover. To make things more complicated, the backend of the service was written in a way that it expects only one model to work with. The engineering team asked for one object that can work with either dataset.

These kinds of scenarios and constraints pop up frequently in the daily life of a data scientist. To illustrate a solution to this I'll use the Boston housing dataset from scikit-learn. Fit a LinearRegression model to the data and package the solution in a single class that uses the `@property` decorator to handle the two datasets mentioned.

## Prepare the Data

```python
import numpy as np
from sklearn import datasets, linear_model
X, y = datasets.load_boston(return_X_y=True)
```

The first dataset will have columns 3-5 missing, and the second one will have the columns 9-13 missing. We'll also split the sample into two different buckets. For this example, we'll assume that two data sources can be identified from their number of features.

```python
n_rows = X.shape[0]
n_sub = n_rows // 2
```

```python
# create data for the first model
X_1 = np.c_[X[:n_sub, :2], X[:n_sub, 5:]]
y_1 = y[:n_sub]
X_1.shape
```

    (253, 10)

```python
# create data for the second model
X_2 = X[n_sub:, :9]
y_2 = y[n_sub:]
X_2.shape
```

    (253, 9)


## The Model


```python
model_source_a = linear_model.LinearRegression()
model_source_a.fit(X_1, y_1)
```

```python
model_source_b = linear_model.LinearRegression()
model_source_b.fit(X_2, y_2)
```

# The Class

The trained models can now be saved in a class. The class will contain both models and return predictions using the correct one based on the data source.


```python
class MultiEstimator:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        self.active_model = model_a
        self._data = None
    
    def predict(self):
        return self.active_model.predict(self._data)
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, X):
        # set the correct model based on
        # the data source
        if X.shape[1] == 10:
            self.active_model = self.model_a
        elif X.shape[1]  == 9:
            self.active_model  = self.model_b
        # set the data
        self._data = X
```


```python
est = MultiEstimator(model_source_a, model_source_b)
sample_1 = X_1[np.random.choice(n_sub, 10)]
est.data = sample_1
est.predict()
```
    array([33.65265532, 26.27365789, 13.39783497, 41.62090646, 31.41320706,
           22.37489665, 21.32433741, 22.81443648, 26.57501182, 20.77409669])

```python
sample_2 = X_2[np.random.choice(n_sub, 10)]
est.data = sample_2
est.predict()
```
    array([26.78951923, 20.86852426, 13.43668696, 21.15623019, 13.98538914,
           14.84914382, 22.77190894, 31.98879291, 20.98384074, 16.24467224])

This shows that we have created a file and it is now ready to be shared with the engineering team.

## Conclusion

In this post, I have shown what the decorators are, what does the `@property` decorator do in classes and a sample Data Science use case to give an idea to start with. I also really like to use these classes in model configurations where attributes such as regularization parameters might depend on solvers used at that point (e.g. LogisticRegression in scikit-learn). 

Thanks for reading thus far and let me know what you think in the comments below.
