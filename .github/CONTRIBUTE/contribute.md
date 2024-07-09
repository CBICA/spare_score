---
name: Contribution guide
about: How to contribute
title: "[CONTRIBUTE] "
labels: ''
assignees: ''
---

### **How to contribute?**
* Open a pull request exaplaining your addition/fix. You can also see the [Bug report](../ISSUE_TEMPLATE/bug_report.md)

* Implementation is not everything. You have to add test cases for every function that you add at the [tests](../../tests/) folder.

* Last but not least, some documentation will be good for the users to learn how to use your code. You can add your documentation at the [docs](../../docs/) folder.

### **Good practices**
* We have automated test cases that are located at the [tests](../../tests/) folder. The workflow will automatically run the test cases on your PR, but, before creating a PR make sure your test cases run correctly localy using this command ```cd tests/unit && python -m unittest discover -s . -p "*.py"```

* Make sure that you add any new libraries that you may use at [dev-dependencies](../../dev-dependencies.txt) as well as updating the [setup.py](../../setup.py) file(if needed).

* Try to add docstrings on every new function that you implement as it will automatically be generated to documentation.Comments should look like this
    - For functions:
    ```python
        def your_function(param1: type, param2: type, ...) -> return_type:
            """
                Describe briefly what your function does

                Args:
                    param1(type): Describe what param1 is used for.
                    param2(type): Describe what param2 is used for.
                    etc.
                Return:
                    return_type: Explain what your function returns.
            """
    ```
    - For classes:
    ```python
        class your_class:
            """
                Describe briefly what your class does

                Static attributes:
                    param1(type): Explain what param1 is used for.
                    param2(type): Explain what param2 is used for.
                    etc.
                
                Methods:
                    func1(param1, param2, ...):
                        Here you have to explain what func1 does using the tutorial above.
                    You have to do the same for every function.
                
                Note that you can add more docstrings inside function if you wish.
            """
        def __init__(self, param1: type, param2: type, ...):
            ...

        def func1(self, param1: type, param2: type, ...) -> return_type: 
            ...
    ```

* As seen above, it is a good practice to always set parameter types in functions as well as return types. 

### **Useful files for contributions**

* You can find all the csv samples at the [data](../../spare_scores/data/) folder.
  
* You can find all the existing model weights at the [mdl](../../spare_scores/mdl/) folder.
 
* All the code that is used for the documentation is located at the [contents](../../docs/source/) folder and you can manually add more.