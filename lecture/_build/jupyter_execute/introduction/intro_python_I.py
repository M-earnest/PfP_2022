# Introduction V - Introduction to Python - I



[Michael Ernst](https://github.com/M-earnest)  
Phd student - [Fiebach Lab](http://www.fiebachlab.org/), [Neurocognitive Psychology](https://www.psychologie.uni-frankfurt.de/49868684/Abteilungen) at [Goethe-University Frankfurt](https://www.goethe-university-frankfurt.de/en?locale=en)  

## Before we get started...
<br>

- most of what you’ll see within this lecture was prepared by Ross Markello, Michael Notter and Peer Herholz and further adapted for this course by Peer Herholz 
- based on Tal Yarkoni's ["Introduction to Python" lecture at Neurohackademy 2019](https://neurohackademy.org/course/introduction-to-python-2/)
- based on [IPython notebooks from J. R. Johansson](http://github.com/jrjohansson/scientific-python-lectures)
- based on http://www.stavros.io/tutorials/python/ & http://www.swaroopch.com/notes/python
- based on https://github.com/oesteban/biss2016 &  https://github.com/jvns/pandas-cookbook

[Peer Herholz (he/him)](https://peerherholz.github.io/)  
Research affiliate - [NeuroDataScience lab](https://neurodatascience.github.io/) at [MNI](https://www.mcgill.ca/neuro/)/[MIT](https://www.mit.edu/)  
Member - [BIDS](https://bids-specification.readthedocs.io/en/stable/), [ReproNim](https://www.repronim.org/), [Brainhack](https://brainhack.org/), [Neuromod](https://www.cneuromod.ca/), [OHBM SEA-SIG](https://ohbm-environment.org/), [UNIQUE](https://sites.google.com/view/unique-neuro-ai)  

<img align="left" src="https://raw.githubusercontent.com/G0RELLA/gorella_mwn/master/lecture/static/Twitter%20social%20icons%20-%20circle%20-%20blue.png" alt="logo" title="Twitter" width="32" height="20" /> <img align="left" src="https://raw.githubusercontent.com/G0RELLA/gorella_mwn/master/lecture/static/GitHub-Mark-120px-plus.png" alt="logo" title="Github" width="30" height="20" />   &nbsp;&nbsp;@peerherholz 

## Objectives 📍

* learn basic and efficient usage of the python programming language
    * what is python & how to utilize it
    * building blocks of & operations in python 

## What is Python?

* Python is a programming language
* Specifically, it's a **widely used/very flexible**, **high-level**, **general-purpose**, **dynamic** programming language
* That's a mouthful! Let's explore each of these points in more detail...

### Widely-used
* Python is the fastest-growing major programming language
* Top 3 overall (with JavaScript, Java)

<center><img src="https://149351115.v2.pressablecdn.com/wp-content/uploads/2017/09/growth_major_languages-1-1400x1200.png" width="800px" style="margin-bottom: 10px;"></center>

### High-level
Python features a high level of abstraction
* Many operations that are explicit in lower-level languages (e.g., C/C++) are implicit in Python
* E.g., memory allocation, garbage collection, etc.
* Python lets you write code faster

#### File reading in Java
```java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
 
public class ReadFile {
    public static void main(String[] args) throws IOException{
        String fileContents = readEntireFile("./foo.txt");
    }
 
    private static String readEntireFile(String filename) throws IOException {
        FileReader in = new FileReader(filename);
        StringBuilder contents = new StringBuilder();
        char[] buffer = new char[4096];
        int read = 0;
        do {
            contents.append(buffer, 0, read);
            read = in.read(buffer);
        } while (read >= 0);
        return contents.toString();
    }
}
```

#### File-reading in Python
```python
open(filename).read()
```

### General-purpose
You can do almost everything in Python
* Comprehensive standard library
* Enormous ecosystem of third-party packages
* Widely used in many areas of software development (web, dev-ops, data science, etc.)

<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/general_purpose_meme.png"></center>

### Dynamic
Code is interpreted at run-time
* No compilation process*; code is read line-by-line when executed
* Eliminates delays between development and execution
* The downside: poorer performance compared to compiled languages

<center><img src="https://imgs.xkcd.com/comics/python.png"></center>

(Try typing `import antigravity` into a new cell and running it!)

What we will do in this section of the course is a _short_ introduction to `Python` to help beginners to get familiar with this `programming language`.

It is divided into the following chapters:

- [Module](#Module)
- [Help and Descriptions](#Help-and-Descriptions)
- [Variables and types](#Variables-and-types)
    - [Symbol names](#Symbol-names)
    - [Assignment](#Assignment)
    - [Fundamental types](#Fundamental-types)
- [Operators and comparisons](#Operators-and-comparisons)
    - [Shortcut math operation and assignment](#Shortcut-math-operation-and-assignment)
- [Strings, List and dictionaries](#Strings,-List-and-dictionaries)
    - [Strings](#Strings)
    - [List](#List)
    - [Tuples](#Tuples)
    - [Dictionaries](#Dictionaries)
- [Indentation](#Indentation)
- [Control Flow](#Control-Flow)
    - [Conditional statements: `if`, `elif`, `else`](#Conditional-statements:-if,-elif,-else)
- [Loops](#Loops)
    - [`for` loops](#for-loops)
    - [`break`, `continue` and `pass`](#break,-continue-and-pass)
- [Functions](#Functions)
    - [Default argument and keyword arguments](#Default-argument-and-keyword-arguments)
    - [`*args` and `*kwargs` parameters](#*args-and-*kwargs-parameters)
    - [Unnamed functions: `lambda` function](#Unnamed-functions:-lambda-function)
- [Classes](#Classes)
- [Modules](#Modules)
- [Exceptions](#Exceptions)

Here's what we will focus on in the first block:

- [Module](#Module)
- [Help and Descriptions](#Help-and-Descriptions)
- [Variables and types](#Variables-and-types)
    - [Symbol names](#Symbol-names)
    - [Assignment](#Assignment)
    - [Fundamental types](#Fundamental-types)
- [Operators and comparisons](#Operators-and-comparisons)
    - [Shortcut math operation and assignment](#Shortcut-math-operation-and-assignment)
- [Strings, List and dictionaries](#Strings,-List-and-dictionaries)
    - [Strings](#Strings)
    - [List](#List)
    - [Tuples](#Tuples)
    - [Dictionaries](#Dictionaries)

## Modules

Most of the functionality in `Python` is provided by *modules*. To use a module in a Python program it first has to be imported. A module can be imported using the `import` statement. 


<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/python_import.png"></center>

For example, to import the module `math`, which contains many standard mathematical functions, we can do:

import math

This includes the whole module and makes it available for use later in the program. For example, we can do:

import math

x = math.cos(2 * math.pi)

print(x)

Importing the whole module us often times unnecessary and can lead to longer loading time or increase the memory consumption. An alternative to the previous method, we can also choose to import only a few selected functions from a module by explicitly listing which ones we want to import:

from math import cos, pi

x = cos(2 * pi)

print(x)

You can make use of `tab` again to get a list of `functions`/`classes`/etc. for a given `module`. Try it out via navigating the cursor behind the `import statement` and press `tab`:

from math import 

Comparably you can also use the `help` function to find out more about a given `module`:

import math
help(math)

It is also possible to give an imported module or symbol your own access name with the `as` additional:

import numpy as np
from math import pi as number_pi

x  = np.rad2deg(number_pi)

print(x)

You can basically provide any name (given it's following `python`/`coding` conventions) but focusing on intelligibility won't be the worst idea: 

import matplotlib as pineapple

pineapple.

##### Exercise 1.1

Import the `max` from `numpy` and find out what it does.

# write your solution in this code cell

from numpy import max
help(max)

##### Exercise 1.2

Import the `scipy` package and assign the access name `middle_earth` and check its `functions`.

# write your solution in this code cell

import scipy as middle_earth

help(middle_earth)

##### Exercise 1.3

What happens when we try to import a `module` that is either misspelled or doesn't exist in our `environment` or at all?

1. `python` provides us a hint that the `module` name might be misspelled
2. we'll get an `error` telling us that the `module` doesn't exist
3. `python` automatically searches for the `module` and if it exists downloads/installs it

import welovethiscourse

## Namespaces and imports
* Python is **very** serious about maintaining orderly `namespaces`
* If you want to use some code outside the current scope, you need to explicitly "`import`" it
* Python's import system often annoys beginners, but it substantially increases `code` clarity
    * Almost completely eliminates naming conflicts and confusion

## Help and Descriptions

Using the function `help` we can get a description of almost all functions. 

help(math.log)

math.log(10)

math.log(10, 2)

## Variables and data types
* in programming `variables` are things that store `values`
* in `Python`, we declare a `variable` by **assigning** it a `value` with the `=` sign
    * `name = value`
    * code `variables` **!=** math variables
        * in mathematics `=` refers to equality (statement of truth), e.g. `y = 10x + 2`
        * in coding `=` refers to assignments, e.g. `x = x + 1`
    * Variables are pointers, not data stores!
    
<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/python_data_types.png"></center>

* `Python` supports a variety of `data types` and `structures`:
    * `booleans`
    * `numbers` (`ints`, `floats`, etc.)
    * `strings`
    * `lists`
    * `dictionaries`
    * many others!
* We don't specify a variable's type at assignment

## Variables and types


### Symbol names 

Variable names in Python can contain alphanumerical characters `a-z`, `A-Z`, `0-9` and some special characters such as `_`. Normal variable names must start with a letter. 

By convention, variable names start with a lower-case letter, and Class names start with a capital letter. 

In addition, there are a number of Python keywords that cannot be used as variable names. These keywords are:

    and, as, assert, break, class, continue, def, del, elif, else, except, exec, finally, for, from, global, if, import, in, is, lambda, not, or, pass, print, raise, return, try, while, with, yield

### Assignment

(Not your homework assignment but the operator in `python`.)

The assignment operator in `Python` is `=`. `Python` is a `dynamically typed language`, so we do not need to specify the type of a `variable` when we create one.

`Assigning` a `value` to a new `variable` _creates_ the `variable`:

# variable assignment
x = 1.0

Again, this does not mean that `x` equals `1` but that the `variable` `x` has the `value` `1`. Thus, our `variable` `x` is _stored_ in the respective `namespace`:

x

This means that we can directly utilize the `value` of our `variable`: 

x + 3

Although not explicitly specified, a `variable` does have a `type` associated with it. The `type` is _derived_ from the `value` it was `assigned`.

type(x)

If we `assign` a new `value` to a `variable`, its `type` can change.

x = 1

type(x)

This outline one further _very important_ characteristic of `python` (and many other programming languages): `variables` can be directly overwritten by `assigning` them a new `value`. We don't get an error like "This `namespace` is already taken." Thus, always remember/keep track of what `namespaces` were already used to avoid unintentional deletions/errors (reproducibility/replicability much?). 

ring_bearer = 'Bilbo'
ring_bearer

ring_bearer = 'Frodo'
ring_bearer

If we try to use a variable that has not yet been defined we get an `NameError` (Note for later sessions, that we will use in the notebooks `try/except` blocks to handle the exception, so the notebook doesn't stop. The code below will try to execute `print` function and if the `NameError` occurs the error message will be printed. Otherwise, an error will be raised. You will learn more about exception handling later.):

try:
    print(Peer)
except(NameError) as err:
    print("NameError", err)
else:
    raise

Variable names:

* Can include `letters` (A-Z), `digits` (0-9), and `underscores` ( _ )
* Cannot start with a `digit`
* Are **case sensitive** (questions: where did "lower/upper case" originate?)

This means that, for example:

* `shire0` is a valid variable name, whereas `0shire` is not
* `shire` and `Shire` are different variables

##### Exercise 2.1

Create the following `variables` `n_elves`, `n_dwarfs`, `n_humans` with the respective values `3`, `7.0` and `nine`.

# write your solution here

n_elves = 3
n_dwarfs = 7.0
n_humans = "nine"

##### Exercise 2.2

What's the output of `n_elves + n_dwarfs`?

1. `n_elves + n_dwarfs`
2. 10
3. 10.0

n_elves + n_dwarfs

##### Exercise 2.3

Consider the following lines of code. 

`ring_bearer = 'Gollum'`  
`ring_bearer`  
`ring_bearer = 'Bilbo'`  
`ring_bearer`  

What is the final output?

1. `'Bilbo'`
2. `'Gollum'`
3. neither, the variable got deleted

ring_bearer = 'Gollum'
ring_bearer  
ring_bearer = 'Bilbo'
ring_bearer

### Fundamental types & data structures

* Most code requires more _complex structures_ built out of _basic data `types`_
* `data type` refers to the `value` that is `assigned` to a `variable` 
* `Python` provides built-in support for many common structures
    * Many additional structures can be found in the [collections](https://docs.python.org/3/library/collections.html) module

Most of the time you'll encounter the following `data types`

* `integers` (e.g. `1`, `42`, `180`)
* `floating-point numbers` (e.g. `1.0`, `42.42`, `180.90`)
* `strings` (e.g. `"Rivendell"`, `"Weathertop"`)
* `Boolean` (`True`, `False`)

If you're unsure about the `data type` of a given `variable`, you can always use the `type()` command.

#### Integers

Lets check out the different `data types` in more detail, starting with `integers`. `Intergers` are _natural numbers_ that can be _signed_ (e.g. `1`, `42`, `180`, `-1`, `-42`, `-180`).

x = 1
type(x)

n_nazgul = 9
type(n_nazgul)

remaining_rings = -1
type(remaining_rings)

#### Floating-point numbers

So what's the difference to `floating-point numbers`? `Floating-point numbers` are _decimal-point number_ that can be _signed_ (e.g. `1.0`, `42.42`, `180.90`, `-1.0`, `-42.42`, `-180.90`).

x_float = 1.0
type(x_float)

n_nazgul_float = 9.0
type(n_nazgul_float)

remaining_rings_float = -1.0
type(remaining_rings_float)

#### Strings

Next up: `strings`. `Strings` are basically `text elements`, from `letters` to `words` to `sentences` all can be/are `strings` in `python`. In order to define a `string`, `Python` needs **quotation marks**, more precisely `strings` start and end with quotation marks, e.g. `"Rivendell"`. You can choose between `"` and `'` as both will work (NB: `python` will put `'` around `strings` even if you specified `"`). However, it is recommended to decide on one and be consistent.

location = "Weathertop"
type(location)

abbreviation = 'LOTR'
type(abbreviation)

book_one = "The fellowship of the ring"
type(book_one)

#### Booleans

How about some `Boolean`s? At this point it gets a bit more "abstract". While there are many possible `numbers` and `strings`, a Boolean can only have one of two `values`: `True` or `False`. That is, a `Boolean` says something about whether something _is the case or not_. It's easier to understand with some examples. First try the `type()` function with a `Boolean` as an argument. 

b1 = True
type(b1)

b2 = False
type(b2)

lotr_is_awesome = True
type(lotr_is_awesome)

Interestingly, `True` and `False` also have `numeric values`! `True` has a value of `1` and `False` has a value of `0`. 

True + True

False + False

#### Converting data types

As mentioned before the `data type` is not set when `assigning` a `value` to a `variable` but determined based on its properties. Additionally, the `data type` of a given `value` can also be changed via set of functions.

- `int()` -> convert the `value` of a `variable` to an `integer`
- `float()` -> convert the `value` of a `variable` to a `floating-point number`
- `str()` -> convert the `value` of a `variable` to a `string`
- `bool()` -> convert the `value` of a `variable` to a `Boolean`  

int("4")

float(3)

str(2)

bool(1)

##### Exercise 3.1

Define the following `variables` with the respective `values` and `data types`: `fellowship_n_humans` with a `value` of two as a `float`, `fellowship_n_hobbits` with a `value` of four as a `string` and `fellowship_n_elves` with a value of one as an `integer`. 

# write your solution here

fellowship_n_humans = 2.0
fellowship_n_hobbits = 'four'
fellowship_n_elves = 1

##### Exercise 3.2

What outcome would you expect based on the following lines of code?

1. `True - False`
2. `type(True)`

1. `1`
2. `bool`

##### Exercise 3.3

Define two `variables`, `fellowship_n_dwarfs` with a `value` of one as a `string` and `fellowship_n_wizards` with a `value` of one as a `float`. Subsequently, change the `data type` of `fellowship_n_dwarfs` to `integer` and the `data type` of `fellowship_n_wizard` to `string`.    

fellowship_n_dwarfs = 1.0
fellowship_n_wizards = '1.0'

int(fellowship_n_dwarfs)

str(fellowship_n_wizards)

### Why do programming/science in Python?

Lets go through some advantages of the `python` programming language.  
<br/><br/>

<center><img src="https://funvizeo.com/media/memes/9114fb92b16ca1b8/java-python-think-why-waste-time-word-when-few-word-trick-meme-7a08727102156f3c-e9db4e91c4b2a7d5.jpg"></center>
<center><sup><sup><sup><sup> https://funvizeo.com/media/memes/9114fb92b16ca1b8/java-python-think-why-waste-time-word-when-few-word-trick-meme-7a08727102156f3c-e9db4e91c4b2a7d5.jpg </sup></sup></sup></sup></center>

### Easy to learn
* Readable, explicit syntax
* Most packages are very well documented
    * e.g., `scikit-learn`'s [documentation](http://scikit-learn.org/stable/documentation.html) is widely held up as a model
* A huge number of tutorials, guides, and other educational materials

### Comprehensive standard library
* The [Python standard library](https://docs.python.org/2/library/) contains a huge number of high-quality modules
* When in doubt, check the standard library first before you write your own tools!
* For example:
    * `os`: operating system tools
    * `re`: regular expressions
    * `collections`: useful data structures
    * `multiprocessing`: simple parallelization tools
    * `pickle`: serialization
    * `json`: reading and writing JSON

### Exceptional external libraries

* `Python` has very good (often best-in-class) external `packages` for almost everything
* Particularly important for "data science", which draws on a very broad toolkit
* Package management is easy (`conda`, `pip`)
* Examples:
    * Web development: flask, Django
    * Database ORMs: SQLAlchemy, Django ORM (w/ adapters for all major DBs)
    * Scraping/parsing text/markup: beautifulsoup, scrapy
    * Natural language processing (NLP): nltk, gensim, textblob
    * Numerical computation and data analysis: numpy, scipy, pandas, xarray, statsmodels, pingouin
    * Machine learning: scikit-learn, Tensorflow, keras
    * Image processing: pillow, scikit-image, OpenCV
    * audio processing: librosa, pyaudio
    * Plotting: matplotlib, seaborn, altair, ggplot, Bokeh
    * GUI development: pyQT, wxPython
    * Testing: py.test
    * Etc. etc. etc.

### (Relatively) good performance
* `Python` is a high-level dynamic language — this comes at a performance cost
* For many (not all!) use cases, performance is irrelevant most of the time
* In general, the less `Python` code you write yourself, the better your performance will be
    * Much of the standard library consists of `Python` interfaces to `C` functions
    * `Numpy`, `scikit-learn`, etc. all rely heavily on `C/C++` or `Fortran`

### Python vs. other data science languages

* `Python` competes for mind share with many other languages
* Most notably, `R`
* To a lesser extent, `Matlab`, `Mathematica`, `SAS`, `Julia`, `Java`, `Scala`, etc.

### R
* [R](https://www.r-project.org/) is dominant in traditional statistics and some fields of science
    * Has attracted many SAS, SPSS, and Stata users
* Exceptional statistics support; hundreds of best-in-class libraries
* Designed to make data analysis and visualization as easy as possible
* Slow
* Language quirks drive many experienced software developers crazy
* Less support for most things non-data-related

### MATLAB
* A proprietary numerical computing language widely used by engineers
* Good performance and very active development, but expensive
* Closed ecosystem, relatively few third-party libraries
    * There is an open-source port (Octave)
* Not suitable for use as a general-purpose language

### So, why Python?
Why choose Python over other languages?
* Arguably none of these offers the same combination of readability, flexibility, libraries, and performance
* Python is sometimes described as "the second best language for everything"
* Doesn't mean you should always use Python
    * Depends on your needs, community, etc.
    
<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/general_purpose_meme.png"></center>    

### You can have your cake _and_ eat it!
* Many languages—particularly R—now interface seamlessly with Python
* You can work primarily in Python, fall back on R when you need it (or vice versa)
* The best of all possible worlds?

### The core Python "data science" stack
* The Python ecosystem contains tens of thousands of packages
* Several are very widely used in data science applications:
    * [Jupyter](http://jupyter.org): interactive notebooks
    * [Numpy](http://numpy.org): numerical computing in Python
    * [pandas](http://pandas.pydata.org/): data structures for Python
    * [Scipy](http://scipy.org): scientific Python tools
    * [Matplotlib](http://matplotlib.org): plotting in Python
    * [scikit-learn](http://scikit-learn.org): machine learning in Python
* We'll cover the first three very briefly here
    * Other tutorials will go into greater detail on most of the others

### The core "Python for psychology" stack
* The `Python ecosystem` contains tens of thousands of `packages`
* Several are very widely used in psychology research:
    * [Jupyter](http://jupyter.org): interactive notebooks
    * [Numpy](http://numpy.org): numerical computing in `Python`
    * [pandas](http://pandas.pydata.org/): data structures for `Python`
    * [Scipy](http://scipy.org): scientific `Python` tools
    * [Matplotlib](http://matplotlib.org): plotting in `Python`
    * [seaborn](https://seaborn.pydata.org/index.html): plotting in `Python`
    * [scikit-learn](http://scikit-learn.org): machine learning in `Python`
    * [statsmodels](https://www.statsmodels.org/stable/index.html): statistical analyses in `Python`
    * [pingouin](https://pingouin-stats.org/): statistical analyses in `Python`
    * [psychopy](https://www.psychopy.org/): running experiments in `Python`
    * [nilearn](https://nilearn.github.io/stable/index.html): brain imaging analyses in `Python``
    * [mne](https://mne.tools/stable/index.html): electrophysiology analyses in `Python` 
* Execept `scikit-learn`, `nilearn` and `mne`, we'll cover all very briefly in this course
    * there are many free tutorials online that will go into greater detail and also cover the other `packages`

## Homework assignment #3

Your third homework assignment will entail working through a few tasks covering the contents discussed in this session within of a `jupyter notebook`. You can download it #################adapt link ############### [here](https://www.dropbox.com/s/bafzg60rxcxwwzg/PFP_assignment_3_intro_python_1.ipynb?dl=1). In order to open it, put the `homework assignment notebook` within the folder you stored the `course materials`, start a `jupyter notebook` as during the sessions, navigate to the `homework assignment notebook`, open it and have fun!  

**Deadline: 17/11/2021, 11:59 PM EST**