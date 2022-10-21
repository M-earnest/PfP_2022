# Introduction VI - Introduction to Python II

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
    * building blocks of & operations in python 
    * `operators` & `comparisons`
    * `strings`, `lists`, `tuples` & `dictionaries` 

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

## Recap of the last session

Before we dive into new endeavors, it might be important to briefly recap the things we've talked about so far. Specifically, we will do this to evaluate if everyone's roughly on the same page. Thus, if some of the aspects within the recap are either new or fuzzy to you, please have a quick look at the respective part of the [last session](https://peerherholz.github.io/Python_for_Psychologists_Winter2021/introduction/intro_python_I.html) again and as usual: ask questions wherever something is not clear.

## What is Python?

* Python is a programming language
* Specifically, it's a **widely used/very flexible**, **high-level**, **general-purpose**, **dynamic** programming language
* That's a mouthful! Let's explore each of these points in more detail...

## Module

Most of the functionality in Python is provided by *modules*. To use a module in a Python program it first has to be imported. A module can be imported using the `import` statement. 


<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/python_import.png"></center>

Assuming you want to import the entire `pandas` module to do some `data exploration`, `wrangling` and `statistics`, how would you do that?

# Pleae write your solution in this cell

import pandas

As this might be a bit hard to navigate, specifically for `finding`/`referencing` `functions`. Thus, it might be a good idea to provide a respective `access name`. For example, could you show how you would provide the `pandas` module the `access name` `pd`?

# Please write your solution in this cell

During your analyzes you recognize that some of the analyses you want to run require functions from the `statistics` `module` `pingouin`. Is there a way to only import the `functions` you want from this `module`, e.g. the `wilcoxon` test from within `pingouin.nonparametric`?

# Please write your solution in this cell

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

### Assignment

(Not your homework assignment but the operator in `python`.)

The assignment operator in `Python` is `=`. `Python` is a `dynamically typed language`, so we do not need to specify the type of a `variable` when we create one.

`Assigning` a `value` to a new `variable` _creates_ the `variable`:

Within your analyzes you need to create a `variable` called `n_students` and assign it the `value` `21`, how would that work?

# Please write your solution in this cell

n_students = 21

Quickly after you realize that the value should actually be `20`. What options do you have to change the `value` of this `variable`?

# Please write your solution in this cell

n_students = 20
n_students = n_students - 1

During the analyzes you noticed that the `data type` of `n_students` changed. How can you find out the `data type`? 

# Please write your solution in this cell

type(n_students)

Is there a way to change the `data type` of `n_students` to something else, e.g. `float`?

# Please write your solution in this cell

float(n_students)

Along the way you want to create another `variable`, this time called `acquisition_time` and the `value` `December`. How would we do that and what `data type` would that be? 

from pingouin.nonparametric import wilcoxon

import pandas as pd

# Please write your solution here

acquisition_time = "December"
type(acquisition_time)

As a final step you want to create two `variables` that indicate that the outcome of a statistical test is either `significant` or not. How would you do that for the following example: for `outcome_anova` it's _true_ that the result was `significant` and for `outcome_ancova` it's _false_ that the result was `significant`?   

# Please write your solution in this cell 

outcome_anova = True
outcome_ancova = False

Alright, thanks for taking the time to go through this `recap`. Again: if you could solve/answer all questions, you should have the information/knowledge needed for this session. 

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

## Operators and comparisons

One of the most basic utilizations of `python` might be simple `arithmetic operations` and `comparisons`. `operators` and `comparisons` in `python` work as one would expect:

* `Arithmetic operators` available in `python`: `+`, `-`, `*`, `/`, `**` power, `%` modulo
* `comparisons` available in `python`: `<`, `>`, `>=` (greater or equal), `<=` (less or equal), `==` (equal), `!=` (not equal) and `is` (identical)

Obviously, these `operators` and `comparisons` can be used within tremendously complex analyzes and actually build their basis.

Lets check them out further via a few quick examples, starting with `operators`:

[1 + 2, 
 1 - 2,
 1 * 2,
 1 / 2,
 1 ** 2,
 1 % 2]

In `Python 2.7`, what kind of `division` (`/`) will be executed, _depends_ on the type of the numbers involved. If all numbers are `integers`, the `division` will be an `integer division`, otherwise, it will be a `float division`. In `Python 3` this has been changed and fractions aren't lost when `dividing` `integers` (for `integer` `division` you can use another operator, `//`). In `Python 3` the following two `operations` will give the same result (in `Python 2` the first one will be treated as an `integer division`). It's thus important to remember that the `data type` of  `division outcomes` will always be `float`. 

print(1 / 2)
print(1 / 2.0)

`Python` also respects `arithemic rules`, like the sequence of `+`/`-` and `*`/`/`.  

1 + 2/4

1 + 2 + 3/4

The same holds true for `()` and `operators`:

(1 + 2)/4

(1 + 2 + 3)/4

Thus, always watch out for how you define `arithmetic operations`!

Just as a reminder: the `power operator` in `python` is `**` and not `^`:

2 ** 2

This `arithmetic operations` also show some "handy" properties in combination with `assignments`, specifically you can apply these `operations` and modify the `value` of a given `variable` "in-place". This means that you don't have to `assign` a given `variable` a new `value` via an additional line like so:

a = 2
a = a * 2
print(a)

but you can `shortcut` the command `a = a * 2` to `a *= 2`. This also works with other `operators`: `+=`, `-=` and `/=`.

b = 3
b *= 3
print(b)

Interestingly, we meet `booleans` again. This time in the form of `operators`. So `booleans` can not only be referred to as a `data type` but also `operators`. Whereas the `data type` entails the `values` `True` and `False`, the `operators` are spelled out as the words `and`, `not`, `or`. They therefore allow us to evaluate if 

- something **`and`** something else is the case 
- something is **`not`** the case
- something **`or`** something else is the case

How about we check this on an example, i.e. the `significance` of our test results from the `recap`: 

outcome_anova = True

outcome_ancova = False

outcome_anova and outcome_ancova

not outcome_ancova

outcome_anova or outcome_ancova

While the "classic" `operators` appear to be rather simple and the "boolean" `operators` rather abstract, a sufficient understanding of both is very important to efficiently utilize the `python` programming language. However, don't worry: we'll use them throughout the entire course going forward to gain further experience. 

After spending a look at `operators`, it's time to check out `comparisons` in more detail. Again, most of them might seem familiar and work as you would expect. Here's the list again:

* `Comparisons` in `python` `>`, `<`, `>=` (greater or equal), `<=` (less or equal), `==` (equal), `!=` (not equal) and `is` (identical)

The first four are the "classics" and something you might remember from your `math classes` in high school. Nevertheless, it's worth to check how they exactly work in `python`. 

If we compare `numerical values`, we obtain `booleans` that indicate if the `comparisons` is `True` or `False`. Lets start with the "classics". 

2 > 1, 2 < 1

2 > 2, 2 < 2

2 >= 2, 2 <= 2

So far so good and no major surprises. Now lets have a look at those `comparisons` that might be less familiar. At first, `==`. You might think: "What, a double `assignment`?" but actually `==` is the `equal` `comparison` and thus `compares` two `variables`, `numbers`, etc., evaluating if they are `equal` to each other.    

1 == 1

outcome_anova == outcome_ancova

'This course' == "cool"

1 == 1 == 2

One interesting thing to mention here is that `equal` `values` of different `data types`, i.e. `integers` and `floats`, are still evaluated as equal by `==`:

1 == 1.0

Contrarily to evaluating if two or more things are `equal` via `==`, we can utilize `!=` to evaluate if two are more things are `not equal`. The behavior of these `comparison` concerning the `outcome` is however identical: we get `booleans`. 

2 != 3

outcome_anova = True 
outcome_ancova = False
outcome_anova != outcome_ancova 

1 != 1 != 2

There's actually one very specific `comparison` that only works for one `data type`: `string comparison`. The `string comparison` is reflected by the word `in` and evaluates if a `string` **`is`** part of another `string`. For example, you can evaluate if a `word` or certain `string pattern` **`is`** part of another `string`. Two fantastic beings are going to help showcasing this! 

Please welcome, the [Wombat](http://en.wikipedia.org/wiki/Wombat) & the [Capybara](http://en.wikipedia.org/wiki/Capybara).
<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/wombat_capybara.png" width=500></center>

"cool" in "Wombats are cool"

"ras ar" in "Wombats and capybaras are cool"

The `string comparison` can also be combined with the `boolean` operator to evaluate if a `string` or `string pattern` **`is not`** part of another `string`.

"stupid" not in "Wombats and capybaras"

Before we finish the `operators` & `comparison` part, it's important to outline one important aspects that you've actually already seen here and there but was never mentioned/explained in detail: `operators` & `comparisons` work directly on `variables`, that is their `values`. For example, if we want to change the number of a `variable` called `n_lectures` from `5` to `6`, we can simply run: 

n_lectures = 5
n_lectures = n_lectures + 1 
n_lectures

or use the `shortcut` as seen before

n_lectures = 5
n_lectures += 1
n_lectures

This works with other `types` and `operators`/`comparisons` too, for example `strings` and `==`: 

'Wombats' == 'Capybaras'

##### Exercise 4.1 

You want to compute the `mean` of the following `reaction times`: `1.2`, `1.0`, `1.5`, `1.9`, `1.3`, `1.2`, `1.7`. Is there a way to achieve that using `operators`?  

# Please write your solution here

(1.2 + 1.0 + 1.5 + 1.9 + 1.3 + 1.2 + 1.7)/7

Spoiler: there are of course many existing `functions` for all sorts of equations and statistics so you don't have to write it yourself every time. For example, we could also compute the `mean` using `numpy`'s `mean` `function`:  

import numpy as np
np.mean([1.2, 1.0, 1.5, 1.9, 1.3, 1.2, 1.7]) 

##### Exercise 4.2

Having computed the `mean`, you need to compare it to a reference `value`. The latter can be _possibly_ found in a `string` that entails data from a previous analyses. If that's the case, the `string` should contain the words `"mean reaction time"`. Is there a way to evaluate this?

The `string` would be: "In contrast to the majority of the prior studies the mean reaction time of the here described analyses was `1.2`."

# Please write your solution here

reference = "In contrast to the majority of the prior studies the mean reaction time of the here described analyses was `1.2`."
"mean reaction time" in reference

###### Exercise 4.3 

Having found the reference `value`, that is `1.2` we can compare it to our `mean`. Specifically, you want to know if the `mean` is `less or equal` than `1.2`. The outcome of the `comparison` should then be the `value` of a new `variable` called `mean_reference_comp`. 

# Please write your solution here

mean = (1.2 + 1.0 + 1.5 + 1.9 + 1.3 + 1.2 + 1.7)/7
mean_reference_comp = mean <= 1.2
mean_reference_comp

**Fantastic work folks, really really great! Time for a quick party!**

</br>

<center><img src="https://media4.giphy.com/media/kyLYXonQYYfwYDIeZl/giphy.gif?cid=ecf05e47vl8lju7wj80ak9p5azuq5imkcbv64d2jple35roj&rid=giphy.gif&ct=g" width="300"></center>

<center><a href="https://giphy.com/gifs/sesamestreet-sesame-street-50th-anniversary-kyLYXonQYYfwYDIeZl">via GIPHY</a></center>

Having already checked out `modules`, `help & descriptions`, `variables and data types`, `operators and comparisons`, we will continue with the final section of the first block of our `python` introduction. More precisely, we will advance to new, more complex `data types` and `structures`: `strings`, `lists`, `tuples` and `dictionaries`.

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

## Strings, List and dictionaries

So far, we've explored `integers`, `float`, `strings` and `boolean` as `fundamental types`. However, there are a few more that are equally important within the `python` programing language and allow you to easily achieve complex behavoir and ease up your everyday programming life: `strings`, `lists`, `tuples` and `dictionaries`.

</br>
<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/python_data_types_2.png"></center>

### Strings

Wait, what? Why are we talking about strings again? Well, actually, `strings` are more than a "just" fundamental type. There are quite a few things you can do with `strings` that we haven't talked about yet. However, first things first: `strings` contain text:


statement = "The wombat and the capybara are equally cute. However, while the wombat lives in Australia, the capybara can be found in south america."
statement

type(statement)

So, what else can we do? For example, we can get the `length` of the `string` which reflects the number of `characters` in the `string`. The respective `len` function is one of the `python functions` that's always available to you, even without `importing` it. Notably, `len` can `operate` on various `data types` which we will explore later.

len(statement)

The `string` `data types` also allows us to `replace` parts of it, i.e. `substrings`, with a different `string`. The respective syntax is `string.replace("substring_to_replace", "replacement_string")`, that is, `.replace` searches for `"substring_to_replace"` and `replaces` it with `"replacement_string"`. If we for example want to state that `wombats` and `capybaras` are `awesome` instead of `cute`, we could do the following:

statement.replace("cute", "awesome")

Importantly, `strings` are not `replaced` in-place but require a new `variable assignment`.

statement

statement_2 = statement.replace("cute", "awesome")
statement_2

We can also `index` a `string` using `string[x]` to get the `character` at the specified `index`:

statement_2

statement_2[1]

Pump the breaks right there: why do we get `h` when we specify `1` as `index`? Shouldn't this get us the `first index` and thus `T`? 

<center><img src="https://media1.giphy.com/media/BXOEmFSzNkOObZhIA3/giphy.gif?cid=ecf05e47iuhbrrici7z2tn33dxeu0xxi96i19uk7qphiojb3&rid=giphy.gif&ct=g" width="300"></center>

<center><a href="https://giphy.com/gifs/theoffice-the-office-tv-secret-santa-BXOEmFSzNkOObZhIA3">via GIPHY</a></center>

**HEADS UP EVERYONE: INDEXING IN `PYTHON` STARTS AT 0**

</br>
<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/index_string.png"></center>

This means that the `first index` is `0`, the `second index` `1`, the `third index` 2, etc. . This holds true independent of the `data type` and is one of the major confusions when folks start programming in `python`, so always watch out!

statement_2

statement_2[0]

statement_2[1]

statement_2[2]

If we want to get more than one `character` of a `string` we can use the following syntax `string[start:stop]` which extracts characters between `index` `start` and `stop`. This technique is called *slicing*.

statement_2[4:10]

If we omit either (or both) of `start` or `stop` from `[start:stop]`, the default is the beginning and the end of the `string`, respectively:

statement_2[:10]

statement_2[10:]

statement_2[:]

We can also define the `step size` using the syntax `[start:end:step]` (the default value for `step` is `1`, as we saw above):

statement_2[::1]

statement_2[::2]

#### String formatting 

Besides `operating` on `strings` we can also apply different `formatting styles`. More precisely, this refers to different ways of displaying `strings`. The main `function` we'll explore regarding this will be the `print` function. Comparable to `len`, it's one of the `python` `functions` that's always available to you, even without `import`. 

For example, if we `print` `strings` added with `+`, they are concatenated without space: 

print("The" + "results" + "were" + "significant")

The `print` `function` concatenates `strings` differently, depending how the `inputs` are specified. If we just provide all `strings` without anything else, they will be concatenated without spaces:

print("The" "results" "were" "significant")

If we provide `strings` separated by `,`, they will be concatenated with spaces:

print("The", "results", "were", "significant")

Interestingly, the `print` `function` converts all `inputs` to `strings`, no matter their actual `type`:

print("The", "results", "were", "significant", 0.049, False)

Another very cool and handy option that we can specify `placeholders` which will be filled with an `input` according to a given `formatting style`. `Python` has two `string formatting styles`. An example of the old style is below, the `placeholder` or `specifier` `%.3f` transforms the `input` `number` into a `string`, that corresponds to a `floating point number` with `3 decimal places` and the `specifier` `%d` transforms the `input` `number` into a `string`, corresponding to a `decimal number`.

print("The results were significant at %.3f" %(0.049))

print("The results were significant at %d" %(0.049))

As you can see, you have to be very careful with `string formatting` as important information might otherwise get lost!  

We can achieve the same outcome using the new style string formatting which uses `{}` followed by `.format()`.

print("The results were significant at {:.3f}" .format(0.049))

If you would like to include `line-breaks` and/or `tabs` in your `strings`, you can use `\n` and `\t` respectively: 

print("Geez, there are some many things \nPython can do with \t strings.")

We can of course also combine the different `string formatting` options:

print("Animal: {}\nHabitat: {}\nRating: {}".format("Wombat", "Australia", 5))





#### Single Quote
You can specify strings using single quotes such as `'Quote me on this'`.
All white space i.e. spaces and tabs, within the quotes, are preserved as-is.

#### Double Quotes
Strings in double quotes work exactly the same way as strings in single quotes. An example is `"What's your name?"`.

#### Triple Quotes

You can specify multi-line strings using triple quotes - (`"""` or `'''`). You can use single quotes and double quotes freely within the triple quotes. An example is:

'''I'm the first line. Check how line-breaks are shown in the second line.
Do you see the line-break?
"What's going on here?," you might ask.
Well, "that's just how tiple quotes work."
'''

##### Exercise 5.1

Create two `variable`s called `info_wombat` and `info_capybara` and provide them the following `values` respectively:

"The wombat is quadrupedal marsupial and can weigh up to 35 kg."

"The capybara is the largest rodent on earth. Its relatives include the guinea pig and the chinchilla." 

Once created, please verify that the `type` is `string`.

# Please write your solution here

info_wombat = "The wombat is quadrupedal marsupial and can weigh up to 35 kg."
info_capybara = "The capybara is the largest rodent on earth. Its relatives include the guinea pig and the chinchilla."

print(type(info_wombat))
print(type(info_capybara))

##### Exercise 5.2

Compute the `length` and `print` within the `strings` "The wombat information has [insert length here] characters." and "The capybara information has [insert length here] characters." After that, please compare the length of the `strings` and `print` if they are equal.

# Please write your solution here

print("The wombat information has %s characters." %len(info_wombat))
print("The capybara information has %s characters." %len(info_capybara))
print("The information has an equal amount of characters: %s" %str(len(info_wombat)==len(info_capybara)))

##### Exercise 5.3

Get the following `indices` from the `info_wombat` and `info_capybara` respectively: `4`-`10` and `4`-`12`. Replace the resulting word in `info_wombat` with `capybara` and the resulting word in `info_capybara` with `wombat`.

print(info_wombat[4:10])
print(info_capybara[4:12])

print(info_wombat.replace('wombat', 'capybara'))
print(info_capybara.replace('capybara', 'wombat'))

### List

Next up: `list`s. In general, `list`s are very similar to `strings`. One crucial difference is that `list` elements (things in the `list`) can be of any `type`: `integers`, `floats`, `strings`, etc. . Additionally, `types` can be freely mixed within a `list`, that is, each `element` of a list can be of a different `type`. `List`s are among the `data types` and `structures` you'll work with almost every time you do something in `python`. They are super handy and comparably to `strings`, have a lot of "in-built" functionality.


</br>
<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/python_data_types_2.png"></center>

The basic `syntax` for creating `list`s in `python` is `[...]`:

[1,2,3,4]

type([1,2,3,4])

You can of course also set `list`s as the `value` of a `variable`. For example, we can create a `list` with our `reaction times` from before:

reaction_times = [1.2, 1.0, 1.5, 1.9, 1.3, 1.2, 1.7]
print(type(reaction_times))
print(reaction_times)

Going back to the comparison with `strings`, we can use the same `index` and `slicing` techniques to manipulate `list`s as we could use on `strings`: `list[index]`, `list[start:stop]`.

print(reaction_times)
print(reaction_times[1:3])
print(reaction_times[::2])

<center><img src="https://media1.giphy.com/media/BXOEmFSzNkOObZhIA3/giphy.gif?cid=ecf05e47iuhbrrici7z2tn33dxeu0xxi96i19uk7qphiojb3&rid=giphy.gif&ct=g" width="300"></center>

<center><a href="https://giphy.com/gifs/theoffice-the-office-tv-secret-santa-BXOEmFSzNkOObZhIA3">via GIPHY</a></center>

**HEADS UP EVERYONE: INDEXING IN `PYTHON` STARTS AT 0**

</br>
<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/index_list.png"></center>

This means that the `first index` is `0`, the `second index` `1`, the `third index` 2, etc. . This holds true independent of the `data type` and is one of the major confusions when folks start programming in `python`, so always watch out!

Thus, to get the first index of our `reaction_times`, we have to do the following:

reaction_times[0]

There's another important aspect related to `index` and `slicing`. Have a look at the following example that _should_ get us the `reaction times` from index `1` to `4`: 

print(reaction_times)
print(reaction_times[1:4])

Isn't there something missing, specifically the last `index` we wanted to grasp, i.e. `4`?

<center><img src="https://media1.giphy.com/media/BXOEmFSzNkOObZhIA3/giphy.gif?cid=ecf05e47iuhbrrici7z2tn33dxeu0xxi96i19uk7qphiojb3&rid=giphy.gif&ct=g" width="300"></center>

<center><a href="https://giphy.com/gifs/theoffice-the-office-tv-secret-santa-BXOEmFSzNkOObZhIA3">via GIPHY</a></center>

**HEADS UP EVERYONE: SLICING IN `PYTHON` EXCLUDES THE "STOP" INDEX**

</br>
<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/index_list_slicing.png"></center>

This means that the `slicing` technique gives you everything up to the `stop` `index` but does not include the `stop` `index` itself. For example, `reaction_times[1:4]` will return the `list elements` from index `1` up to `4` but not the fourth `index`. This holds true independent of the `data type` and is one of the major confusions when folks start programming in `python`, so always watch out!

So, to get to `list elements` from `index` `0` - `4`, including `4`, we have to do the following:

print(reaction_times)
print(reaction_times[0:5])

As mentioned before, `elements` in a `list` do not all have to be of the same `type`:

mixed_list = [1, 'a', 4.0, 'What is happening?']

print(mixed_list)

Another nice thing to know: `python` `list`s can be `inhomogeneous` and `arbitrarily nested`, meaning that we can define and access `list`s within `list`s. Is this `list`-ception??

nested_list = [1, [2, [3, [4, [5]]]]]

print("our nest list looks like this: %s" %nested_list)
print("the length of our nested list is: %s" %len(nested_list))
print("the first index of our nested_list is: %s" %nested_list[0])
print("the second index of our nested_list is: %s" %nested_list[1])
print("the second index of our nested_list is a list we can index again via nested_list[1][1]: %s" %nested_list[1][1])

Lets have a look at another `list`. Assuming you obtain data describing the favorite `movies` and `snacks` of a sample population, we can put the respective responses in `list`s for easy handling:

movies = ['The_Intouchables', 'James Bond', 'Forrest Gump', 'Retired Extremely Dangerous', 'The Imitation Game',
          'The Philosophers', 'Call Me by Your Name', 'Shutter Island', 'Love actually', 'The Great Gatsby',
          'Interstellar', 'Inception', 'Lord of the Rings - The Two Towers', 'Fight Club', 'Shutter Island',
          'Harry Potter', 'Harry Potter and the MatLab-Prince', 'Shindlers List', 'Inception']

BTW: thanks so much for these, I really enjoyed going through the `notebooks`. However, `Harry Potter and the MatLab-Prince`....that one really got me. I mean there was even a short summary of the story...

snacks = ['pancakes', 'Banana', 'dominoes', 'carrots', 'hummus', 'chocolate', 'chocolate', 'Pringles', 'snickers',
          'chocolate','Kinder bueno', 'sushi',  'mint chocolate', 'fruit', 'dried mango', 'dark chocolate', 
          'too complicated', 'snickers', 'Rocher']

A smaller subsample also provided their favorite animal. (If you also provided one but it doesn't show up it means that you most likely used the "add attachment" option to add images or a way to refers to local files but doesn't embed them directly in the `notebook`. Unfortunately, things can be a bit strange there...so don't worry, we address this in subsequent sessions.)  

animals = ['cat', 'lizard', 'coral', 'elephant', 'barred owl', 'groundhog']

So lets check what we can do with these lists. At first, here they are again:

print('The favorite movies were: %s' %movies)
print('\n')
print('The favorite snacks were: %s'%snacks)
print('\n')
print('The favorite animals were: %s' %animals)

Initially we might want to `count` how many responses there were. We can achieve this via our old friend the `len` `function`. If we want to also check if we got responses from all 19 participants of our sample population, we can directly use `comparisons`.  

print('Regarding movies there were %s responses' %len(movies))
print('We got responses from all participants: %s' %str(len(movies)==19))

We can do the same for `snacks` and `animals`:

print('Regarding snacks there were %s responses' %len(snacks))
print('We got responses from all participants: %s' %str(len(snacks)==19))

print('Regarding animals there were %s responses' %len(animals))
print('We got responses from all participants: %s' %str(len(animals)==19))

Another thing we might want to check is the number of `unique` responses, that is if some `values` in our `list` appear multiple times and we might also want to get these `values`.  In `python` we have various ways to do this, most often you'll see (and most likely use) `set` and `numpy`'s `unique` `functions`. While the first is another example of in-built `python functions` that don't need to be `imported`, the second is a `function` of the `numpy` `module`. They however can achieve the same goal, that is getting us the number of `unique` `values`. 

print('There are %s unique responses regarding movies' %len(set(movies)))

import numpy as np
print('There are %s unique responses regarding movies' %len(np.unique(movies)))

The `function`s themselves will also give us the `list` of `unique` `values`:

import numpy as np
print('The unique responses regarding movies were:  %s' %np.unique(movies))

Doing the same for `snacks` and `animals` again is straightforward:

print('There are %s unique responses regarding snacks' %len(np.unique(snacks)))
print('The unique responses regarding snacks were:  %s' %np.unique(snacks))

print('There are %s unique responses regarding animals' %len(np.unique(animals)))
print('The unique responses regarding animals were:  %s' %np.unique(animals))

##### In-built functions

As indicated before, `list`s have a great set of in-built functions that allow to perform various `operations`/`transformations` on them: `sort`, `append`, `insert` and `remove`. Please note, as these `function`s are part of the `data type` "`list`", you don't prepend (`sort(list)`) but append them: `list.sort()`, `list.append()`, `list.insert()` and `list.remove()`.  

Lets start with `sort` which, as you might have expected, will `sort` our `list`.

movies.sort()
movies

Please note that our `list` is modified/changed in-place. While it's nice to not have to do a new `assignment`, this can become problematic if the `index` is of relevance!

`.sort()` also allows you to specify _how_ the `list` should be sorted: `ascending` or `descending`. This is controlled via the `reverse` argument of the `.sort()` `function`. By default, `list`s are sorted in an `descending` order. If you want to sort your `list` in an `ascending` order you have to set the `reverse` argument to `True`: `list.sort(reverse=True)`.

movies.sort(reverse=True)
movies

We of course also want to sort our `list`s of `snacks` and `animals`:

snacks.sort()
snacks

animals.sort()
animals

Lets assume we got new data from two participants and thus need to update our `list`, we can simply use `.append()` to, well, append or add these new entries:

movies.append('My Neighbor Totoro')
movies

We obviously do the same for `snacks` and `animals` again:

snacks.append('Wasabi Peanuts')
snacks

animals.append('bear')
animals

Should the `index` of the new `value` be important, you have to use `.insert` as `.append` will only ever, you guessed it: append. The `.insert` `functions` takes two `arguments`, the `index` where a new `value` should be inserted and the `value` that should be inserted: `list.insert(index, value)`. The `index` of the subsequent `values` will shift +1 accordingly. Assuming, we want to add a new `value` at the `third index` of each of our `list`s, this how we would do that:

movies.insert(2, 'The Big Lebowski')
print(movies)

snacks.insert(2, 'PB&J')
print(snacks)

animals.insert(2, 'Manatee')
print(animals)

If you want to change the `value` of a `list` `element` (e.g. you noticed an error and need to change the `value`), you can do that directly by `assigning` a new `value` to the `element` at the given `index`:

print('The element at index 15 of the list movies is: %s\n' %movies[15])
movies[15] = 'Harry Potter and the Python-Prince'
print('It is now %s\n' %movies[15])
print(movies)

Please note that this is final and the original `value` overwritten. The characteristic of modifying `lists` by `assigning` new `values` to `elements` in the `list` is called `mutable` in technical jargon.

If you want to remove an `element` of a given `list` (e.g. you noticed there are unwanted duplicates, etc.), you basically have two options `list.remove(element)` and `del list[index]` and which one you have to use depends on the goal. As you can see `.remove(element)` expects the `element` that should be removed from the `list`, that is the `element` with the specified `value`. For example, if we want to remove the duplicate `Shutter Island`, we can do the following:  

movies.remove("Shutter Island")
print(movies)

As you can see, only one "Shutter Island" is removed and not both, this is because `.remove(element)` only removes the first `element` of the `specified` value. Thus, if there are more `elements` you want to remove, the `del` `function` could be more handy.  

You may have noticed that the `del` `function` expects the `index` of the `element` that should be removed from the `list`. Thus, if we, for example, want to remove the duplicate `Inception`, we can also achieve that via the following: 

del movies[12]
print(movies)

If there are multiple `elements` you want to remove, you can make use of `slicing` again. For example, our `snack` `list` has multiple `elements` with the `value` `chocolate`. Nothing against `chocolate`, but you might want to have this `value` only once in our `list`. To achieve that, we basically combine `slicing` and `del` via indicating what `indices` should be removed:

del snacks[7:9]
print(snacks)

`List`s play a very important role in `python` and are for example used in `loops` and other `flow control structures` (discussed in the next session). There are a number of convenient `functions` for generating `list`s of various `types`, for example, the `range` function (note that in Python 3 `range` creates a generator, so you have to use the `list` `function` to convert the output to a `list`). Creating a `list` that ranges from `10` to `50` advancing in steps of `2` is as easy as:

start = 10
stop = 50
step = 2

list(range(start, stop, step))

This can be very handy if you want to create `participant list`s and/or `condition list`s for your experiment and/or analyzes.

##### Exercise 6.1 

Create a `variable` called `rare_animals` that stores the following `values`: 'Addax', 'Black-footed Ferret', 'Northern Bald Ibis', 'Cross River Gorilla', 'Saola', 'Amur Leopard', 'Philippine Crocodile', 'Sumatran Rhino', 'South China Tiger', 'Vaquita'. After that, please count how many `elements` the `list` has and provide the info within the following statement: "There are [insert number of elements here] animals in the list."     

# Please write your solution here

rare_animals = ['Addax', 'Black-footed Ferret', 'Northern Bald Ibis', 'Cross River Gorilla', 'Saola', 'Amur Leopard', 'Philippine Crocodile', 'Sumatran Rhino', 'South China Tiger', 'Vaquita']
print("There are %s animals in the list." %len(rare_animals))

##### Exercise 6.2

Please add 'Manatee' to the list and subsequently evaluate how many unique `elements` the list has. 

# Please write your solution here

rare_animals.append('Manatee')
len(set(rare_animals))

##### Exercise 6.3

Learning that the `manatee` is not endangered anymore, please remove it from the list. Unfortunately, we have to add "Giant Panda". Could you please do that at `index` `3`.

# Please write your solution here

rare_animals.remove('Manatee') # or del rare_animals[11]
rare_animals.insert(2, 'Giant Panda')
rare_animals

### Tuples

`Tuples` are like `lists`, except that they cannot be modified once created, that is they are **immutable**. 

</br>
<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/python_data_types_2.png"></center>

In `python`, `tuples` are created using the syntax `(..., ..., ...)`, or even `..., ...`:

point = (10, 20, 'Whoa another thing')
print(type(point))
print(point)

`Elements` of `tuples` can also be referenced via their respective `index`:

print(point[0])
print(point[1])
print(point[2])

However, as mentioned above, if we try to `assign` a new `value` to an `element` in a `tuple` we get an `error`:

try:
    point[0] = 20
except(TypeError) as er:
    print("TypeError:", er)
else:
    raise

Thus, `tuples` also don't have the set of `functions` to modify `elements` `lists` do.

##### Exercise 7.1

Please create a `tuple` called `deep_thought` with the following `values`: 'answer', `42`.

# Please write your solution here

deep_thought = ('answer', 42)
deep_thought

### Dictionaries

`Dictionaries` are also like `lists`, except that each element is a `key-value pair`. That is, `elements` or `entries` of the `dictionary`, the `values`, can only be assessed via their respective `key` and not via `indexing`, `slicing`, etc. . 

</br>
<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/python_data_types_2.png"></center>

The syntax for dictionaries is `{key1 : value1, ...}`.

`Dictionaries` are fantastic if you need to organize your data in a highly `structured` way where a precise mapping of a multiple of `type`s is crucial and `list`s might be insufficient. For example, we want to have the information we assessed above for our `list`s in a detailed and holistic manner. Instead of specifying multiple `list`s and `variables`, we could also create a `dictionary` that comprises all of that information.

movie_info = {"n_responses" :        len(movies),
              "n_responses_unique" : len(set(movies)),
              "responses" :          movies}

print(type(movie_info))
print(movie_info)

See how great this is? We have everything in one place and can access the information we need/want via the respective `key`.

movie_info['n_responses']

movie_info['n_responses_unique']

movie_info['responses']

As you can see and mentioned before: like `list`s, each `value` of a `dictionary` can entail various `types`: `integer`, `float`, `string` and even `data types`: `list`s, `tuples`, etc. . However, in contrast to `list`s, we can access `entries` can only via their `key name` and not `indices`:

movie_info[1]

If you try that, you'll get a `KeyError` indicating that the `key` whose `value` you want to access doesn't exist.
If you're uncertain about the `keys` in your `dictionary`, you can get a `list` of all of them via `dictionary.keys()`:

movie_info.keys()

Comparably, if you want to get all the `values`, you can use `dictionary.values()` to obtain a respective `list`:

movie_info.values()

Assuming you want to add new information to your `dictionary`, i.e. a new `key`, this can directly be done via `dictionary[new_key] = value`. For example, you run some stats on our `list` of `movies` and determined that this selection is `significantly awesome` with a `p value` of `0.000001`, we can easily add this to our `dictionary`:

movie_info['movie_selection_awesome'] = True
movie_info['movie_selection_awesome_p_value'] = 0.000001
movie_info

As with `list`s, the `value` of a given `element`, here `entry` can be modified and deleted. This means they are `mutable`. If we for example forgot to correct our `p value` for `multiple comparisons`, we can simply overwrite the original with corrected one: 

movie_info['movie_selection_awesome_p_value'] = 0.001
movie_info

Assuming we then want to delete the `p value` from our `dictionary` because we remembered that the sole focus on `significance` and `p values` brought the entirety of science down to a very bad place, we can achieve this via:

del movie_info['movie_selection_awesome_p_value']
movie_info

##### Exercise 8.1

Oh damn, we completely forgot to create a comparable `dictionary` for our `snacks` `list`. How would create one that follows the example from the `movie` `list`? NB: you can skip the `p value` right away: 

# Please write your solution here

snack_info = {"n_responses" :            len(snacks),
              "n_responses_unique" :     len(set(snacks)),
              "responses" :              snacks,
              "snack_selection_awesome": True}

print(type(snack_info))
print(snack_info)

##### Exercise 8.2 

Obviously, we would love to do the same thing for our `list` of `animals` again!

# Please write your solution here

animal_info = {"n_responses" :            len(animals),
              "n_responses_unique" :      len(set(animals)),
              "responses" :               animals,
              "animal_selection_awesome": True}

print(type(animal_info))
print(animal_info)

### Everything is an object in Python

* All of these `data types` are actually just `objects` in `python`
* *Everything* is an object in `python`!
* The `operations` you can perform with a `variable` depend on the `object`'s definition
* E.g., the `multiplication operator` `*` is defined for some `objects` but not others

</br>
<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/python_data_types_2.png"></center>

## Homework assignment #4

Your fourth homework assignment will entail working through a few tasks covering the contents discussed in this session within of a `jupyter notebook`. You can download it [here](https://www.dropbox.com/s/x2suy8dfcmb3yuf/PFP_assignment_4_intro_python_2.ipynb?dl=1). In order to open it, put the `homework assignment notebook` within the folder you stored the `course materials`, start a `jupyter notebook` as during the sessions, navigate to the `homework assignment notebook`, open it and have fun! NB: a substantial part of it will be _optional_ and thus the notebook will look way longer than it actually is.

Also, in preparation for our switch to more applied things, starting with `experiments` in `python` and thus endeavors in `psychopy`, please create a new respective `conda environment` (remember those?) via the following lines:

`conda create -n psychopy psychopy`  
`conda activate psychopy`  
`pip install jedi psychtoolbox pygame pyo pyobjc python-vlc ujson`  

and then test if everything works via running `psychopy` (type it and then press "enter") from within the new `environment`?

**Deadline: 17/11/2022, 11:59 PM EST**