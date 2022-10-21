# Introduction VII - Introduction to Python III

[Maren Wehrheim]()

## Before we get started ...
<br>

- most of what you’ll see within this lecture was prepared by Ross Markello, Michael Notter and Peer Herholz and further adapted for this course by Maren Wehrheim 
- based on Tal Yarkoni's ["Introduction to Python" lecture at Neurohackademy 2019](https://neurohackademy.org/course/introduction-to-python-2/)
- based on [IPython notebooks from J. R. Johansson](http://github.com/jrjohansson/scientific-python-lectures)
- based on http://www.stavros.io/tutorials/python/ & http://www.swaroopch.com/notes/python
- based on https://github.com/oesteban/biss2016 &  https://github.com/jvns/pandas-cookbook

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

# Recap of the last session

Before we dive into new endeavors, it might be important to briefly recap the things we've talked about so far. Specifically, we will do this to evaluate if everyone's roughly on the same page. Thus, if some of the aspects within the recap are either new or fuzzy to you, please have a quick look at the respective part of the [first session](https://peerherholz.github.io/Python_for_Psychologists_Winter2021/introduction/intro_python_I.html) and [second session](https://peerherholz.github.io/Python_for_Psychologists_Winter2021/introduction/intro_python_II.html) again and as usual: ask questions wherever something is not clear.

This recap will cover the following topics from the last sessions:

- variables and types
- operators and comparison
- strings, lists and dictionaries

## Variables and data types

Two session ago we declared several variables. Could you here create a variable for the following:

- a variable random_number with a number between 0 and 10
- a variable called my_favorite_food indicating your favorite food

What data types do each of these variables have?

# Please write your solution here

random_number = 7  # integer
my_favorite_food = 'pasta'  # string

## operators and comparison

Please calculate the following and `assign` them to the `variables` `a`, `b`, `c`, etc., if needed:

- 10 divided by 3
- 6 to the power of 2
- 5 times 29 times 48
- Update c by adding 10
- Update b by dividing it by 5
- Is c divisible by 5?
- Is a larger than b?
- Are c and b equal?

# Please write your solution here

a = 10 / 3
b = 6 ** 2
c = 5 * 29 * 48
c += 10
b /= 5
print('Is c divisible by 5?', c % 5 == 0)
print('Is a larger than b?', a > b)
print('Are c and b equal?', c == b)

## strings, lists and dictionaries

You just did an experiment asking 10 people for a random number between 0 and 100. What data structure would you use to store these values within python? 
Assume the values were: 1; 50; 23; 77; 91; 3; 34; 81; 55; 49. How would you define a variable of your selected data type to store these values?

# Please write your solution here

# you would use a list
random_numbers = [1, 50, 23, 77, 91, 3, 34, 81, 55, 49]

From the above random numbers you now want to calculate the mean an test if it is above. How would you do that?

# Please write your solution here

mean_value = (1 + 50 + 23 + 77 + 91 + 3 + 34 + 81 + 55 + 49)/ 10
is_above_50 = mean_value > 50

Define a variable rand_num_sub_4 that contains the random number from the fourth subject of your experiment and print the statement 'Subject number four picked 77 as random number' thereby acessing your variable. 
Assign the variable using indexing. Remember: Where do you start to index in Python?

# Please write your solution here

rand_num_sub_4 = random_numbers[3]
print(f'Subject number four picked {rand_num_sub_4} as random number')

Oh no you made several mistakes! Update your list to correct the mistakes.
- The random number of the fourth participant wasn't actually 77 but 76. 
- You forgot to keep track of another participant and actually recorded 11 subjects. You forgot the 8th subject with the number 33. Please insert into your list
- You actually don't want people to choose the number 50. Please remove this value from your list

# Please write your solution here

random_numbers[3] = 76
random_numbers.insert(7, 33)
random_numbers.remove(50)
print(random_numbers)

Instead of only declaring a list containing all the random numbers, you now want to also assign a personal ID to each subject that you asked. In our case this would simply be Sub_1 to Sub_10 for the first, second, etc. subject, respectively. Define a dictionary to store this information. Then print all the keys and also redefine the rand_num_sub_4 variable from before, this time acessing your dictionary.

# Please write your solution here

random_num_dict = {'Sub_1': 1,
                   'Sub_2': 23,
                   'Sub_3': 76,
                   'Sub_4': 91,
                   'Sub_5': 3,
                   'Sub_6': 34,
                   'Sub_7': 81,
                   'Sub_8': 81,
                   'Sub_9': 55,
                   'Sub_10': 49}

print(random_num_dict.keys())

rand_num_sub_4 = random_num_dict['Sub_4']

lord_of_the_rings = "One Ring to rule them all, One Ring to find them, One Ring to bring them all, and in the darkness bind them."

Above we defined the inscription on the ring from lord of the rings. We also want to create a fantasy saga. But we are little copy cats. Therefore, we want to define a new statement for the lord of the stones, and only replace the word Ring with stone in the above sentence. How would you do that?

# Please write your solution here

lord_of_the_stones = lord_of_the_rings.replace('Ring', 'Stone')

How long is this statement now compared to before? Also what was at the indices 40 to 45 before and now in your new sentence?

# Please write your solution here

print(f'Length before {len(lord_of_the_rings)}, Length now {len(lord_of_the_stones)}')
print(f'Word at indices before: {lord_of_the_rings[40:46]}, now: {lord_of_the_stones[40:46]}')

You can split a string into a list of words using list.split(' '). This statementn splits the string at the whitespaces and generates a list.
Split the lord of the stones string into a list of words, sort it and print the 10th element. 

# Please write your solution here

splitted_statement = lord_of_the_stones.split(' ')
sorted_statement = sorted(splitted_statement)
print('The 10th element is:', sorted_statement[9])

Alright, thanks for taking the time to go through this recap. Again: if you could solve/answer all questions, you should have the information/knowledge needed for this session.

# Indentation & Control Flow

## Objectives 📍
<br>

- learn about indentation (some form of code structuring)
- control how our code is executed 
    - Conditionals (`if` - `else` statements statements)
    - Iteration (e.g., `for`-loops, `while` statements…)

## Indentation

`Python` uses `whitespaces` to define `code blocks`. Using `whitespaces` at the beginning of a line is the `indentation`. This means that a `codeblock` that is `indented` with the same number of leading `whitespaces` or `tabs` should be run together. In other words: the `indentation` is part of the `syntax` in `python` and one of the major distinctions regarding other programming languages like, e.g. `Matlab`.  

Usually in `Python` we use `four whitespaces` for `indentation` of `codeblocks`.

Let's see what that means:

days_til_christmas = 15
current_weekday = 'Thursday'

Each such set of statements is called a block, meaning that the lines/`variable` assignments will be run together. We will see examples of how blocks are important later on. What happens when we introduce a "wrong" `indentation`?

days_til_christmas = 15
  current_day = 'Thursday'

One thing you should remember is that a wrong `indentation` raises an `IndentationError`.

## Control Flow & structures

* programming language, i.e. `python`, features that allow us to `control` how code is `executed`
* Conditionals (`if` - `else` statements statements)
* Iteration (e.g., `for`-loops, `while` statements…)
* [Read more here](https://docs.python.org/3/tutorial/controlflow.html)

## Conditional statements: if, elif, else

The `python syntax` for `conditional execution` of `code` use the keywords `if`, `elif` (else if), `else`:

<center><img src="https://www.learnbyexample.org/wp-content/uploads/python/Python-elif-Statement-Syntax.png" width=600></center>
<center><small><small><small>https://www.learnbyexample.org/wp-content/uploads/python/Python-elif-Statement-Syntax.png</small></small></small></center>


<center><img src="https://www.alphacodingskills.com/python/img/python-if-elif-else.png" width=600></center>
<center><small><small><small>https://www.alphacodingskills.com/python/img/python-if-elif-else.png</small></small></small></center>

statement1 = False
statement2 = False

if statement1:
    print("statement1 is True")
    
elif statement2:
    print("statement2 is True")
    
else:
    print("statement1 and statement2 are False")

For the first time, here we encountered a peculiar and unusual aspect of the Python programming language: Program blocks are defined by their indentation level. In Python, the extent of a code block is defined by the indentation level (usually a tab or say four white spaces). This means that we have to be careful to indent our code correctly, or else we will get syntax errors. 

**Examples:**

# Good indentation
statement1 = statement2 = True

if statement1:
    if statement2:
        print("both statement1 and statement2 are True")

# Bad indentation! This would lead to error
if statement1:
    if statement2:
    print("both statement1 and statement2 are True")  # this line is not properly indented

statement1 = False 

if statement1:
    print("printed if statement1 is True")
    
    print("still inside the if block")

if statement1:
    print("printed if statement1 is True")
    
print("now outside the if block")

# We can even combine statements
if statement1 and statement2:
    print("printed if statement1 and statement2 are True")
elif statement1 or statement2:
    print("printed if either statement1 or statement2 is True")
else:
    print("printed if no statement is True")

##### Exercise 1.1


You want to go to the cinema, but you first need to check whether you have enough money. First define a variable indicating the amount of money in your wallet `money_in_wallet` as 6 EUR and the ticket price for the cinema `ticket_price` as 10 EUR. Indicate with a print statement if you can go to the cinema

### Write your solution here ###


money_in_wallet = 6
price_of_ticket = 10


if money_in_wallet >= price_of_ticket:
    print("Let's go to the cinema")
else:
    print("I have to stay home :(")

#### Exercise 1.2


Different films cost different amounts of money. Use if statements to tests which films you can afford and print them. The films cost:

- James Bond: 15 EUR
- Spider Man - No Way Home: 11 EUR
- Dune: 6 EUR
- Ghostbusters: 5 EUR

Note: as always in coding, there are several ways to get the right solution. 

### Write your solution here ###


# possible solution one

if money_in_wallet >= 15:
    print('I can watch every film: James Bond, Spider Man, Dune & Ghostbusters')
elif money_in_wallet >= 11:
    print('I can watch Spider Man, Dune & Ghostbusters')
elif money_in_wallet >= 6:
    print('I can watch Dune & Ghostbusters')
elif money_in_wallet >= 5:
    print('I can watch Ghostbusters')
else:
    print('I can\'t watch any film.')
    

# possible solution two:

affordable_films = []
if money_in_wallet >= 15:
    affordable_films.append('James Bond')

if money_in_wallet >= 11:
    affordable_films.append('Spider Man')

if money_in_wallet >= 6:
    affordable_films.append('Dune')
    
if money_in_wallet >= 5:
    affordable_films.append('Ghostbusters')

print('I can watch the following films:', ', '.join(affordable_films))


#### Exercise 1.3


It's your lucky day! You happen to find 34 EUR right in front of your house. You want to celebrate this by inviting as many of your 5 friends as possible to the cinema. How many friends can you invite, while also paying for yourself? First update your money in your wallet and then test this using if statements. 

### Write your solution here ###



money_in_wallet += 44

if money_in_wallet >= (5+1) * 10:
    print('I can invite all 5 friends')
elif money_in_wallet >= (4+1) * 10:
    print('I can invite 4 friends.')
elif money_in_wallet >= (3+1) * 10:
    print('I can invite 3 friends.')
elif money_in_wallet >= (2+1) * 10:
    print('I can invite 2 friends.')
elif money_in_wallet >= (1+1) * 10:
    print('I can invite 1 friend.')
elif money_in_wallet >= (0+1) * 10:
    print('I cannot invite any of my friends.')
else:
    print('I can\'t even go to the cinema by myself.')

#### Exercise 1.4 - Last one, I promise!


This year you want to treat your dog with a christmas present but obviously only if it was a good boy. Depending on the money in your wallet, you can either buy a new toy for 10 EUR or go on a nice long walk if you don't have any money. Write a nested if statement to test which present you can buy if your dog was a good boy. Your current endowment is 11 EUR.


<center><img src="https://i.imgur.com/yb1CFEh.png"></center>

### Write your solution here ###


was_good_boy = True
endowment = 11
if was_good_boy:
    if endowment >= 10:
        print('Buy the toy')
    else:
        print('Can\'t afford any present. Go on a walk.')
else:
    print('Only presents for good boys')


These many `if` statements were very tedious. Can we not do this more efficiently?

<center><img src="https://i.pinimg.com/736x/f1/be/86/f1be8629d5673e3261726e5011c46e5a.jpg"></center>



Wait no more. We have a solution (at least for some situations)

## Loops

In `python`, loops can be programmed in a number of different ways. The most common is the `for` loop, which is used together with `iterable objects`, such as `list`s. The `basic syntax` is:

<center><img src="https://www.learnbyexample.org/wp-content/uploads/python/Python-for-Loop-Syntax.png" width=600></center>
<center><small><small><small>https://www.learnbyexample.org/wp-content/uploads/python/Python-for-Loop-Syntax.png</small></small></small></center>

for x in [1,2,3]:
    print(x)

The `for` loop iterates over the elements of the supplied list and executes the containing block once for each element. Any kind of list can be used in the `for` loop. For example:

for x in range(4): # by default range start at 0
    print(x)

Note: `range(4)` does not include `4`! Try to remember the aspects of `indexing` and `slicing` we addressed during the session on `string`s and `list`s. 

for x in range(-3,3):
    print(x)

for word in ["scientific", "computing", "with", "python"]:
    print(word)

To iterate over key-value pairs of a dictionary:

params = {
    'parameter1': 'A',
    'parameter2': 'B',
    'parameter3': 'C',
    'parameter4': 'D'
}

for key, value in params.items():
    print(key + " = " + str(value))

Sometimes it is useful to have access to the indices of the values when iterating over a `list`. We can use the `enumerate` function for this:

for idx, x in enumerate(range(-3,3)):
    print(idx, x)

### `break`, `continue` and `pass`

To control the flow of a certain loop you can also use `break`, `continue` and `pass`.

</br>

<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/python_break_continue_pass.png" width=900></center>


rangelist = list(range(10))
print(list(rangelist))

for number in rangelist:
    # Check if number is one of
    # the numbers in the tuple.
    if number in [4, 5, 7, 9]:
        # "Break" terminates a for without
        # executing the "else" clause.
        break
    else:
        # "Continue" starts the next iteration
        # of the loop. It's rather useless here,
        # as it's the last statement of the loop.
        print(number)
        continue
else:
    # The "else" clause is optional and is
    # executed only if the loop didn't "break".
    pass # Do nothing

#### Exercise 2.1

Use a for loop to print every **even** number from 0 to 10

Hint: you can check if a number is even with `number % 2 == 0`

### Write your solution here ###


for num in range(10):
    if num % 2 == 0:
        print(num)
    else: 
        continue

#### Exercise 2.2

Use a for loop that iterates over all days in december (31) and always prints the number of days left until christmas. This loop should break once christmas is reached and should wish you a merry christmas. 


### Write your solution here ###


christmas_day = 24
for i in range(31):
    if i < christmas_day:
        print(f'{christmas_day - i} days until christmas')
    else:
        print('It\'s christmas day!')
        break    

#### Exercise 2.3

create a list of your three most favourite foods and iterate over it. In each iteration print the position and the food. For example:
- 1. Pasta
- 2. Nutella
- 3. Wraps

### Write your solution here ###


fav_food = ['Pasta', 'Nutella', 'Wraps']
for i, food in enumerate(fav_food):
    print(f'{i+1}. {food}')

**List comprehensions: Creating lists using `for` loops**:

A convenient and compact way to initialize lists:


<center><img src="https://www.learnbyexample.org/wp-content/uploads/python/Python-List-Comprehension-Syntax.png" width=600></center>
<center><small><small><small><small><small><small><small>https://www.learnbyexample.org/wp-content/uploads/python/Python-List-Comprehension-Syntax.png</small></small></small></small></small></small></small></center>

<center><img src="https://4.bp.blogspot.com/-uRPZqKbIGwQ/XRtgWhC6qqI/AAAAAAAAH0w/--oGnwKsnpo00GwQgH2gV3RPwHwK8uONgCLcBGAs/s1600/comprehension.PNG" width=600></center>
<center><small><small><small><small><small><small><small>https://4.bp.blogspot.com/-uRPZqKbIGwQ/XRtgWhC6qqI/AAAAAAAAH0w/--oGnwKsnpo00GwQgH2gV3RPwHwK8uONgCLcBGAs/s1600/comprehension.PNG</small></small></small></small></small></small></small></center>

l1 = [x**2 for x in range(0,5)]

print(l1)

You can also use an if statement in your list comprehension. But be careful with the ordering. A single if can be after the for loop. However, if and else together have to be in front. Lets see some examples.

l2 = [x**2 for x in range(0,5) if x > 2]
print(l2)

l3 = [x**2 for x in range(0,5) if x > 2 else x]
print(l3)

Now lets put the if-else statement before the `for` loop

l3 = [x**2 if x > 2 else x for x in range(0,5)]
print(l3)

#### Exercise 2.4

Use list comprehension to create a list containing all letters that are in the word 'human'

### Write your solution here ###


h_letters = [ letter for letter in 'human' ]
print( h_letters)

#### Exercise 2.5
Use list comprehension to create a list with all **even** numbers from 0 to 19 

### Write your solution here ###


number_list = [ x for x in range(20) if x % 2 == 0]
print(number_list)

#### Exercise 2.6
Use list comprehension to create a list with all **uneven** numbers from 0 to 19 and set all even numbers to zero

### Write your solution here ###


number_list = [ 0 if x % 2 == 0 else x for x in range(20) ]
print(number_list)

## `while` loops:

A `while loop` is used when you want to perform a task `indefinitely`, until a particular `condition` is met. It is s a `condition-controlled loop`.

<center><img src="https://www.learnbyexample.org/wp-content/uploads/python/Python-while-Loop-Syntax.png" width=600></center>
<center><small><small><small><small><small><small><small>https://www.learnbyexample.org/wp-content/uploads/python/Python-while-Loop-Syntax.png</small></small></small></small></small></small></small></center>

<center><img src="https://media.geeksforgeeks.org/wp-content/uploads/20191101170515/while-loop.jpg" width=600></center>
<center><small><small><small><small><small><small><small>https://media.geeksforgeeks.org/wp-content/uploads/20191101170515/while-loop.jpg</small></small></small></small></small></small></small></center>

i = 0

while i < 5:
    print(i)
    
    i = i + 1
    
print("done")

Note that the `print "done"` statement is not part of the `while` loop body because of the difference in the indentation.

`while`loops can be dangerous to use. For example if you would forget the `i = i+1` statement

# if you run this cell you will need to stop the kernel, as it is an infinite loop
# i = 0
# while i < 5:
#     print(i)

You can also include an `else` statement after your `while` loop, to include a codeblock that should be executed once the condition returns False

counter = 0

while counter < 3:
    print("Inside loop")
    counter = counter + 1
else:
    print("Inside else")

#### Exercise 2.7

Calculate the number of friends that you can invite to the cinema using a `while`-loop

Remember:
- money = 43
- ticket_price = 10

Hint: You can alter your money to calculate this. 
    

### Write your solution here ###


## Functions

Rule of thumb for great `programming`: "Whenever you copy-paste while coding, you do something wrong.'

We use `functions` to solve problems that are `repetitive`. 

What should you put into a `function`: 
- Anything, that you will do more than once
- All `code blocks` that have some kind of `meaning`, e.g. calculating the `square root` of a `value`
- `Code` that can be increased in readability, where you would like to add a comment

A `function` ...
- is a `block of code` that only runs when explicitly called
- can accept `arguments` (or `parameters`) that `alter` its `behavior`
- can accept any number/`type` of `inputs`, but always `return` a `single object`

Note: `functions` can return `tuples` (may look like `multiple objects`)

A function in `Python` is defined using the keyword `def`, followed by a `function name`, a `signature` within parentheses `()`, and a colon `:`. The following `code`, with one additional level of `indentation`, is the `function body`. 

</br>
</br>

<center><img src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/python_functions.png" width=700></center>

Let's check this different components in further detail.

def say_hello():
    # block belonging to the function
    print('hello world')

say_hello() # call the function

The `function` above does not take any `arguments` but only executes the `code` within. As there is no explicit `return statement` the function will simply return `None`. 

Let's check that. To access the `return value` of a `function`, you simply `assign` the `function` to a `variable`, as we learned above and in the previous sessions.

greetings = say_hello()
print(greetings)

As you see, the `function` still printed the `'hello world'` but the returned `value` that was stored in `greetings` is `None`. 

A `function` can also accept `arguments` within the `parentheses`, making it super flexible. For example:

def calc_power(x, p):
    power = x ** p
    return power

print(calc_power(5,2))
print(calc_power(3,3))


As we used a `return statement` this time, the `power` that we calculated is `returned` and can be `printed` or `assigned` to a `variable`. 


These were all simple examples, however, we can also put more complex `code blocks` into a `function`.

def get_maximum(a, b):
    maximum = None
    if a > b:
        print( a, 'is maximum')
        maximum = a
    elif a == b:
        print(a, 'is equal to', b)
        maximum = a
    else:
        print(b, 'is maximum')
        maximum = b
    return maximum

# directly pass literal values
maximum = get_maximum(3, 4)

x = 7
y = 7

# pass variables as arguments
maximum = get_maximum(x, y)

We can also `return` multiple `values` from a `function` using `tuples`:

def calc_powers(x):
    """
    Return a few powers of x.
    """
    return x ** 2, x ** 3, x ** 4

x = 5
powers = calc_powers(x)
print(powers)

As you see, your `output variable` `powers` is a `tuple` that contains the `output values` of the `function`. We can however, split this `tuple` directly into the specific `values`. This can help to make your `code` more readable. 

x2, x3, x4 = calc_powers(5)
print(x3)

**Very important**: `Variables` inside a `function` are treated as `local variables` and therefore don't interfere with `variables` outside the scope of the `function`.

x = 50

def func(x):
    print('x is', x)
    x = 2
    print('Changed local x to', x)

func(x)
print('x is still', x)

To access a `local function variable`, we can extend the `local scope` with the keyword `global`.

x = 50

def func():
    global x

    print('x is', x)
    x = 2
    print('Changed global x to', x)

func()
print('Value of x is', x)

Optionally, but highly recommended, we can define a so called `"docstring"`, which is a `description` of the `functions` purpose and behavior. The `docstring` should follow directly after the `function definition`, before the `code` in the `function body`.

You can also define the `input` and `return` `parameters` in the `docstring`. There are several `conventions` to do this.

def func_with_docstring(s):
    """
    Print a string 's' and tell how many characters it has    
    :param s (string): input string of any length
    :returns: None
    """
    
    print(s + " has " + str(len(s)) + " characters")

help(func_with_docstring)

func_with_docstring('So much fun to write functions')

### Positional vs. keyword arguments

- `Positional arguments` are defined by `position` and must be passed
    - `Arguments` in the `function signature` are filled _in order_
- `Keyword arguments` have a `default value`
    - `Arguments` can be passed in _arbitrary order_ (_after any `positional arguments`_)

You might not think that at the moment, but `coding` is all about readability. You only write it once, but you will probably read it several times. 

In `Python` we can increase readability when calling a `function` by also naming our `positional arguments`. For example:

def calc_power(x, power):
    return x ** power

We could now simply input the variables that we want to compute:

calc_power(2, 2)

The input in the function is **positionally** defined. Hence, the first parameter represents `x` and the second `power`. But does this really tell you what is happening, just from reading the `function`? 
We can increase readability by calling the `parameters` by their names:

calc_power(x=2, power=2)

Now everyone that looks at your code, can directly see what is happening. If we explicitly list the name of the `arguments` in the `function calls`, they do not need to come in the same order as in the `function definition`. This is called **keyword** `arguments` and is often very useful in `functions` that take a lot of `optional arguments`.

Additionally we could also give `default values` to the `arguments` the `function` takes:

def calc_power(x, power=2):
    return x ** power

If we don’t provide a `value` for the `power` argument when calling the the function `calc_power` it `defaults` to the `value` provided in the `function definition`:

x = 3
calc_power(x)

Such `default values` are especially useful for `functions` that take a lot of `arguments`, as it reduces the amount of `arguments` that you have to pass to the `function`. 

`Arguments` with a `default value` ...
- have to be `defined` /called _AFTER_ the `positional values`
- don't have to be called in order

#### Let's quickly talk about function names...

`Function names` are very important. They will allow you to tell the story of your code. Time spent on naming is never wasted time in `coding`.

The 'pythonic' way to write `functions` is to also imply in the name what the `function` will do. 

Some examples:
- when you calculate something you can use `calc` in your `function name`. 
- when you want to test if something is `true` with your `function` you could use `is`. E.g., `is_above_value(x, value=10)` could return `True` if the `input value` `x` is above the `default value`.
- use `print` if your `function` only `prints` `statements`



#### Exercise 3.1

Define a `function` called `get_longest_string` that takes two `strings` as `input` and `returns` the `longer string`. If the `strings` have the same `length`, the `first` one should be `returned`. 

Call your `function` once, using `positional arguments` and once using `keyword arguments`.

## Write your solution here

def get_longest_string(string_one, string_two):
    if len(string_one) >= len(string_two):
        return string_one
    else:
        string_two
        
longest_string = get_longest_string('yoda', 'skywalker')
longest_string = get_longest_string(string_one='yoda', string_two='skywalker')

#### Exercise 3.2

Define a `function` `happy_holidays` that wishes the user happy holidays. It takes the name of the `user` as `input` `argument`. When the `user` does not define a name, your `function` should use a `default value`. 
Then call it once, `inputting` a `user name` and once without. 

## Write your solution here

def happy_holidays(user_name='you awesome individual'):
    print(f'I wish you happy holidays, {user_name}.')

merry_christmas(user_name='Maren')
merry_christmas(user_name='you awesome individual')

## Homework assignment #5

Your fourth homework assignment will entail working through a few tasks covering the contents discussed in this session within of a `jupyter notebook`. You can download it [here](https://www.dropbox.com/s/grci8e8g2gx7jst/PFP_assignment_5_intro_python_3.ipynb?dl=1). In order to open it, put the `homework assignment notebook` within the folder you stored the `course materials`, start a `jupyter notebook` as during the sessions, navigate to the `homework assignment notebook`, open it and have fun! NB: a substantial part of it will be _optional_ and thus the notebook will look way longer than it actually is.

**Deadline: ?**