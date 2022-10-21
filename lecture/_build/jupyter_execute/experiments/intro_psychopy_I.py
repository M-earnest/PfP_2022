# Experimentation I - Introduction to PsychoPy I

[Peer Herholz (he/him)](https://peerherholz.github.io/)  
Habilitation candidate  - [Fiebach Lab](http://www.fiebachlab.org/), [Neurocognitive Psychology](https://www.psychologie.uni-frankfurt.de/49868684/Abteilungen) at [Goethe-University Frankfurt](https://www.goethe-university-frankfurt.de/en?locale=en)    
Research affiliate - [NeuroDataScience lab](https://neurodatascience.github.io/) at [MNI](https://www.mcgill.ca/neuro/)/[McGill](https://www.mcgill.ca/)  
Member - [BIDS](https://bids-specification.readthedocs.io/en/stable/), [ReproNim](https://www.repronim.org/), [Brainhack](https://brainhack.org/), [Neuromod](https://www.cneuromod.ca/), [OHBM SEA-SIG](https://ohbm-environment.org/), [UNIQUE](https://sites.google.com/view/unique-neuro-ai)  

<img align="left" src="https://raw.githubusercontent.com/G0RELLA/gorella_mwn/master/lecture/static/Twitter%20social%20icons%20-%20circle%20-%20blue.png" alt="logo" title="Twitter" width="32" height="20" /> <img align="left" src="https://raw.githubusercontent.com/G0RELLA/gorella_mwn/master/lecture/static/GitHub-Mark-120px-plus.png" alt="logo" title="Github" width="30" height="20" />   &nbsp;&nbsp;@peerherholz 



## Objectives 📍

* get to know the [PsychoPy](https://www.psychopy.org/) library 
* learn basic and efficient usage of its `module`s and `function`s to create _simple_ experiments

## Important note

The main content of this section will be presented via a mixture of [slides]() and [VScode]() which is further outlined in the [respective section of the course website]() and [slides](). As noted there, `jupyter notebooks` aren't the best way to work on and run experiments using `PsychoPy`, instead we need to switch to `IDE`s for this part of the course. Specifically, we will use `VScode` for this.

<img align="center" src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/ipynb_IDE.png" alt="logo" title="jupyter" width="800" height="300" /> 



This `notebook` is thus not intended as the main resource and you shouldn't try to test/run the experiment via the here included `code cells`. Rather, this is meant to be an add-on resource that presents some of the content and especially `code` in a more condensed form. We hope it will be useful/helpful for you.

## Outline

Within this notebook we will go through the basic and required steps to create a new experiment using `PsychoPy`, including:

1. Prerequisites  
1.1 Computing environment  
1.2 Folders & Files
2. `PsychoPy` basics  
2.1 The general idea  
2.2 Input via `dialog boxes`  
2.3 Presenting instructions  
3. `PsychoPy`'s working principles  
3.1 `draw`ing & `flip`ping  
3.2 `trial`s  
4. Input/output    
4.1 collecting responses  
4.2 saving data  
5. A very simple experiment

## Prerequisites

Starting new experiments in `PsychoPy` follows the same guidelines as starting new projects in general. This includes the following:

- create and store everything in a dedicated place on your machine 

- create and use a dedicated computing environment

- document everything or at least as much as possible

- test and save things in very short intervals, basically after every change

<img align="center" src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/experiment_prereq.png" alt="logo" title="jupyter" width="800" height="200" /> 



### Computing environments

As addressed during the first weeks of the course, `computing environments` are essential when programming, not only in `Python`. This refers to reproducibility, transfer/sharing of code and many other factors. Lucky for us `Python` makes it easy to `create` and `manage` `computing environments`, for example using [conda]().

We can thus also use it to create a new `computing environment` specifically dedicated to creating and running a new experiment using `PsychoPy`. Here we will name it `psychopy` and include/install a few dependencies we already know: 

%%bash

conda create -n psychopy psychopy

conda activate psychopy

pip install jedi psychtoolbox pygame pyo pyobjc   
            python-vlc ujson


With these few steps we have our (initial) `computing environment` ready to go!

Let's continue with creating `folders` and `files` we need.

### Folders & files

As mentioned above, it's a good idea to keep things handy and organized. Obviously, this also holds true for running experiments using `PsychoPy`. While there are several ways we could do this, at the minimum we need a dedicated `folder` or `directory` somewhere on our machine within which we will store all corresponding information and `files`. Creating a new `directory` as no biggie using `bash`, we can simply use `mkdir` and specify the wanted `path` and `name`. For the sake of simplicity, let's put everything in a folder called `psychopy_experiment` on our `Desktop`s. 

%%bash
mkdir /path/to/your/Desktop/psychopy_environment

Now we will change our `current working directory` to this new folder

%%bash
cd /path/to/your/Desktop/psychopy_environment

and once there, create a new `python script`, i.e. an empty `python file`. For this we can use `bash`s `touch` `function` followed by the desired `filename`. Keeping things simple again, we will name it `experiment.py` (notice the file extension `.py`).

%%bash
touch experiment.py

Within this, currently empty, `python file`/`script` we will put our `python code` needed to `run` the `experiment`. 

### VScode setup

Once more: please remember that we're switching to an `IDE` for this part of the course, specifically `VScode`, as `jupyter notebooks` aren't the most feasible way to implement/test/run `experiment`s via `PsychoPy`. Therefore, please open `VScode` and within it open the folder we just created (`File` -> `Open Folder`). Next, click on the `experiment.py` file which should open in the `editor` window and finally also open a `terminal` via `Terminal` -> `New Terminal` and activate the `computing environment` we created above. With that your setup is ready for our `PsychoPy` adventure and should look roughly like below:

<img align="center" src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/vscode_setup.png" alt="logo" title="jupyter" width="800" height="400" /> 

## `PsychoPy` basics

It's already time to talk about `PsychoPy`, one of the `python libraries` intended to run `experiments` and acquire `data`. For more information regarding different software and options, their advantages and drawbacks, please consult the [slides](). 

<img align="center" src="https://www.psychopy.org/_static/psychopyLogoOnlineStrap_h480.png" alt="logo" title="jupyter" width="600" height="300" /> 

Make sure to check the [`PsychoPy` website](https://www.psychopy.org/), [documentation](https://psychopy.org/documentation.html) and [forum](https://discourse.psychopy.org/).



### What is `PsychoPy`

- Psychology software in `Python`, i.e. a `Python library`, i.e. completely written in `Python`
- 2002-2003: Jon Pierce began work on this for his own lab (visual neuroscience)
- 2003-2017: a purely volunteer-driven, evenings and weekends project
- 2017-now: still open source and free to install but with professional support


### Idea/goals of `PsychoPy`

- allow scientists to run as wide a range of experiments as possible, easily and via standard computer hardware
- precise enough for psychophysics
- intuitive enough for (undergraduate) psychology (no offence)
- flexible enough for everything else
- capable of running studies in the lab or online


### Things to check/evaluate

- computer hardware settings & interfaces 
- rapid software development
- always check version
- set `version` in `experiment`
- use `environments` for `experiments`
- don’t change version in running `experiments`


First things first: do you have a working `PsychoPy` installation?

We can simply check that via starting an `ipython` session from our `terminal`:

%%bash

ipython

and from within there then try `import`ing `PsychoPy`:

import psychopy

### The general idea

If you don't get an `import error`, at least the basic `installation` should be ok!

Cool, we are now ready to actually do some `coding`! As said before we will do that in our `experiment.py` `script`. While the transition from `jupyter notebooks` to `python scripts` might seem harsh at first, it’s actually straight-forward: the steps we conducted/`commands` we `run` in an incremental fashion will also be indicated/`run` in an incremental fashion here, just within one `python script` line-by-line.

So, what's the first thing we usually do? That's right: `import`ing `modules` and `function`s we need. Comparably to `jupyter notebook`, we will do that at the beginning of our `script`. Please note, that we will go through a realistic example of `coding` `experiments` in `python` and thus might not all `modules`/`functions` we will actually need when we start. Thus, we will add them at the beginning as we go along!

However, we actually haven't checked out what `modules`/`functions` `PsychoPy` has. Let's do that first.


- [psychopy.core](https://psychopy.org/api/core.html): various basic functions, including timing & experiment termination   
- [psychopy.gui](https://psychopy.org/api/gui.html): various basic functions, including timing & experiment termination   
- [psychopy.event](https://psychopy.org/api/event.html): handling of keyboard/mouse/other input from user  
- psychopy.[visual](https://psychopy.org/api/visual/index.html)/[sound](https://psychopy.org/api/sound/index.html): presentation of stimuli of various types (e.g. images, sounds, etc.)   
- [psychopy.data](https://psychopy.org/api/data.html): handling of condition parameters, response registration, trial order, etc.  
- many more …: we unfortunately can’t check out due to time constraints  

Nice, looks like a decent collection of useful/feasible `modules`/`functions`. The question now is: which ones do we need to implement and `run` our `experiment`? Wait a minute...we genuinely didn't even talk about the `experiment` yet...

Let’s assume we have obtained some data regarding favorite `movies`, `snacks` and `animals` from a group of fantastic students (obviously talking about you rockstars!) and now want to test how each respectively provided item is `perceived`/`evaluated` by our entire sample: how would we do that? 

<img align="center" src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/experiment_outline.png" alt="logo" title="jupyter" width="800" height="300" /> 

As you can see, we need all of them!

### Input via dialog boxes

Many `experiments` start with a `GUI dialog box` that allow users/participant to input certain information, for example `participant id`, `session`, `group`, `data storage path`, etc. . We can implement this crucial aspect via the [psychopy.gui module](https://psychopy.org/api/gui.html). Initially, we need to `import` it and thus need to start a new section in our `python script` and after that, we can define the `GUI dialog box` we want to create via a `dictionary` with respective `key-value pairs`.

Please note: at this we will start populating our `experiment.py` `script`. Thus, you should copy paste the respective content of the `code cells` into your `experiment.py` `script` you opened in `VScode`. As we will go step-by-step, the `code`/`script` will get longer and longer. 

#===============
# Import modules
#===============

from psychopy import gui, core # import psychopy modules/functions

#========================================
# Create GUI dialog box for user input
#========================================

# Get subject name, age, handedness and other information through a dialog box
exp_name = 'PfP_2021' # set experiment name
exp_info = {
            'participant': '', # participant name as string
            'age': '', # age name as string
            'left-handed':False, # handedness as boolean
            'like this course':('yes', 'no'), # course feedback as tuple
            'data path': '', # data path as string
            }
dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name) # create GUI dialog box from dictionary

# If 'Cancel' is pressed, quit experiment
if dlg.OK == False:
    core.quit()

That’s actually all we need to test our `GUI dialog box`. In order to do that, we need to `run`/`execute` our `python script` called `experiment.py`. This is achieved via typing `python experiment.py` in the `VScode terminal` and pressing `enter` this will `run`/`execute` the `python script` `experiment.py` via the `python` installed in our `conda environment` called `psychopy`. Again, please don't `run`/`execute` `code` in this `jupyter notebook`!

python experiment.py

If everything works/is set correctly you should see a `GUI dialog box` appearing on your screen asking for the information we indicated in our `experiment.py` `python script` (chances are the layout on your end looks a bit different than mine, that’s no biggie). After entering all requested information and clicking `ok` the `GUI dialog box` should close and no errors should appear.


<img align="center" src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/gui_example.png" alt="logo" title="jupyter" width="350" height="200" /> 

The next aspect we should take care of is the `data handling`, i.e. defining a `data filename` and `path` where it should be saved. We can make use of the `exp_info dictionary` right away and extract important information from there, for example, the `experiment` name and `participant ID`. Additionally, we will obtain the `date` and `time` via the `psychopy.core module`. We will also create a unique `filename` for the resulting `data` and check if the set `data path` works out via the `os` `module`.

#===============
# Import modules
#===============

from psychopy import gui, core # import psychopy modules/functions
import os # import os module

#========================================
# Create GUI dialog box for user input
#========================================

# Get subject name, age, handedness and other information through a dialog box
exp_name = 'PfP_2021' # set experiment name
exp_info = {
            'participant': '', # participant name as string
            'age': '', # age name as string
            'left-handed':False, # handedness as boolean
            'like this course':('yes', 'no'), # course feedback as tuple
            'data path': '', # data path as string
            }
dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name) # create GUI dialog box from dictionary

# If 'Cancel' is pressed, quit experiment
if dlg.OK == False:
    core.quit()

#=================================================
# Data storage: basic information, filename & path
#=================================================

# Get date and time
exp_info['date'] = data.getDateStr() # get date and time via data module
exp_info['exp_name'] = exp_name # set experiment name

# Check if set data path exists, if not create it
if not os.path.isdir(exp_info['data path']):
    os.makedirs(exp_info['data path'])

# Create a unique filename for the experiment data    
data_fname = exp_info['participant'] + '_' + exp_info['date'] # create initial file name from participant ID and date/time
data_fname = os.path.join(exp_info['data path'], data_fname) # add path from GUI dialog box

### Presenting instructions

After having set some crucial backbones of our `experiment`, it’s time to actually start it. Quite often, `experiments` start with several messages of `instructions` that explain the `experiment` to the `participant`. Thus, we will add a few here as well, starting with a common `“welcome” text message`. To display things in general but also text, the [psychopy.visual module](https://psychopy.org/api/visual/index.html) is the way to go. What we need to do now is define a general `experiment window` to utilize during the entire `experiment` and a `text` to be displayed on it. 

#===============
# Import modules
#===============

from psychopy import gui, core, visual # import psychopy modules/functions
import os # import os module

#========================================
# Create GUI dialog box for user input
#========================================

# Get subject name, age, handedness and other information through a dialog box
exp_name = 'PfP_2021' # set experiment name
exp_info = {
            'participant': '', # participant name as string
            'age': '', # age name as string
            'left-handed':False, # handedness as boolean
            'like this course':('yes', 'no'), # course feedback as tuple
            'data path': '', # data path as string
            }
dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name) # create GUI dialog box from dictionary

# If 'Cancel' is pressed, quit experiment
if dlg.OK == False:
    core.quit()

#=================================================
# Data storage: basic information, filename & path
#=================================================

# Get date and time
exp_info['date'] = data.getDateStr() # get date and time via data module
exp_info['exp_name'] = exp_name # set experiment name

# Check if set data path exists, if not create it
if not os.path.isdir(exp_info['data path']):
    os.makedirs(exp_info['data path'])

# Create a unique filename for the experiment data    
data_fname = exp_info['participant'] + '_' + exp_info['date'] # create initial file name from participant ID and date/time
data_fname = os.path.join(exp_info['data path'], data_fname) # add path from GUI dialog box


#===============================
# Creation of window and messages
#===============================

# Open a window
win = visual.Window(size=(800,600), color='gray', units='pix', fullscr=False) # set size, background color, etc. of window

# Define experiment start text
welcome_message = visual.TextStim(win,
                                text="Welcome to the experiment. Please press the spacebar to continue.",
                                color='black', height=40)

## PsychoPy working principles

Here, we came across one of `PsychoPy`’s core working principles: we need a `general experiment window`, i.e. a place we can display/present something on. You can define a variety of different `windows` based on different `screens`/`monitors` which should however be adapted to the `setup` and `experiment` at hand (e.g. `size`, `background color`, etc.). Basically all `experiments` you will set up will require to define a `general experiment window` as without it no `visual stimuli` (e.g. `images`, `text`, `movies`, etc.) can be displayed/presented or how `PsychoPy` would say it: `drawn`

Speaking of which: this is the next core working principle we are going to see and explore is the difference between `draw`ing something and showing it.

### `Draw`ing & `flip`ping

In `PsychoPy` (and many other comparable software) there’s a big difference between `draw`ing and showing something. While we need to `draw` something on/in a `window` that alone won’t actually show it. This is because `PsychoPy` internally uses `“two screens”` one `background` or `buffer` `screen` which is not seen (yet) and one `front screen` which is (currently) seen. When you `draw` something it’s always going to be `draw`n on the `background`/`buffer` `screen`, thus “invisible” and you need to `flip` it to the `front screen` to be “visible”.

<img align="center" src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/draw_flip.png" alt="logo" title="jupyter" width="800" height="400" /> 

Why does `PsychoPy` (and other comparable software) work like that? The idea/aim is always the same: increase performance and minimize delays (as addressed in the [slides]()). `Draw`ing something might take a long time, depending on the stimulus at hand, but `flip`ping something already drawn from the `buffer` to the `front screen` is fast(er). It can thus ensure better and more precise timing. This can work comparably for `images`, `sounds`, `movies`, etc. where things are set/`draw`n/pre-loaded and presented exactly when needed.  

#===============
# Import modules
#===============

from psychopy import gui, core, visual # import psychopy modules/functions
import os # import os module

#========================================
# Create GUI dialog box for user input
#========================================

# Get subject name, age, handedness and other information through a dialog box
exp_name = 'PfP_2021' # set experiment name
exp_info = {
            'participant': '', # participant name as string
            'age': '', # age name as string
            'left-handed':False, # handedness as boolean
            'like this course':('yes', 'no'), # course feedback as tuple
            'data path': '', # data path as string
            }
dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name) # create GUI dialog box from dictionary

# If 'Cancel' is pressed, quit experiment
if dlg.OK == False:
    core.quit()

#=================================================
# Data storage: basic information, filename & path
#=================================================

# Get date and time
exp_info['date'] = data.getDateStr() # get date and time via data module
exp_info['exp_name'] = exp_name # set experiment name

# Check if set data path exists, if not create it
if not os.path.isdir(exp_info['data path']):
    os.makedirs(exp_info['data path'])

# Create a unique filename for the experiment data    
data_fname = exp_info['participant'] + '_' + exp_info['date'] # create initial file name from participant ID and date/time
data_fname = os.path.join(exp_info['data path'], data_fname) # add path from GUI dialog box


#===============================
# Creation of window and messages
#===============================

# Open a window
win = visual.Window(size=(800,600), color='gray', units='pix', fullscr=False) # set size, background color, etc. of window

# Define experiment start text
welcome_message = visual.TextStim(win,
                                text="Welcome to the experiment. Please press the spacebar to continue.",
                                color='black', height=40)

#=====================
# Start the experiment
#=====================

# display welcome message
welcome_message.draw() # draw welcome message to buffer screen
win.flip() # flip it to the front screen

Let’s give it a try via `python experiment.py`. If everything works/is set correctly you should see the `GUI dialog box` again but this time after clicking `OK`, the `text` we defined as a welcome message should appear next.


python experiment.py

However, it only appears very briefly and in contrast to our `GUI dialog box` we don’t need to press anything
to advance. This is because we didn’t tell `PsychoPy` that we want to `wait` for a distinct `key press` before we advance further, we need the [psychopy.event module](https://psychopy.org/api/event.html). Through its `.waitKeys()` `function` we can define that nothing should happen/we shouldn't advancing unless a certain `key` is pressed. While we are at it, let’s add a few more messages to our `experiment`. One will be presented right after the welcome message and explain very generally what will happen in the `experiment`. Another one will be presented at the end of the experiment and display a general “that’s it, thanks for taking part” message. The syntax for creating, `draw`ing and presenting these message is identical to the one we just explored, we only need to change the `text`.

#===============
# Import modules
#===============

from psychopy import gui, core, visual, event # import psychopy modules/functions
import os # import os module

#========================================
# Create GUI dialog box for user input
#========================================

# Get subject name, age, handedness and other information through a dialog box
exp_name = 'PfP_2021' # set experiment name
exp_info = {
            'participant': '', # participant name as string
            'age': '', # age name as string
            'left-handed':False, # handedness as boolean
            'like this course':('yes', 'no'), # course feedback as tuple
            'data path': '', # data path as string
            }
dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name) # create GUI dialog box from dictionary

# If 'Cancel' is pressed, quit experiment
if dlg.OK == False:
    core.quit()

#=================================================
# Data storage: basic information, filename & path
#=================================================

# Get date and time
exp_info['date'] = data.getDateStr() # get date and time via data module
exp_info['exp_name'] = exp_name # set experiment name

# Check if set data path exists, if not create it
if not os.path.isdir(exp_info['data path']):
    os.makedirs(exp_info['data path'])

# Create a unique filename for the experiment data    
data_fname = exp_info['participant'] + '_' + exp_info['date'] # create initial file name from participant ID and date/time
data_fname = os.path.join(exp_info['data path'], data_fname) # add path from GUI dialog box


#===============================
# Creation of window and messages
#===============================

# Open a window
win = visual.Window(size=(800,600), color='gray', units='pix', fullscr=False) # set size, background color, etc. of window

# Define experiment start text
welcome_message = visual.TextStim(win,
                                text="Welcome to the experiment. Please press the spacebar to continue.",
                                color='black', height=40)


# Define trial start text
start_message = visual.TextStim(win,
                                text="In this experiment you will rate different movies, snacks and animals on a scale from 1 to 7. Please press the spacebar to start.",
                                color='black', height=40)

# Define experiment end text
end_message = visual.TextStim(win,
                                text="You have reached the end of the experiment, thanks for participating.",
                                color='black', height=40)


#=====================
# Start the experiment
#=====================

# display welcome message
welcome_message.draw() # draw welcome message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 

# display start message
start_message.draw() # draw start message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 


#======================
# End of the experiment
#======================

# Display end message
end_message.draw() # draw end message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 

Let’s give it a try via `python experiment.py`. If everything works/is set correctly you should see the `GUI dialog box` and after clicking `OK`, the `text` we defined as a welcome message should appear next, followed by the start message and finally the end message. In all cases, the `experiment` should only advance if you press `spacebar` or quit when you press `escape`. 

python experiment.py

Having this rough frame of our `experiment`, it’s actually time to add the `experiment` itself: the `evaluation` of our `movies`, `snacks` and `animals`.

### `trials`

Quick reminder: our `experiment` should collect responses from `participants` regarding our `list`s of `movies`, `snacks` and `animals`, specifically their respective `rating`. Thus we need to add/implement two aspects in our `experiment`: the **presentation of stimuli** and their **rating**/

Starting with the `presentation of stimuli`, we will keep it simple for now and present them via `text`. However, any ideas how we could begin working on this? That’s right: we need to define `list`s with our stimuli!

```
movies = [‘Forrest Gump’, ‘Interstellar’, ‘Love Actually’, …]
```


Within that process, we can already think about the next step: quite often `experiments` `shuffle` the order of `stimuli` across participants to avoid sequence/order effects. We will do the same and implement that via the [numpy.random module](https://numpy.org/doc/stable/reference/random/index.html), specifically its [.shuffle()](https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html) `function` which will allow us to randomly `shuffle` our previously created `list`.

```
rnd.shuffle(movies)
movies 
[‘Interstellar’, ‘Love Actually’, ‘Forrest Gump’,  …]
```


After that we need to bring our `shuffle`d `stimuli list` into the format required by `PsychoPy`. Specifically, this refers to the definition of `experiment trials`, i.e. `trials` that will be presented during the `experiment`, including their properties (e.g. `content`, `order`, `repetition`, etc.). In `PsychoPy` this is achieved via the [data.TrialHandler()](https://psychopy.org/api/data.html#psychopy.data.TrialHandler) `function` for which we need to convert our `shuffle`d `stimuli list` into a `list` of `dictionaries` of the form `“stimulus”: value`.

```
stim_order = []

for stim in movies:
    stim_order.append({'stimulus': stim})
    
stim_order

[{‘stimulus’:‘Interstellar’},
 {‘stimulus’:‘Love Actually’},
 {‘stimulus’:‘Forrest Gump’},…]
```

The result is then provided as `input` for the [data.TrialHandler()](https://psychopy.org/api/data.html#psychopy.data.TrialHandler) `function`.     

With that we can simply `loop` over the `trials` in the `trials object` and during each `iteration` `draw` and `flip` the respective `value` of the `dictionary key` `“stimulus”` to present the stimuli of our `list` “movies” one-by-one after one another. 

```
for trial in trials:

    # display/draw respective stimulus within each iteration, notice how the stimulus is set "on the fly"
    visual.TextStim(win, text=trial['stimulus'], bold=True, pos=[0, 30], height=40).draw()
```


Additionally, we want to display the question `“How much do you like the following?”` above the respective stimulus to remind participant about the task. Within each `iteration` of our `for-loop` we will also allow participants to `quit` the `experiment` by pressing `“escape”` via the `event.getKeys()` `function` as we don’t want to wait for a `key` to be `pressed` but want to do something whenever a certain `key` is `pressed`.

#===============
# Import modules
#===============

from psychopy import gui, core, visual, event, data # import psychopy modules/functions
import os # import os module
import numpy.random as rnd # import random module from numpy

#========================================
# Create GUI dialog box for user input
#========================================

# Get subject name, age, handedness and other information through a dialog box
exp_name = 'PfP_2021' # set experiment name
exp_info = {
            'participant': '', # participant name as string
            'age': '', # age name as string
            'left-handed':False, # handedness as boolean
            'like this course':('yes', 'no'), # course feedback as tuple
            'data path': '', # data path as string
            }
dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name) # create GUI dialog box from dictionary

# If 'Cancel' is pressed, quit experiment
if dlg.OK == False:
    core.quit()

#=================================================
# Data storage: basic information, filename & path
#=================================================

# Get date and time
exp_info['date'] = data.getDateStr() # get date and time via data module
exp_info['exp_name'] = exp_name # set experiment name

# Check if set data path exists, if not create it
if not os.path.isdir(exp_info['data path']):
    os.makedirs(exp_info['data path'])

# Create a unique filename for the experiment data    
data_fname = exp_info['participant'] + '_' + exp_info['date'] # create initial file name from participant ID and date/time
data_fname = os.path.join(exp_info['data path'], data_fname) # add path from GUI dialog box


#===============================
# Creation of window and messages
#===============================

# Open a window
win = visual.Window(size=(800,600), color='gray', units='pix', fullscr=False) # set size, background color, etc. of window

# Define experiment start text
welcome_message = visual.TextStim(win,
                                text="Welcome to the experiment. Please press the spacebar to continue.",
                                color='black', height=40)


# Define trial start text
start_message = visual.TextStim(win,
                                text="In this experiment you will rate different movies, snacks and animals on a scale from 1 to 7. Please press the spacebar to start.",
                                color='black', height=40)

# Define experiment end text
end_message = visual.TextStim(win,
                                text="You have reached the end of the experiment, thanks for participating.",
                                color='black', height=40)


#==========================
# Define the trial sequence
#==========================

# Define a list of trials with their properties:

# create empty list
stim_order = [] 

# convert list into list of dictionaries with key ('stimulus') - value pairing
for stim in movies:
    stim_order.append({'stimulus': stim}) 

# use dictionary to create trials object via data.TrialHandler, specifying further settings
trials = data.TrialHandler(stim_order, nReps=1, extraInfo=exp_info,
                           method='sequential') # create


#=====================
# Start the experiment
#=====================

# display welcome message
welcome_message.draw() # draw welcome message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 

# display start message
start_message.draw() # draw start message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 

# Run through the trials, stimulus by stimulus
for trial in trials:

    # display/draw task question to remind participants
    visual.TextStim(win, text='How much do you like the following?', pos=[0, 90], italic=True).draw()

    # display/draw respective stimulus within each iteration, notice how the stimulus is set "on the fly"
    visual.TextStim(win, text=trial['stimulus'], bold=True, pos=[0, 30], height=40).draw()

    # after everything is drawn, flip it to the front screen
    win.flip()

    # if participants press `escape`, stop the experiment
    if event.getKeys(['escape']):
        core.quit()


#======================
# End of the experiment
#======================

# Display end message
end_message.draw() # draw end message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 



Let’s give it a try via `python experiment.py`. If everything works/is set correctly you should see the `GUI dialog box` and after clicking `OK`, the `text` we defined as a welcome message should appear next, followed by the start message. Subsequently, you should see all of our `stimuli` one after another and the same question above them every `trial`. Finally, you should see the end message. 

python experiment.py

While this is already great, the same thing as during our initial tests with the instruction screens happened: the `text`, i.e. our `stimuli`, is only very briefly shown on screen and disappears before we can do anything. That’s because we didn’t specify that we want to collect `responses` before moving on to the next `stimulus` yet. We need to add the `rating` to our `experiment`, specifically the `trials`.

## Input/output

`PsychoPy` offers quite a bit of possible options to collect `responses`: simple yes/no questions, rating scales, visual analog scales, voice recordings, etc. and store outputs (`files`, different levels of detail, etc.).

### Collecting responses

For the `experiment` at hand a simple `rating scale` (yes, a `Likert scale` to make it Psychology, hehe) should be sufficient. As with the other components we have explored so far, we need to implement/add this via two steps: `defining`/`creating` a `rating scale` and `draw`ing/presenting it.

We can easily define and tweak a rating scale via `PsychoPy`’s [visual.RatingScale()](https://psychopy.org/api/visual/ratingscale.html#psychopy.visual.RatingScale) `function` which allows us to set the `range` of `values`, `labels`, `size`, etc..

```
ratingScale = visual.RatingScale(win, 
          scale = None,      
          low = 1,           
          high = 7,          
          showAccept = True, 
          markerStart = 4,   
          labels = ['1 - Not at all', '7 - A lot'], 
          pos = [0, -80])
```


We then need to `draw` it and indicate that we want to `wait` until a `rating` was conducted before we advance to the next `trial`. 

```
    while ratingScale.noResponse:
```

Additionally, we are going to display a small helpful message describing the `rating` and make sure that the `rating scale` is reset back to its `default status` before the next `trial` starts.

```
    ratingScale.reset()
```


Even though `participants` could already perform the `rating` of the `stimuli`, we don’t track and collect the respective `responses` yet. These need to be obtained from the `rating scale` before we `reset` it at the end of the `trial`. As indicated before `visual.RatingScale()` creates an `object`/`class`/`data type` with many inbuilt `functions`, this includes `.getRating()` and `.getRT()` to collect the provided `rating` and corresponding `response time`:

```
    rating = ratingScale.getRating()
    rt = ratingScale.getRT()
```

We can then store both `values` per `trial` in the `trials object` via its `.addData()` `function`:

```
    trials.addData('rt', rt)
    trials.addData('rating', rating)
```

#===============
# Import modules
#===============

from psychopy import gui, core, visual, event, data # import psychopy modules/functions
import os # import os module
import numpy.random as rnd # import random module from numpy

#========================================
# Create GUI dialog box for user input
#========================================

# Get subject name, age, handedness and other information through a dialog box
exp_name = 'PfP_2021' # set experiment name
exp_info = {
            'participant': '', # participant name as string
            'age': '', # age name as string
            'left-handed':False, # handedness as boolean
            'like this course':('yes', 'no'), # course feedback as tuple
            'data path': '', # data path as string
            }
dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name) # create GUI dialog box from dictionary

# If 'Cancel' is pressed, quit experiment
if dlg.OK == False:
    core.quit()

#=================================================
# Data storage: basic information, filename & path
#=================================================

# Get date and time
exp_info['date'] = data.getDateStr() # get date and time via data module
exp_info['exp_name'] = exp_name # set experiment name

# Check if set data path exists, if not create it
if not os.path.isdir(exp_info['data path']):
    os.makedirs(exp_info['data path'])

# Create a unique filename for the experiment data    
data_fname = exp_info['participant'] + '_' + exp_info['date'] # create initial file name from participant ID and date/time
data_fname = os.path.join(exp_info['data path'], data_fname) # add path from GUI dialog box


#===============================
# Creation of window and messages
#===============================

# Open a window
win = visual.Window(size=(800,600), color='gray', units='pix', fullscr=False) # set size, background color, etc. of window

# Define experiment start text
welcome_message = visual.TextStim(win,
                                text="Welcome to the experiment. Please press the spacebar to continue.",
                                color='black', height=40)


# Define trial start text
start_message = visual.TextStim(win,
                                text="In this experiment you will rate different movies, snacks and animals on a scale from 1 to 7. Please press the spacebar to start.",
                                color='black', height=40)

# Define experiment end text
end_message = visual.TextStim(win,
                                text="You have reached the end of the experiment, thanks for participating.",
                                color='black', height=40)


#==========================
# Define the trial sequence
#==========================

# Define a list of trials with their properties:

# create empty list
stim_order = [] 

# convert list into list of dictionaries with key ('stimulus') - value pairing
for stim in movies:
    stim_order.append({'stimulus': stim}) 

# use dictionary to create trials object via data.TrialHandler, specifying further settings
trials = data.TrialHandler(stim_order, nReps=1, extraInfo=exp_info,
                           method='sequential') # create


#================================
# Define a rating scale
#================================

ratingScale = visual.RatingScale(win, 
          scale = None,          # This makes sure there's no subdivision on the scale
          low = 1,               # This is the minimum value I want the scale to have
          high = 7,             # This is the maximum value of the scale
          showAccept = True,    # This shows the user's chosen value in a window below the scale
          markerStart = 4,       # This sets the rating scale to have its marker start on 5
          labels = ['1 - Not at all', '7 - A lot'], # This creates the labels
          pos = [0, -80]) # set the position of the rating scale within the window


#=====================
# Start the experiment
#=====================

# display welcome message
welcome_message.draw() # draw welcome message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 

# display start message
start_message.draw() # draw start message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 

# Run through the trials, stimulus by stimulus
for trial in trials:

    # wait until a rating was conducted before advancing to the next trial
    while ratingScale.noResponse:


        # display/draw task question to remind participants
        visual.TextStim(win, text='How much do you like the following?', pos=[0, 90], italic=True).draw()

        # display/draw respective stimulus within each iteration, notice how the stimulus is set "on the fly"
        visual.TextStim(win, text=trial['stimulus'], bold=True, pos=[0, 30], height=40).draw()

        # display/draw the rating scale
        ratingScale.draw()
        
        # display/draw help message regarding rating scale
        visual.TextStim(win, text='(Move the marker along the line and click "enter" to indicate your rating from 1 to 7.)', 
                        pos=[0,-200], height=14).draw()
        
        # after everything is drawn, flip it to the front screen
        win.flip()

        # if participants press `escape`, stop the experiment
        if event.getKeys(['escape']):
            core.quit()
            
    # get the current rating        
    rating = ratingScale.getRating()

    # get the response time of the current rating 
    rt = ratingScale.getRT()

    # add the current rating to the trials object
    trials.addData('rt', rt)
    
    # add the response time of the current rating to the trials object
    trials.addData('rating', rating)

    # reset the rating scale
    ratingScale.reset()            


#======================
# End of the experiment
#======================

# Display end message
end_message.draw() # draw end message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 



Let’s give it a try via `python experiment.py`. If everything works/is set correctly you should see the `GUI dialog box` and after clicking `OK`, the `text` we defined as a welcome message should appear next, followed by the start message. Subsequently, you should see all of our `stimuli` one after another and the same question above them every `trial`, this time not advancing until you provided a `rating`. Finally, you should see the end message.

python experiment.py

Our `experiment` works as expected but we don’t get any `output files`. The reason is once again simple: we actually didn’t tell `PsychoPy` that we would like to `save` our `data` to an `output file`. Importantly, our `data` is stored in the `trials object`.

### Saving data

Because things work like a charm and we’re using `Python`-based tools, the `trials object` has an `in-built function`, called `.saveAsWideText()`, that easily allows this by indicating the desired `filename`, `type` and `delimiter`.

```
trials.saveAsWideText(data_fname + '.csv', delim=',')
```


#===============
# Import modules
#===============

from psychopy import gui, core, visual, event, data # import psychopy modules/functions
import os # import os module
import numpy.random as rnd # import random module from numpy

#========================================
# Create GUI dialog box for user input
#========================================

# Get subject name, age, handedness and other information through a dialog box
exp_name = 'PfP_2021' # set experiment name
exp_info = {
            'participant': '', # participant name as string
            'age': '', # age name as string
            'left-handed':False, # handedness as boolean
            'like this course':('yes', 'no'), # course feedback as tuple
            'data path': '', # data path as string
            }
dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name) # create GUI dialog box from dictionary

# If 'Cancel' is pressed, quit experiment
if dlg.OK == False:
    core.quit()

#=================================================
# Data storage: basic information, filename & path
#=================================================

# Get date and time
exp_info['date'] = data.getDateStr() # get date and time via data module
exp_info['exp_name'] = exp_name # set experiment name

# Check if set data path exists, if not create it
if not os.path.isdir(exp_info['data path']):
    os.makedirs(exp_info['data path'])

# Create a unique filename for the experiment data    
data_fname = exp_info['participant'] + '_' + exp_info['date'] # create initial file name from participant ID and date/time
data_fname = os.path.join(exp_info['data path'], data_fname) # add path from GUI dialog box


#===============================
# Creation of window and messages
#===============================

# Open a window
win = visual.Window(size=(800,600), color='gray', units='pix', fullscr=False) # set size, background color, etc. of window

# Define experiment start text
welcome_message = visual.TextStim(win,
                                text="Welcome to the experiment. Please press the spacebar to continue.",
                                color='black', height=40)


# Define trial start text
start_message = visual.TextStim(win,
                                text="In this experiment you will rate different movies, snacks and animals on a scale from 1 to 7. Please press the spacebar to start.",
                                color='black', height=40)

# Define experiment end text
end_message = visual.TextStim(win,
                                text="You have reached the end of the experiment, thanks for participating.",
                                color='black', height=40)


#==========================
# Define the trial sequence
#==========================

# Define a list of trials with their properties:

# create empty list
stim_order = [] 

# convert list into list of dictionaries with key ('stimulus') - value pairing
for stim in movies:
    stim_order.append({'stimulus': stim}) 

# use dictionary to create trials object via data.TrialHandler, specifying further settings
trials = data.TrialHandler(stim_order, nReps=1, extraInfo=exp_info,
                           method='sequential') # create


#================================
# Define a rating scale
#================================

ratingScale = visual.RatingScale(win, 
          scale = None,          # This makes sure there's no subdivision on the scale
          low = 1,               # This is the minimum value I want the scale to have
          high = 7,             # This is the maximum value of the scale
          showAccept = True,    # This shows the user's chosen value in a window below the scale
          markerStart = 4,       # This sets the rating scale to have its marker start on 5
          labels = ['1 - Not at all', '7 - A lot'], # This creates the labels
          pos = [0, -80]) # set the position of the rating scale within the window


#=====================
# Start the experiment
#=====================

# display welcome message
welcome_message.draw() # draw welcome message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 

# display start message
start_message.draw() # draw start message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 

# Run through the trials, stimulus by stimulus
for trial in trials:

    # wait until a rating was conducted before advancing to the next trial
    while ratingScale.noResponse:


        # display/draw task question to remind participants
        visual.TextStim(win, text='How much do you like the following?', pos=[0, 90], italic=True).draw()

        # display/draw respective stimulus within each iteration, notice how the stimulus is set "on the fly"
        visual.TextStim(win, text=trial['stimulus'], bold=True, pos=[0, 30], height=40).draw()

        # display/draw the rating scale
        ratingScale.draw()
        
        # display/draw help message regarding rating scale
        visual.TextStim(win, text='(Move the marker along the line and click "enter" to indicate your rating from 1 to 7.)', 
                        pos=[0,-200], height=14).draw()
        
        # after everything is drawn, flip it to the front screen
        win.flip()

        # if participants press `escape`, stop the experiment
        if event.getKeys(['escape']):
            core.quit()
            
    # get the current rating        
    rating = ratingScale.getRating()

    # get the response time of the current rating 
    rt = ratingScale.getRT()

    # add the current rating to the trials object
    trials.addData('rt', rt)
    
    # add the response time of the current rating to the trials object
    trials.addData('rating', rating)

    # reset the rating scale
    ratingScale.reset()            


#======================
# End of the experiment
#======================

# Save all data to a file
trials.saveAsWideText(data_fname + '.csv', delim=',')

# Display end message
end_message.draw() # draw end message to buffer screen
win.flip() # flip it to the front screen
keys = event.waitKeys(keyList=['space', 'escape']) # wait for spacebar key press before advancing or quit when escape is pressed 


If you now try it again, everything should work as before bu after finishing the `experiment` you should see a new `file` within the indicated `data path` containing all `information` stored about the `experiment`: `trials`, `stimuli`, `responses`, `reaction times`, etc. . 

## A very simple experiment

Believe it or not folks, with that we already created our first working `PsychoPy` `experiment`. Using only a small amount of, _very readable_, `code`, we can obtain `ratings` for our `stimuli`. Obviously, this is a very simple `experiment` but nevertheless a good start, showcasing a lot of the core things you should know to start using `PsychoPy` for `experiments`. All the things addressed here are usually also part of much more complex `experiments`, as well as build their basis.

<img align="center" src="https://raw.githubusercontent.com/PeerHerholz/Python_for_Psychologists_Winter2021/master/lecture/static/experiment_outline.png" alt="logo" title="jupyter" width="800" height="350" /> 

## Outro

As usually: awesome work folks! The transition from basic `Python` aspects to applied together is definitely no cake-walk, especially when simultaneously switching to `python scripts` from `jupyter notebooks` but you did a great job!
Thank so much for sticking with us throughout this!

<img align="center" src="https://media4.giphy.com/media/7zWYE1ostmPWZdygj3/giphy.gif?cid=ecf05e475wkt0wsob0pgmaggnreymvom43vxe6ainhr1dzh0&rid=giphy.gif&ct=g" alt="logo" title="jupyter" width="600" height="300" /> 