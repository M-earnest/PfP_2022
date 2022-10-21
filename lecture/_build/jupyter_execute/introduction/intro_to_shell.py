## Introduction II - the (unix) command line: bash


[Michael Ernst](https://github.com/M-earnest)  
Phd student - [Fiebach Lab](http://www.fiebachlab.org/), [Neurocognitive Psychology](https://www.psychologie.uni-frankfurt.de/49868684/Abteilungen) at [Goethe-University Frankfurt](https://www.goethe-university-frankfurt.de/en?locale=en)

## Before we get started 1...
<br>

- most of what you’ll see within this lecture was prepared by Ross Markello and further adapted by Peer Herholz & Michael Ernst
- based on the Software Carpentries "[Introduction to the Shell](https://swcarpentry.github.io/shell-novice/)" under CC-BY 4.0

[Peer Herholz (he/him)](https://peerherholz.github.io/)  
Research affiliate - [NeuroDataScience lab](https://neurodatascience.github.io/) at [MNI](https://www.mcgill.ca/neuro/)/[MIT](https://www.mit.edu/)  
Member - [BIDS](https://bids-specification.readthedocs.io/en/stable/), [ReproNim](https://www.repronim.org/), [Brainhack](https://brainhack.org/), [Neuromod](https://www.cneuromod.ca/), [OHBM SEA-SIG](https://ohbm-environment.org/), [UNIQUE](https://sites.google.com/view/unique-neuro-ai)  

<img align="left" src="https://raw.githubusercontent.com/G0RELLA/gorella_mwn/master/lecture/static/Twitter%20social%20icons%20-%20circle%20-%20blue.png" alt="logo" title="Twitter" width="32" height="20" /> <img align="left" src="https://raw.githubusercontent.com/G0RELLA/gorella_mwn/master/lecture/static/GitHub-Mark-120px-plus.png" alt="logo" title="Github" width="30" height="20" />   &nbsp;&nbsp;@peerherholz 

## Before we get started 2...
    
We're going to be working with a dataset from https://swcarpentry.github.io/shell-novice/data/shell-lesson-data.zip.

Download that file and unzip it on your Desktop!

## Goals

* learn basic and efficient usage of the shell for various tasks
    * navigating directories
    * file handling: copy, paste, create, delete

## What is the "shell"?

* The shell is a **command-line interface** (CLI) to your computer
    * This is in contrast to the **graphical user interfaces** (GUIs) that you normally use!
* The shell is _also_ a scripting language that can be used to automate repetitive tasks

### But what's this "bash shell"?

It's one of many available shells!

* `sh` - Bourne **SH**ell
* `ksh` - **K**orn **SH**ell
* `dash` - **D**ebian **A**lmquist **SH**ell
* `csh` - **C** **SH**ell
* `tcsh` - **T**ENEX **C** **SH**ell
* `zsh` - **Z** **SH**ell
* `bash` - **B**ourne **A**gain **SH**ell  <-- We'll focus on this one!

### WHY so many?


* They all have different strengths / weaknesses
* You will see many of them throughout much of neuroimaging software, too!
    * `sh` is most frequently used in FSL (FMRIB Software Library)
    * `csh`/`tcsh` is very common in FreeSurfer and AFNI (Analysis of Functional NeuroImages)

### So we're going to focus on the bash shell?

Yes! It's perhaps **the most common** shell, available on almost every OS:

* It's the default shell on most Linux systems
* It's the default shell in the Windows Subsytem for Linux (WSL)
* It's the default shell on Mac <=10.14
    * `zsh` is the new default on Mac Catalina (for licensing reasons 🙄)
    * But `bash` is still available!!

### Alright, but why use the shell at all?

Isn't the GUI good enough?

* Yes, but the shell is **very powerful**
* Sequences of shell commands can be strung together to quickly and reproducibly make powerful pipelines
* Also, you need to use the shell to accesss remote machines/high-performance computing environments (like Compute Canada or the local Goethe-Cluster)

**NOTE:** We will not be able to cover all (or even most) aspects of the shell today. 

But, we'll get through some _basics_ that you can build on going forward.

## The (bash) shell

Now, let's open up your terminal!

* **Windows**: Open the Ubuntu application
* **Mac/Linux**: Open the Terminal (Command + Space Bar / Ctrl + Alt + t)


When the shell is first opened, you are presented with a prompt, indicating that the shell is waiting for input:

```
$
```

The shell typically uses `$` as the prompt, but may use a different symbol.

**IMPORTANT:** When typing commands, either in this lesson or from other sources, **do not type the prompt**, only the commands that follow it!

### Am I using bash?

Let's check! You can use the following command to determine what shell you're using:

echo $SHELL

If that doesn't say something like `/bin/bash`,

- then simply type `bash`, press `Enter`, and try running the command again
- there might be other ways depending on your `OS/installation`, **please let us know**

**Note**: The `echo` command does exactly what its name implies: it simply echoes whatever we provide it to the screen!

(It's like `print` in Python / R or `disp` in MATLAB or `printf` in C or ...)

### What's with the `$SHELL`?

* Things prefixed with `$` in bash are (mostly) **environmental variables** 
    * All programming languages have variables!
* We can assign variables in bash but when we want to reference them we need to add the `$` prefix
* We'll dig into this a bit more later, but by default our shell comes with some preset variables
    * `$SHELL` is one of them!

Soooo, let's try our ~first~ second command in bash!

This command lists the contents of our current directory:

ls

What happens if we make a typo? Or if the program we want isn't installed on our computer?

Will the computer magically understand what we were trying to do?

ks

Nope! But you will get a (moderately) helpful error message 😁

### The cons of the CLI

* You need to know the names of the commands you want to run!
* Sometimes, commands are not immediately obvious
    * E.g., why `ls` over `list_contents`?

### Key Points

* A shell is a program whose primary purpose is to accept commands and run programs
* The shell’s main advantages are its high action-to-keystroke ratio, its support for automating repetitive tasks, and its capacity to access remote machines
* The shell’s main disadvantages are its primarily textual nature and how cryptic its commands and operation can be

## Navigating Files and Directories

* The **file system** is the part of our operating system for managing files and directories
* There are a lot of commands to create/inspect/rename/delete files + directories
    * Indeed these are perhaps the most common commands you'll be using in the shell!

### So where are we right now?

* When we open our terminal we are placed *somewhere* in the file system!
    * At any time while using the shell we are in exactly one place
* Commands mostly read / write / operate on files wherever we are, so it's important to know that!
* We can find our **current working directory** with the following command:

pwd

* Many bash commands are acronyms or abbreviations (to try and help you remember them).
    * The above command, `pwd`, is an acronym for "**p**rint **w**orking **d**irectory"

### OS-dependent paths

* The printed directory may look different depending on your operating system
    * Though if you're all on Linux / Mac / WSL it _should_ look something like the above...
* On Windows you may see something like: `C:\Users\grogu`
    * We'll be assuming the `/Users/grogu` notation for the rest of these examples!

### The file system
Let's take a look at an example file-system:
<img src="http://swcarpentry.github.io/shell-novice/fig/filesystem.svg" width="400px" style="margin-bottom: 10px;float: right">

* The top is the **root directory**, which holds the ENTIRE FILE SYSTEM. 
* Inside are several other directories:
    * `bin` contains some built-in programs
    * `data` is where we store miscellaneous data files
    * `Users` is where personal user directories are
    * `tmp` is for temporary storage of files
* Our current directory is inside `Users`!


### The file system
Let's take a look at an example file-system:
<img src="http://swcarpentry.github.io/shell-novice/fig/filesystem.svg" width="400px" style="margin-bottom: 10px;float: right">


#### The `/` character

* Refers to the root directory when it appears at the start of a path
* Is used as a separator between directories when it appears inside a path

### Inside `Users`
* The `Users` directory contains different folders for the different users on your computer
* If you are the only user on your computer then there is likely only one!
    * But shared computers can have multiple
* When you open a new terminal it defaults to your home directory (e.g., `/Users/nelle` or `/Users/grogu`)

<img src="http://swcarpentry.github.io/shell-novice/fig/home-directories.svg" width="400px" style="margin-bottom: 10px;float: right">


So let's remind ourselves how to see where we are and figure out what's in our directory:

pwd

ls

(Your results are likely different than this!)

`ls`, as we saw before, prints the contents of your **current working directory**. 

We can make it tell us a bit more information about our directory by providing an **option** to the `ls` command:

ls -F

This `-F` option adds a helpful marker to the end of each of the listed contents:

* A `/` indicates it's a directory
* A `@` indicates it's a link
* A `*` indicates it's an executable

(Your contents may also be color coded depending on your default options!)

Note: our home directory contains only **sub-directories**.

### General syntax of a shell command

Consider this command as a general example:

ls -F /

We have:

1. A **command** (`ls`), 
2. An **option** (`-F`), also called a **flag** or a **switch**, and
3. An **argument** (`/`)

#### Options (a.k.a. flags, switches)

* Options change the behavior of a command
* They generally start with either a `-` or `--`
* They are case sensitive!

For example, `ls -s` will display the size of the contents of the provided directory:

#cd /mnt/c/Users/my_user_name --> for the windows people
ls -s /Users/peerherholz/Desktop/shell-lesson-data/data

Whereas `ls -S` will sort the contents of the provided directory *by size*:

ls -S /Users/peerherholz/Desktop/shell-lesson-data/data

#### Options (cont'd)

What happens if I type an invalid option?

ls -j

Again, we get a (somewhat) helpful error message!

#### Arguments (a.k.a parameters)

* These tell the command what to operate on!
* They are only *sometimes* optional (as with `ls`)
    * In these cases, providing them will also change the behavior of the command!

ls

ls /Users/peerherholz/Desktop

#### Getting help

`ls` has **lots** of options. How can we find out about them?

Either `man ls` or `ls --help`!  
This will vary depending on: (1) the command and (2) your operating system!  
Generally try `man` first:

man ls

When you run that command in a terminal, your terminal will be turned into a page that can be navigated via:

* The `↑` / `↓` arrows (move up/down one line)
* The `B` / `Spacebar` keys (move up/down one page), or t
* The scroll bar (if you're lucky!)

To quit and get your "old" terminal back, press `q`!

### Combining options

You can use multiple options at the same time! If the options are single letters (like most of those with `ls`), you can combine them with the same `-` flag:

ls -sS /Users/peerherholz/Desktop/shell-lesson-data

### Exercise:

1. Using the `man ls`, find out how the `-l` option should change the behavior of `ls`.

ls -l /Users/peerherholz/Desktop/shell-lesson-data/data

### Exercise:

2. Now, find out how the `-h` option should change the behavior of `ls`.

ls -h /Users/peerherholz/Desktop/shell-lesson-data/data

### Exercise:

3. What will happen when we use them *together*?

ls -lh /Users/peerherholz/Desktop/shell-lesson-data/data

### Exercise:

* `ls` lists contents in alphabetical order by default.
* The `-t` option lists items by the time they were last modified (instead of alphabetically)
* The `-r` option lists the contents of a direcotry in reverse order

**Question**: What file is displayed **last** when you type `ls -tr`?

ls -tr

**Answer**: The most recently changed file is listed last!

### Exploring other directories

Providing an argument to `ls` lets us list the content of other directories (besides our current working directory):

ls -F /Users/peerherholz/Desktop

(For those of you on Mac + Linux: this should match what you *see* on your desktop! For those of you using the WSL: this will be a bit different, unfortunately.)

We can now do two things:

1. List the contents of one of the directories on our Desktop:

ls -F /Users/peerherholz/Desktop/shell-lesson-data

We can now do two things:

2. Actually *change* to a different directory, moving out of our home directory

#cd /mnt/c/Users/my_user_name/Desktop -> windows
cd /Users/peerherholz/Desktop

cd shell-lesson-data

cd data

The `cd` command (**c**hange **d**irectory) changes the **shell's idea** of what directory we're in.

The above commands will change us, one-by-one, into `Desktop`, then `data-shell`, then `data`.

Note the lack of output! This is normal for `cd`.

### Where are we now?

Let's check our **current working directory**:

pwd

And the contents of the directory (once more):

ls -F

### How do I get out of here?

We can go "down" into directories, but what about reversing that? What if I want to go back to `data-shell`?

cd data-shell

Nope! `cd` can only see *inside* your current directory. 

There's a special notation to move one directory up:

cd ..

Here, `..` refers to "the directory containing this one". This is also called the **parent** of the current directory.

Let's check that we are where we think we are:

pwd

### Seeing the unseen

`ls` is supposed to list the contents of our directory, but we didn't see `..` anywhere in the listings from before, right?

`..` is a special directory that is normally hidden. We can provide an additional argument to `ls` to make it appear:

ls -Fa

The `-a` argument (show **a**ll contents) will list ALL the contents of our current directory, including special and hidden files/directories, like:

* `..`, which refers to the parent directory
* `.`, which refers to the current working directory

### Hidden files

The last command also revealed a `.bash_profile` file:

ls -Fa

The `.` prefix is usually reserved for configuration files, and prevents them from cluttering the terminal when you use `ls`.

### `pwd`, `cd`, and `ls`

These are some of **the most** common commands you'll use in the shell! So let's learn a little more about them.

cd

`cd` optionally takes no arguments. But where does that land us?

pwd

In our home directory! This is *incredibly* useful if you've gotten lost.

Let's go back to the `data` directory:

cd Desktop/shell-lesson-data/data

pwd

We can string together paths with the `/` separator instead of changing one directory at a time!

### Relative versus absolute paths

We've been using **relative** paths to change directories and list their contents.

Relative here indicates that the path is **relative to your current working directory**.

The alternative is an **absolute** path, which includes the entire path starting at the root directory (`/`).

This is what's been printed with `pwd`:

pwd

We can provide absolute paths to our commands and they'll work, too:

cd /Users/peerherholz/Desktop/shell-lesson-data

pwd

ls -F

### Some helpful shortcuts

#### `~`

The shell will interpret the `~` tilde character as "your home directory". 

That is, `cd ~`, `cd`, and `cd /Users/peerherholz` will all get me to the same place!

cd ~

pwd

If you're unsure, you can determine what your home directory is via the environmental variable `$HOME` (like we did with `$SHELL` before!):

echo $HOME

#### `-`

The shell will interpret the `-` character as "wherever you were last".

Unlike `..`, which moves us "up" one directory, `-` will bring you BACK. 

We were just in `/Users/peerherholz/Desktop/data-shell` and then changed to `/Users/peerherholz` so `cd -` should bring us back:

cd -

### Absolute vs Relative Paths:

Starting from `/Users/amanda/data`, which of the following commands could Amanda use to navigate to her home directory, which is `/Users/amanda`?

1. `cd .`
2. `cd /`
3. `cd /home/amanda`
4. `cd ../..`
5. `cd ~`
6. `cd home`
7. `cd ~/data/..`
8. `cd`
9. `cd ..`

1. No: `.` refers to the **current working directory**
2. No: `/` refers to the **root directory**
3. No: Amanda's home directory is `/Users/amanda`
4. No: this goes up two levels, to `Users`
5. Yes: `~` refers to the home directory, which is `Users/amanda`.
6. No: this would navigate to the `hom` directory inside `Users/amanda/data` (if it exists)
7. Yes: unnecesarily complicated, but correct
8. Yes: this is a shortcut to go back to the user's home directory!
9. Yes: this goes up one directory, to `/Users/amanda`

### Relative Path Resolution:

Based on the following diagram, if `pwd` displays `/Users/thing`, what will `ls -F ../backup` display?

1. `../backup: No such file or directory`
2. `2012-12-01 2013-01-08 2013-01-27`
3. `2012-12-01/ 2013-01-08/ 2013-01-27/`
4. `original/ pnas_final/ pnas_sub/`

<img src="http://swcarpentry.github.io/shell-novice/fig/filesystem-challenge.svg" style="margin-bottom: 10px;float: right">

1. No: there *is* a directory `backup` in `Users`

2. No: this is the content of `/Users/thing/backup`, but `..` means we are one level up from that

3. No: for the same reason as (2)

4. Yes: `../backup` refers to `/Users/backup`

### `ls` Reading Comprehension

Using the filesystem diagram below, if `pwd` displays `/Users/backup`, and `-r` tells `ls` to display things in reverse (alphabetical) order, what command(s) will result in the following output:

```
pnas_sub/ pnas_final/ original/
```

1. `ls pwd`
2. `ls -rF`
3. `ls -rF /Users/backup`

<img src="http://swcarpentry.github.io/shell-novice/fig/filesystem-challenge.svg" style="margin-bottom: 10px;float: right">

1. No: `pwd` is not the name of a directory, it is a command.

2. Yes: `ls` without any arguments will list the contents of the current directory.

3. Yes: providing the absolute path of the directory will work.

### Key points


* The file system is responsible for managing information on the disk
* Information is stored in files, which are stored in directories (folders)
* Directories can also store other (sub-)directories, which forms a directory tree
* `cd path` changes the current working directory
* `ls path` prints a listing of a specific file or directory; `ls` on its own lists the current working directory.
* `pwd` prints the user’s current working directory
* `/` on its own is the root directory of the whole file system
* A relative path specifies a location starting from the current location
* An absolute path specifies a location from the root of the file system
* Directory names in a path are separated with `/` on Unix, but `\` on Windows
* `..` means "the directory above the current one"; `.` on its own means "the current directory"

## Working with Files and Directories

How do we actually _make_ new files and directories from the command line?

First, let's remind ourselves of where we are:

cd ~/Desktop/shell-lesson-data
pwd

ls -F

## Creating a directory

We can create new directories with the `mkdir` (**m**a**k**e **dir**ectory) command:

mkdir thesis

Since we provided a relative path, we can expect that to have been created in our current working directory:

ls -F

(You could have also opened up the file explorer and made a new folder that way, too!)

### Good naming conventions

1. Don't use spaces
2. Don't begin the name with `-`
3. Stick with letters, numbers, `.`, `-`, and `_`
    - That is, avoid other special characters like `~!@#$%^&*()`

### Creating a text file

Let's navigate into our (empty) `thesis` directory and create a new file:

cd thesis

We can make a file via the following command:

touch draft.txt

`touch` creates an **empty** file. We can see that with `ls -l`:

ls -l

### Moving files and directories

Let's start by going back to the `data-shell` directory:

cd ~/Desktop/shell-lesson-data

We now have a `thesis/draft.txt` file, which isn't very informatively named. Let's **m**o**v**e it:

mv thesis/draft.txt thesis/quotes.txt

The first argument of `mv` is the file we're moving, and the last argument is where we want it to go!

Let's make sure that worked:

ls thesis

Note: we can provide more than two arguments to `mv`, as long as the final argument is a directory! That would mean "move all these things into this directory".

Also note: `mv` is **quite dangerous**, because it will silently overwrite files if the destination already exists! Refer to the `-i` flag for "interactive" moving (with warnings!).

### More on `mv`

Note that we use `mv` to change files to a different directory (rather than just re-naming):

mv thesis/quotes.txt .

The `.` means "the current directory", so we _should_ have moved `quotes.txt` out of the `thesis` directory into our current directory.

Let's check that worked as expected:

ls thesis

ls quotes.txt

(Note: providing a filename to `ls` instead of a directory will list only that filename **if it exists**. Otherwise, it will throw an error.)

### Exercise: Moving files to a new folder

After running the following commands, Jamie realizes that she put the files `sucrose.dat` and `maltose.dat` into the wrong folder. The files should have been placed in the `raw` folder.

```bash
$ ls -F
 analyzed/ raw/
$ ls -F analyzed
fructose.dat glucose.dat maltose.dat sucrose.dat
$ cd analyzed
```

Fill in the blanks to move these files to the raw/ folder (i.e. the one she forgot to put them in):

```bash
$ mv sucrose.dat maltose.dat ____/____
```

```bash
mv sucrose.dat maltose.dat ../raw
```

Remember, the `..` refers to the parent directory (i.e., one above the current directory)

### Copying files and directories

The `cp` (**c**o**p**y) command is like `mv`, but copies instead of moving!

cp quotes.txt thesis/quotations.txt

ls quotes.txt thesis/quotations.txt

We can use the `-r` (**r**ecursive) flag to copy a directory and all its contents:

cp -r thesis thesis_backup

ls thesis thesis_backup

### Exercise: Renaming files

Suppose that you created a plain-text file in your current directory to contain a list of the statistical tests you will need to do to analyze your data, and named it: `statstics.txt`

After creating and saving this file you realize you misspelled the filename! You want to correct the mistake and remove the incorrectly named file. Which of the following commands could you use to do so?

1. cp statstics.txt statistics.txt
2. mv statstics.txt statistics.txt
3. mv statstics.txt .
4. cp statstics.txt .

1. No: this would create a file with the correct name but would **not** remove the incorrectly named file

2. Yes: this would rename the file!

3. No, the `.` indicates where to move the file but does not provide a new name.

4. No, the `.` indicates where to copy the file but does not provide a new name.

### Moving and Copying

What is the output of the closing `ls` command in the aequence shown below:

```bash
$ pwd
/Users/jamie/data
$ ls
proteins.dat
$ mkdir recombine
$ mv proteins.dat recombine
$ cp recombine/proteins.dat ../proteins-saved.dat
$ ls
```

1. `proteins-saved.dat recombine`
2. `recombine`
3. `proteins.dat recombine`
4. `proteins-saved.dat`

1. No: `proteins-saved.dat` is located at `/Users/jamie`

2. Yes!

3. No: `proteins.dat` is located at `/Users/jamie/data/recombine`

4. No, `proteins-saved.dat` is located at `/Users/jamie` 

### Removing files

Let's go back to `data-shell` and **r**e**m**ove the `quotes.txt` file we created:

cd ~/Desktop/shell-lesson-data

rm quotes.txt

The `rm` command deletes files. Let's check that the file is gone:

ls quotes.txt

### Deleting is **FOREVER** 💀💀

* The shell DOES NOT HAVE A TRASH BIN.
* You CANNOT recover files that have been deleted with `rm`
* But, you can use the `-i` flag to do things a bit more safely!
    * This will prompt you to type `Y` or `N` before every file that is going to be deleted.

### Removing directories

Let's try and remove the `thesis` directory:

rm thesis

`rm` only works on files, by default, but we can tell it to **r**ecursively delete a directory and all its contents with the `-r` flag:

rm -r thesis

Because **deleting is forever 💀💀**, the `rm -r` command should be used with GREAT CAUTION.

### Operations with multiple files and directories

Oftentimes you need to copy or move several files at once. You can do this by specifiying a list of filenames

#### Exercise: Copy with Multiple Filenames

(Work through these in the `data-shell/data` directory.)

In the example below, what does `cp` do when given several filenames and a directory name?

```bash
$ mkdir backup
$ cp amino-acids.txt animals.txt backup/
```


What does `cp` do when given three or more filenames?
```bash
$ ls
amino-acids.txt  animals.txt  backup/  elements/  morse.txt  pdb/  planets.txt  salmon.txt  sunspot.txt
$ cp amino-acids.txt animals.txt morse.txt
```

1. When given multiple filenames followed by a directory all the files are copied into the directory.

2. When give multiple filenames with no directory, `cp` throws an error:

```bash
cp: target morse.txt is not a directory
```

#### Using wildcards for accessing multiple files at once

`*` is a wildcard which matches zero or more characters.

Consider the `data-shell/molecules` directory:

ls molecules/*

This matches every file in the `molecules` directory.

ls molecules/*pdb

This matches every file in the `molecules` directory ending in `.pdb`.

ls molecules/p*.pdb

This matches all files in the `molecules` directory starting with `p` and ending with `.pdb`

#### Using wildcards for accessing multiple files at once (cont'd)

`?` is a wildcard matching exactly one character.

ls molecules/?ethane.pdb

This matches any file in `molecules` that has one character followed by `ethane.pdb`. Compare to:

ls molecules/*ethane.pdb

Which matches any file in `molecules` that ends in `ethane.pdb`.

#### Using wildcards for accessing multiple files at once (cont'd)

You can string wildcards together, too!

ls molecules/???ane.pdb

This matches and file in `molecules` that has any three characters and ends in `ane.pdb`

Wildcards are said to be "expanded" to create a list of matching files. This happens **before** running the relevant command. For example, the following command will fail:

ls molecules/*pdf

#### Exercise: List filenames matching a pattern

When run in the `molecules` directory, which `ls` command(s) will produce this output?

`ethane.pdb methane.pdb`

1. `ls *t*ane.pdb`
2. `ls *t?ne.*`
3. `ls *t??ne.pdb`
4. `ls ethane.*`

1. No: This will give `ethane.pdb methane.pdb octane.pdb pentane.pdb`

2. No: this will give `octane.pdb pentane.pdb`

3. Yes!

4. No: This only shows file starting with `ethane`

### Key points

* `cp old new` copies a file
* `mkdir path` creates a new directory
* `mv old new` moves (renames) a file or directory
* `rm path` removes (deletes) a file
* `*` matches zero or more characters in a filename, so `*.txt` matches all files ending in `.txt`
* `?` matches any single character in a filename, so `?.txt` matches `a.txt` but not `any.txt`
* The shell does not have a trash bin: once something is deleted, it’s really gone

## Summary

* The bash shell is very powerful!
* It offers a command-line interface to your computer and file system
* It makes it easy to operate on files quickly and efficiently (copying, renaming, etc.)
* Sequences of shell commands can be strung together to quickly and reproducibly make powerful pipelines

## Soapbox

* Bash is *fantastic* and you will (likely) find yourself using it a lot!
* However, for complex pipelines and programs we would _strongly_ encourage you to use a "newer" programming lanuage
    * Like Python, which we will also be discussed in this workshop!
* There are a number of reasons for this (e.g., better control flow, error handling, and debugging)

## References

There are lots of excellent resources online for learning more about bash:

* The GNU Manual is *the* reference for all bash commands: http://www.gnu.org/manual/manual.html
* "Learning the Bash Shell" book: http://shop.oreilly.com/product/9780596009656.do
* An interactive on-line bash shell course: https://www.learnshell.org/

## Finding Things

Oftentimes, our file system can be quite complex, with sub-directories inside sub-directories inside sub-directories.

What happens in we want to find one (or several) files, without having to type `ls` hundreds or thousands of times?

First, let's navigate to the `data-shell/writing` directory:

cd ~/Desktop/shell-lesson-data/writing

The directory structure of `data-shell/writing` looks like:

<img src="http://swcarpentry.github.io/shell-novice/fig/find-file-tree.svg">

Let's get our bearings with `ls`:

ls

Unfortunately, this doesn't list any of the files in the sub-directories. Enter `find`:

find .

Remember, `.` means "the current working directory". Here, `find` provides us a full list of the entire directory structure!

### Filtering `find`

We can add some helpful options to `find` to filter things a bit:

find . -type d

This will list only the **d**irectories underneath our current directory (incluing sub-directories).

Alternatively, we can list only the **f**iles with:

find . -type f

We can also match things by name:

find . -name *.txt

Why didn't this also get the other files??

Remember: wildcards are expanded BEFORE being passed to the command. So, we really want:

find . -name "*.txt"

### Executing with `find`

What if we want to perform some operation on the output of our `find` command? Say, list the file sizes for each file (as in `ls -lh`)?

We can do that with a bit of extra work:

find . -name "*.txt" -exec ls -lh {} \;

Note the **very funky syntax**:

* The `-exec` option means **exec**ute the following command,
* `ls -lh` is the command we want to execute,
* `{}` signifies where the output of `find` should go so as to be provided to the command we're executing, and
* `\;` means "this is the end of command we want to execute"

We can also "pipe" the output of `find` to the `ls -lh` command as follows:

ls -lh $( find . -name "*.txt" )

Here, the `$( )` syntax means "run this command first and insert it's output here", so `ls -lh` is provided the output of the `find . -name "*.txt"` command as arguments.