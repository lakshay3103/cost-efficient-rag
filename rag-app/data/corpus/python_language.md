# Python (Programming Language)

Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.

## History

Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands. It was released in 1991 as Python 0.9.0. Python 2.0 was released in 2000, and Python 3.0 was released in 2008. Python 3.0 was a major revision that was not completely backward-compatible with Python 2. Python 2 reached its official end-of-life on January 1, 2020, meaning it no longer receives security patches or updates.

Guido van Rossum named the language after the British comedy group Monty Python, not the snake.

## Design Philosophy

Python's design philosophy is summarized in a document called "The Zen of Python" (PEP 20), written by Tim Peters. Key principles include "explicit is better than implicit" and "readability counts." Python uses whitespace indentation to delimit code blocks, rather than curly braces as used in languages like C or Java.

## Typing System

Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented, and functional programming. Python uses duck typing, meaning an object's suitability is determined by the presence of certain methods and properties rather than the type of the object itself.

## Implementations

The reference implementation of Python is called CPython, written in C. Other implementations include PyPy (which uses a just-in-time compiler for speed), Jython (which runs on the Java Virtual Machine), and IronPython (which targets the .NET framework). CPython is the most widely used implementation.

## Package Management

Python's standard package installer is called pip, which stands for "Pip Installs Packages." The Python Package Index (PyPI) is the official third-party software repository for Python, hosting hundreds of thousands of packages as of the mid-2020s.

## Performance Characteristics

CPython uses a Global Interpreter Lock (GIL), which allows only one thread to execute Python bytecode at a time within a single process. This means CPython-based multi-threaded programs do not achieve true parallelism for CPU-bound tasks, although I/O-bound tasks can still benefit from threading. Multiprocessing, using separate processes rather than threads, is a common workaround for CPU-bound parallelism in Python.

## Popularity

Python has consistently ranked among the most popular programming languages in surveys such as the TIOBE Index and the Stack Overflow Developer Survey. It is widely used in web development, data science, machine learning, scientific computing, automation, and scripting.
