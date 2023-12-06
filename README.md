# Sudoku recognition and solving

## About

A small application made for the first semester "Project Practice" course.
Was only tested on Linux, other platforms probably wont't work.

## Provisional CNN notice

The neural network is very crippled and will not correctly recognize most 
of the sudokus given, especially if the image is not perfectly clear. We 
are currently working on solution to this problem.

## Linux usage

1. Clone the repository with

    `git clone https://github.com/hezea/sudoku_bot.git`

2. Open repository folder:

    `cd sudoku_bot`

3. Set up virtual environment:

    `./setup.sh`

    1. It *might* be necessary to change permissions of the executables first:

        `chmod u+x setup.sh`

        `chmod u+x run.sh`

        `chmod u+x train.sh`

4. (OPTIONAL) If you wish to train the model yourself, you can run:

    `./train.sh`

    This will overwrite the existing model in `models/mymodel.keras`
    which is the default location used by the application.

5. Run the application:

    `./run.sh`

## To-do

1. Telegram integration
2. Simple recursive solver