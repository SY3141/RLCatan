# RLcatan

Developer Names: Rebecca Di Filippo, Sunny Yao, Matthew Cheung, Jake Read

Date of project start: September 15th 2025

## About our project

The project involves creating an AI-enabled digital twin for real-time decision support in the board game Settlers of Catan. Students will develop a reinforcement learning agent that can learn to play the game through a simulator or API, providing both in-game advice and post-game analysis. The digital twin will observe a physical tabletop game using sensors or cameras, process the game state with computer vision, and provide recommendations to players in real time. The project also includes visualizing the game state, and optionally incorporating human trade behavior modeling through personas, smart glasses for alternative viewpoints, and large language models to explain alternative strategies.

## Organization
The folders and files for this project are as follows:

docs - Documentation for the project
refs - Reference material used for the project, including papers
src - Source code
test - Test cases
etc.


## Installation

Windows CMD

1. `cd src`
2. `python -m venv venv` 
3. `venv\Scripts\activate`
4. pip install -e .
5. `pip install -e .[web,gym,dev]`
6. `docker compose up`


macOS CMD

1. `cd src`
2. `python -m venv venv` 
3. `source venv/bin/activate`
4. `pip install -e .`
4. `pip install -e '.[web,gym,dev]'`
5. `docker-compose up`


## Simulations

For 1v1 rules simulations, pass the arguments below
1. `cd src`
2. `venv\Scripts\activate` 
3. `catanatron-play --num 1 --players AB:2:True,AB:2:True --config-vps-to-win 15 --config-discard-limit 9`

Optional
Create a replay link to view a CLI game --step-db
`catanatron-play --num 1 --players AB:1:True,ABPP:1:True --config-vps-to-win 15 --config-discard-limit 9 --step-db`

To test Placement Player
`catanatron-play --num 1 --players AB:1:True,PP --config-vps-to-win 15 --config-discard-limit 9`

To test placement on alphabetaPlayer
`catanatron-play --num 1 --players AB:1:True,ABPP:1:True --config-vps-to-win 15 --config-discard-limit 9`

To test PPObot
`catanatron-play --num 1 --players PPOP,AB:1:True --config-vps-to-win 15 --config-discard-limit 9`

## Deep Learning Training
Navigate to training folder under \src\rlcatan\training
Run
`python3 looped_trainer.py -runs 5 -iter 1000000`
Iter is the number of training steps, runs is how many times it is trained for those iterations