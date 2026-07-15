# Trajectory Generation Using Motion Primitives
## Overview
In this project, I explore approaches for trajectory generation for robot arms using movement primitives / imitation learning as a way to adapt recorded movement trajectories to a given set of environmental / physical conditions.

## Introduction
Generating trajectories plays a key role in deploying robots in real-world environments.
Often, tasks may require specific trajectory shapes which are generally the same but need adapting to specific environmental and physical conditions.
In order to generate a trajectory that satisfies a variable set of conditions regarding boundaries, temporal evolution, trajectory altering based on physical restrictions or obstacles, it might be necessary to use multiple approaches to achieve a more general solution to the problem at hand.

## Key Goals
1. Implementing different approaches for testing.
2. Exploring potential and limitations of each approach for the application.
3. Finding / developing a flexible solution for the application.

## Requirements
1. variable start and end-point
2. variable time of the complete movement
3. prerecorded movement to shape the spatial and temporal progression
4. timed via points for precise control of temporal and spatial progression
5. constraints for obstacle avoidance (e.g. stay at least 5cm away from this point; stay within 5cm from this point while moving)
6. online changes to constraints

## Preliminary System Architechture For Generating Trajectories
<img src="media/System%20Diagram%20For%20Trajectory%20Generation.png" alt="System Architecture" width="80%"/>

## Approaches
The following approaches are being considered for generating trajectories that can be varied as defined in the requirements section.
- Classical DMP
- ProMP
- ProDMP
- Trajectory Optimization
- MCP / MPPI
- Control Barrier Functions
- ...

A solution that might yield results that fulfill the requirements listed above is most likely a hybrid one, consisting of a trajectory generator in combination with an optimization step in order to fulfil the known constraints.
When constraints change, another optimization step or even replanning the trajectory might be necessary.

## Literature
