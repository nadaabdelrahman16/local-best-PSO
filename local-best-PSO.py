import numpy as np
import matplotlib.pyplot as plt
import copy

# This is the problem we need to solve:
# $$ Minimize: $$  $$ f(x, y) = (x + 2y-7) ^ 2 + (2x+y-5) ^ 2 $$
#
# $$ -10 \leqslant
# x, y \leqslant
# 10 $$


def initpop(npop, x_max, x_min, v_max, dim):
    # Use this function to generate the initial population for the PSO
    # npop: population size
    # x_max: the upper limit for each decision variable (positions). [10,12]
    # x_min: the lower limit for each decision variable (positions). [1,2]
    # v_max: the upper limit for each decision variable (velocity). [2,4]
    # consider that the lower limit of the speed is equal to negative the upper limit
    # dim: number of decision variables
    x_id = np.zeros((npop, dim), dtype=float)
    v_id = np.zeros((npop, dim), dtype=float)
    for i in range(dim):
        x_id[:, i] = np.random.uniform(x_min[i], x_max[i], (npop))
        v_id[:, i] = np.random.uniform(-1 * v_max[i], v_max[i], (npop))

    return x_id, v_id


# x_id: the initial positions. Array of arrays of npop*dim
# v_id: the initial velocities. Array f arrays of npop*dim

def fitCalc(x_i):
    # Use this function to calculate the fitness for the particle
    # The function is Min z= (x+2y-7)^2 + (2x+y-5)^2
    # x_i: single particle position
    fitness = (x_i[0] + 2 * x_i[1] - 7) ** 2 + (2 * x_i[0] + x_i[1] - 5) ** 2
    return fitness
    # fitness: the fitness value of a signle particle.



def updatePid(x_i, x_fitness, p_i, particle_bestFit):
    # Use this function to find single particle best position (particle own history)
    # x_i: single particle position.
    # p_i: the particle best position across all the previous generations.
    # particle_best: particles best fintess values across all the previouse generations.
    if (x_fitness < particle_bestFit):
        p_i = copy.deepcopy(x_i)

    return p_i
    # pi: the particle best position.

def updatePgd(p_i, particle_bestFit, p_g, global_bestFit):
    # Use this function to find the best position in the population
    # p_i: a single particle best position
    # particle_bestFit: a particle fitness value associated to p_i.
    # p_g: a vector of 1*dim of representing the best position in the population across all the previouse generations
    # global_bestFit: fitness value associated to the p_g

    if (particle_bestFit < global_bestFit):
        p_g = copy.deepcopy(p_i)
        global_bestFit = particle_bestFit

    return p_g, global_bestFit


# p_g: the best position in the population.
# global_bestFit: the best fitness in the population.



def updateVidXid(p_i, p_g, x_i, v_i, c_cog, c_soc,dim):  # Use this function to calculate new particle velocity and position
    # p_i: the particle best position across all the previouse generations.
    # p_g: a vector of 1*d of the best position in the population across all the previouse generations
    # x_i: single particle position.
    # v_i: single particle velocity.
    # c_cog: cognitive component accerlaration constant
    # c_soc: social component accerlaration constant

    r_cog = np.random.random(dim)
    r_soc = np.random.random(dim)
    v_i = np.array(v_i) + (c_cog * np.multiply(r_cog, np.subtract(p_i, x_i))) + (
                c_soc * np.multiply(r_soc, np.subtract(p_g, x_i)))
    x_i = np.array(x_i) + v_i

    return x_i, v_i


def PSO(numItr, npop, x_max, x_min, v_max, dim, c_cog, c_soc):
    # Use this function to put all the PSO algorithm together for number of iterations
    # numItr: number of iterations.(generations)
    # npop: population size
    # x_max: the upper limit for each decision variable (positions). [10,12]
    # x_min: the lower limit for each decision variable (positions). [1,2]
    # v_max: the upper limit for each decision variable (velocity). [2,4]
    # c_cog: cognitive constant (c1)
    # c_soc: social constant (c2)
    # dim: the number of decision variable.
    # numParticel: number or particel in each Neighborhood
    # P : best position for all particles
    # best_hist: history of global fitness for each iteration
    # best_p_g: history of best position for each global in each iteration
    # hist_local_bestFit : history of local best fitness in each neighborhood for same generation (iteration)

    # Intialize
    numParticel=3
    numNeighborhood=int(npop/numParticel)
    x, v = initpop(npop, x_max, x_min, v_max, dim)
    # X , V : after division x,v to neighborhoods
    X=np.vsplit(x,numNeighborhood)
    V=np.vsplit(v,numNeighborhood)
    P=np.empty((npop, dim))
    best_hist = np.zeros(numItr, dtype=float)
    p_g = np.zeros(numNeighborhood)

    for iteration in range(numItr):

        hist_local_bestFit = np.zeros(numNeighborhood, dtype=float)
        hist_p_g = np.zeros((numNeighborhood, dim), dtype=float)

        if iteration==0:
            # at first iteration intialize P same as initial pop as best position by (deep copy )
            P=copy.deepcopy(X)
        else:
            # The memory of the previous best position for all particles in each Neighborhood
            P=P

        for i in range(numNeighborhood):
              local_bestFit = 100000000000
              for j in range(numParticel):
                P[i][j] = updatePid(X[i][j], fitCalc(X[i][j]), P[i][j], fitCalc(P[i][j]))
                # enables the particles to do local search in swarm (in neighborhood)
                p_g, local_bestFit = updatePgd(P[i][j], fitCalc(P[i][j]), p_g, local_bestFit)

              hist_p_g[i] = p_g
              hist_local_bestFit[i] = local_bestFit


        # min local bestfitness for all neighborhood be global best fitness for iteration
        best_hist[iteration]=np.min(hist_local_bestFit)
        for i in range(numNeighborhood):
              for j in range(numParticel):
               X[i][j], V[i][j] = updateVidXid(P[i][j], hist_p_g[i], X[i][j], V[i][j], c_cog, c_soc, dim)


    global_bestFit=best_hist[numItr-1]
    # p_gCalc : calculate fitness for all array hist_p_g in last generation and choose that match global fitness to last generation
    p_gCalc=np.zeros(len(hist_p_g))
    for i in range(len(hist_p_g)):
     p_gCalc[i]=fitCalc(hist_p_g[i])
    index=np.where(p_gCalc==np.min(p_gCalc))
    p_g=hist_p_g[index[0][0]]

    return p_g, global_bestFit, best_hist
    # p_g: the position with the best fitness in the final generation.
    # global_bestFit: value associated to p_g



numItr =200
npop = 60
x_max = [10, 10]
x_min = [-10, -10]
v_max = [8, 8]
dim = 2
c_cog = 1.72
c_soc = 1.72

p_g, global_bestFit, best_hist = PSO(numItr, npop, x_max, x_min, v_max, dim, c_cog, c_soc)



plt.plot(best_hist)
plt.xlabel("Iterations")
plt.ylabel("Fitness value")
plt.show()
print("global bestFit ",global_bestFit)
print("best fitness in the final generation ",p_g)


# The social network employed by the gbest PSO reflects the star topology , The local best PSO, uses a ring social network topology (that improves performance PSO)
# where smaller particles are defined for each neighborhood.
# where information exchanged within the neighborhood of the particle, reflecting local knowledge of the environment
# for neighborhoods. The local best position will also be referred to as the neighborhood best position
#  Selection of neighborhoods is done randomly devision npop to neighborhoods , each neighborhood have number of particles (same as intialize in global but devision to neighborhoods )
# dealing with each neighborhood in each iteration as global
#  devision particles, It helps to promote the spread of information regarding good solutions to all particles, irrespective of their current location in the search space.
# All particles then move towards some quantification of what is believed to be a better position by update it (function updateVidXid )
# shows the best so far solution among all the solutions
# for each iteration , each neighborhood find local bestfitness then , global bestfitness for iteration is min local in this generation
# result of plot measures how close the corresponding solution is to the optimum (min problem)
# So i didn't need to change in any function of global implementation Except PSO function
#i noticed that function updatePid didn't work correct in gobal and local it take x_i same as p_i even if p[i] was't best position across all the previous generations
#i fixed that issue and debug it accurately
# i noticed also this problem can Terminate when a number of iterations less than 50 iteration ,no improvement is observed over a number of iterations