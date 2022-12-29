""" 
remeber to change the file saving format
"""
import numpy as np
import argparse
from modules.simulation import Simulation
from modules.utils import make_whip_downwards
from operator import itemgetter
import random

def my_parser( ):
    # Argument Parsers
    parser = argparse.ArgumentParser( description = 'Parsing the arguments for running the simulation' )
    #parser.add_argument( '--version'     , action = 'version'     , version = Constants.VERSION )
    parser.add_argument( '--start_time'  , action = 'store'       , type = float ,  default = 0.0,                   help = 'Start time of the controller'                                                      )
    parser.add_argument( '--model_name'  , action = 'store'       , type = str   ,  default = '2D_model_w_whip_drl_nlopt' ,    help = 'Model name for the simulation'                                                     )
    parser.add_argument( '--ctrl_name'   , action = 'store'       , type = str   ,  default = 'joint_imp_ctrl',      help = 'Model name for the simulation'                                                     )
    parser.add_argument( '--cam_pos'     , action = 'store'       , type = str   ,                                   help = 'Get the whole list of the camera position'                                         )
    parser.add_argument( '--mov_pars'    , action = 'store'       , type = str   ,                                   help = 'Get the whole list of the movement parameters'                                     )
    parser.add_argument( '--target_type' , action = 'store'       , type = int   ,                                   help = 'Save data log of the simulation, with the specified frequency'                     )
    parser.add_argument( '--opt_type'    , action = 'store'       , type = str   ,  default = "nlopt" ,              help = '[Options] "nlopt", "ML_DDPG", "ML_TD3" '                                           )
    parser.add_argument( '--print_mode'  , action = 'store'       , type = str   ,  default = 'normal',              help = 'Print mode, choose between [short] [normal] [verbose]'                             )

    parser.add_argument( '--target_idx'  , action = 'store'       , type = int   ,  default = 1,                     help = 'Index of Target 1~6'                                                               )

    parser.add_argument( '--print_freq'  , action = 'store'       , type = int   ,  default = 10      ,              help = 'Specifying the frequency of printing the data.'                                    )
    parser.add_argument( '--save_freq'   , action = 'store'       , type = int   ,  default = 60      ,              help = 'Specifying the frequency of saving the data.'                                      )
    parser.add_argument( '--vid_speed'   , action = 'store'       , type = float ,  default = 1.      ,              help = 'The speed of the video. It is the gain of the original speed of the video '        )

    parser.add_argument( '--record_vid'  , action = 'store_true'  , dest = "is_record_vid"  ,                        help = 'Record video of the simulation,  with the specified speed'     )
    parser.add_argument( '--save_data'   , action = 'store_true'  , dest = "is_save_data"   ,   default=False,       help = 'Save the details of the simulation'                            )
    parser.add_argument( '--vid_off'     , action = 'store_true'  , dest = "is_vid_off"     ,   default=False,       help = 'Turn off the video'                                            )
    parser.add_argument( '--run_opt'     , action = 'store_true'  , dest = "is_run_opt"     ,                        help = 'Run optimization of the simulation'                            )
    parser.add_argument("--is_oiac", default=False, type=bool, help="OIAC or constant control")
    parser.add_argument("--opt_name", default='GA')
    return parser



class Gene:
    def __init__(self, **data) :
        self.__dict__.update(data)
        self.size=len(data['data'])# length of gene

class GA:
    def __init__(self, parameter) :# parameter=[CXPB, MUTPB, NGEN, POPSIZE]
        self.parameter=parameter
        self.CXPB=self.parameter[0]
        self.MUTPB=self.parameter[1]
        
        pop=[]# action sets
        self.N=self.parameter[2]
        self.popsize=self.parameter[3]
        
        # import simulation class
        parser = my_parser()
        self.args, unknown = parser.parse_known_args()
        self.my_sim=Simulation(self.args)
        self.my_sim.set_camera_pos()
        self.low=self.my_sim.action_space_low
        self.up=self.my_sim.action_space_high
        self.n = self.my_sim.n_act
        # assume the max dist is 5
        self.maxdist=5 
        for i in range (self.popsize):
            mov_arrs=self.my_sim.gen_action()
            self.sim_init(mov_arrs)
            s,r,done=self.my_sim.run(mov_arrs)
            pop.append({'Gene': Gene(data=mov_arrs),'fitness': (self.maxdist-s)/self.maxdist})
            self.my_sim.reset()
            print("ini_iteration:",i," fitness:",(self.maxdist-s)/self.maxdist)
        self.pop=pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population

    def sim_init(self,action):

        init_cond = { "qpos": action[ :self.n ] ,  "qvel": np.zeros( self.n ) }
        self.my_sim.init( qpos = init_cond[ "qpos" ], qvel = init_cond[ "qvel" ] )
        make_whip_downwards( self.my_sim )
        self.my_sim.forward( )

    def evaluate(self,mov_arrs):

        self.sim_init(mov_arrs)
        state,r,done=self.my_sim.run(mov_arrs)
        self.my_sim.reset()
        return (self.maxdist-state)/self.maxdist

    def selectBest(self,pop):

        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)          # from large to small, return a pop
        return s_inds[0]

    def selection(self,individual, k):
        # roulette random method 
        s_inds=sorted(individual, key=itemgetter("fitness"),reverse=True)
        sum_fits=sum(ind["fitness"] for ind in individual )
        chosen=[]

        for i in range(k):
            u=random.random()*sum_fits
            sum_ =0
            for ind in s_inds:
                sum_ +=ind["fitness"]
                if sum_ >= u:
                    chosen.append(ind)
                    break
        chosen=sorted(chosen, key=itemgetter("fitness"), reverse=False)
        return chosen

    def crossoperate(self,offspring):

        dim = len(offspring[0]['Gene'].data)

        geninfo1=offspring[0]['Gene'].data
        geninfo2=offspring[1]['Gene'].data
        pos1 = random.randrange(1, dim)  # select a position in the range from 0 to dim-1,
        pos2 = random.randrange(1, dim)

        newoff1 = Gene(data=[])  # offspring1 produced by cross operation
        newoff2 = Gene(data=[])  # offspring2 produced by cross operation
        temp1 = []
        temp2 = []
        for i in range(dim):
            if min(pos1, pos2) <= i < max(pos1, pos2):
                temp2.append(geninfo2[i])
                temp1.append(geninfo1[i])
            else:
                temp2.append(geninfo1[i])
                temp1.append(geninfo2[i])
        newoff1.data = temp1
        newoff2.data = temp2
    
        return newoff1, newoff2

    def mutation(self,crossoff):

        dim = len(crossoff.data) 
        pos = random.randrange(0, dim)  # chose a position in crossoff to perform mutation.
        crossoff.data[pos] = np.random.choice(np.linspace(self.low[pos], self.up[pos],1000),1,replace=True)[0]
        return crossoff

    def GA_main(self):
        self.best=[]
        popsize=self.popsize
        print("Start of evolution")
    # Begin the evolution
        for g in range(self.N):
            print("############### Generation {} ###############".format(g))
            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, popsize)
            nextoff = []
            while len(nextoff) != popsize:
                # Apply crossover and mutation on the offspring
                # Select two individuals
                offspring = [selectpop.pop() for _ in range(2)]
                if random.random() < self.CXPB:  # cross two individuals with probability CXPB
                    crossoff1, crossoff2 = self.crossoperate(offspring)
                    if random.random() < self.MUTPB:  # mutate an individual with probability MUTPB
                        muteoff1 = self.mutation(crossoff1)
                        muteoff2 = self.mutation(crossoff2)
                        fit_muteoff1 = self.evaluate(muteoff1.data)  # Evaluate the individuals
                        fit_muteoff2 = self.evaluate(muteoff2.data)  # Evaluate the individuals
                        nextoff.append({'Gene': muteoff1, 'fitness': fit_muteoff1})
                        nextoff.append({'Gene': muteoff2, 'fitness': fit_muteoff2})
                    else:
                        fit_crossoff1 = self.evaluate(crossoff1.data)  # Evaluate the individuals
                        fit_crossoff2 = self.evaluate(crossoff2.data)
                        nextoff.append({'Gene': crossoff1, 'fitness': fit_crossoff1})
                        nextoff.append({'Gene': crossoff2, 'fitness': fit_crossoff2})
                else:
                    nextoff.extend(offspring)
    
            # The population is entirely replaced by the offspring
            self.pop = nextoff
            # print("pop {},{},\n{}".format(g, self.pop[0]['Gene'].data,self.pop[1]['Gene'].data))
                #self.pop[2]['Gene'].data,self.pop[3]['Gene'].data, self.pop[4]['Gene'].data, self.pop[5]['Gene'].data)) ,\n{},\n{},\n{},\n{}
    
            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
    
            best_ind = self.selectBest(self.pop)
    
            if best_ind['fitness'] >self.bestindividual['fitness']:
                self.bestindividual = best_ind
            #self.best.append((1-(self.bestindividual['fitness']))*self.maxdist)# transfer the normal distance
            print("Best individual found is {}, its fitness is {:.3f}".format(self.bestindividual['Gene'].data,
                                                        self.bestindividual['fitness']))
            print("  Max fitness of current pop: {}".format(max(fits)))
            self.best.append((1-(max(fits)))*self.maxdist)# current best transfer the normal distance
            np.save(f'./GA_res/GA_{self.args.is_oiac}_result.npy',self.best)
            if self.bestindividual['fitness']>=0.994:# 0.994=(5-0.03)/5 0.03 is threshold
                break
        print("------ End of (successful) evolution ------")
        #np.save(f'GA_result.npy',self.best)

if __name__=="__main__":

    CXPB, MUTPB, NGEN, popsize = 0.85, 0.1, 1000, 10 # popsize must be even number
    parameter=[CXPB, MUTPB, NGEN, popsize]
    run= GA(parameter)
    run.GA_main()
