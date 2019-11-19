# Market-for-Renewables
 
A class in Python language is developed for modeling two settlement electricity markets with renewable energy resources in the congested networks!

This two-settlement market is introduced in the paper titled 'Competitive Market with Renewable Power Producers Achieves Asymptotic Social Efficiency' (link: https://ieeexplore.ieee.org/abstract/document/8586654) consisting of a Day-Ahead (DA) and a Real-Time (RT) market. The clearing process of this market is as follows:
1) In the DA market,
    a) The DA conventional generators submit their quadratic bidding curves to the ISO.
    b) The RPPs submit firm commitments for their power delivery at the future time of the RT market.
    c) Upon receiving these information, the ISO performs an optimal dispatch of the DA conventional generators to meet the load.
2) In the RT market,
        a) The RT conventional generators submit their quadratic bidding curves to the ISO.
        b) The RPPs’ actual generation are realized.
        c) The ISO computes the remaining difference between the generations and loads, and performs an optimal dispatch of the RT conventional generators to resolve the differences.
---------------------------------------------------------------------------------------------------------
----------- ----------- ----------- ----------- Examples ----------- ----------- ----------- -----------
As examples, the codes for the simulations of the paper titled 'On the Market Equilibria with RenewablePower Producers in Power Networks' is available too. These names of the files for these examples are: 'Find_NE', 'Fig_1a', 'Fig_1b', and 'Fig_1c'.

---------------------------------------------------------------------------------------------------------
----------- ----------- ----------- ----------- Description of The Class ----------- ----------- ------
The file containing the class is 'Network_Class.py'.

The class contains the following functions:
1. solve_DA_economic_dispatch_by_ISO(DA_commitments_of_RPPs):
        This module solves the DA optimal dispatch of the DA conventional generators gives the DA commitments of the RPPs.

2. solve_RT_economic_dispatch_by_ISO(Realizations_of_the_RPPs , generations_of_the_DA_generators ):
        This function solves the RT economic dipatch problem given the dispatch schedules of the DA conventional generators and the realizations of the power generation of the RPPs in the RT market.
        
3. find_the_NEs___case_same_DA_and_RT_cong_pttrns( ):
        This function finds the set of pure Nash equilibriums (NEs) in the two-settlement market. The players are the RPPs. The DA and RT conventional generators are assumed to submit their true quadratic cost functions, and these information is known to the RPPs. This function assumes that the RT congestion pattern is the same as the DA congestion pattern. Note that DA and RT congestion patterns for which the RPPs search through are given to the program when building the instance of the 'Network' class, however this function ignores the RT congestion patterns and simply assumes that each RT congestion pattern is the same as its corresponding DA congestion pattern.

4. find_the_NEs___case_different_DA_and_RT_cong_pttrns():
        This function finds the set of pure Nash equilibriums (NEs) in the two-settlement market. The players are the RPPs. The DA and RT conventional generators are assumed to submit their true quadratic cost functions, and these information is known to the RPPs. Note that DA and RT congestion patterns for which the RPPs search through are given to the program when building the instance of the 'Network' class.
        
5. find_DA_comtmnts_of_RPPs_for_curnt_DA_and_RT_cong_patterns(current_DA_congestion_pattern , current_RT_congestion_pattern):        
        This function finds the DA commitments of the RPPs that is a candidate for being a Nash equilibrium for the given pair of DA congestion pattern and RT congestion pattern.
        
6. social_optimum_calculations(self , num_of_scenarios , penalty_factor):
    This function solves a stochastic problem to find the social optimum solution and corresponding expected system cost for the two-settlement market.

7. Expected_System_Cost_at_NE(self , DA_Generations_at_NE , num_of_scenarios , penalty_factor = 5000):
    This function finds the expected system cost for a Nash equilibrium. Note that for a Nash equilibrium, there is a corresponding set of DA dispatches of the DA conventional generators. These DA dispatches are given to this function as the input parameter 'DA_Generations_at_NE'. This function is similar to the function 'social_optimum_calculations' except that in this function the DA dispatches of the DA conventional generators are fixed.

8. calc_susceptance_matrix():
    This function returns the susceptance matrix for a given netwrok.

9. calc_PTDF_matrix():
    This function returns the PTDF matrix for a netwrok.
        
        
        
        

