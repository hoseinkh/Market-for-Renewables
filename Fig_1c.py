####
# this code is for the calculation of the LMP Difference (Figure 1.c)!
#
#
#
#
#
#
#
import numpy as np
from tabulate import tabulate
from Network_Class import Network
import matplotlib.pyplot as plt
#
## ## ## ## ## ## ## ## ## INPUT DATA ## ## ## ## ## ## ## ## ## ## ## ## ##
#
#
index_of_slack_node = 0
#
#
#                                                                       line #
compact_data_of_branches = np.array( [ [ 0  ,  1  ,  40  , 0.06938  ] ,   # 0
                                       [ 0  ,  4  ,  40  , 0.05403  ] ,   # 1
                                       [ 1  ,  2  ,  40  , 0.16699  ] ,   # 2
                                       [ 1  ,  3  ,  48  , 0.05811  ] ,   # 3
                                       [ 1  ,  4  ,  56  , 0.09695  ] ,   # 4
                                       [ 2  ,  3  ,  56  , 0.06701  ] ,   # 5
                                       [ 3  ,  4  ,  80  , 0.05335  ] ,   # 6
                                       [ 3  ,  6  ,  56  , 0.10010  ] ,   # 7
                                       [ 3  ,  7  ,  80  , 0.15010  ] ,   # 8
                                       [ 4  ,  5  ,  48  , 0.09510  ] ,   # 9
                                       [ 5  , 10  ,  48  , 0.09498  ] ,   # 10
                                       [ 5  , 11  ,  88  , 0.12291  ] ,   # 11
                                       [ 5  , 12  ,  40  , 0.18615  ] ,   # 12
                                       [ 6  ,  7  ,  80  , 0.10010  ] ,   # 13
                                       [ 6  ,  8  ,  64  , 0.10010  ] ,   # 14
                                       [ 8  ,  9  ,  48  , 0.06181  ] ,   # 15
                                       [ 8  , 13  ,  40  , 0.12711  ] ,   # 16
                                       [ 9  , 10  ,  32  , 0.18205  ] ,   # 17
                                       [ 11 , 12  ,  32  , 0.05092  ] ,   # 18
                                       [ 12 , 13  ,  48  , 0.17093  ] ] ) #19
#
#
vector_of_nodal_demands = [0 , 43.4 , 68.4 , 41.6 , 15.2 , 22.4 , 20 , 50 , 59   , 18 , 27   , 32.2 , 27.6 , 29.8]
#
#
#
#
#
#
#
#
data_of_DA_conventional_generators = np.array([[ 7  , 0.06 , 3.51 , 44.1 , np.inf] ,
                                               [ 8  , 0.09 , 3.89 , 40.6 , np.inf] ,
                                               [ 11 , 0.08 , 2.15 , 105  , np.inf] ] )
#
data_of_RT_conventional_generators = np.array([[ 5 ,  0.24 , 9.35 , 135 , np.inf] ,
                                               [ 12 , 0.26 , 11.51 , 50  , np.inf] ])
#
#
#
# 
list_of_indices_of_allowed_lines_to_be_congested_in_DA_market = list([18])
list_of_num_of_allowed_lines_to_be_congested_in_DA_market = [1]
#
list_of_indices_of_allowed_lines_to_be_congested_in_RT_market = list([18])
list_of_num_of_allowed_lines_to_be_congested_in_RT_market = [1]
#
#
#
## ## ## ## ## ## ## ## ## ## ## ## 
num_of_samples = 500
#
#
list_of_number_of_divided_RPPs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#list_of_number_of_divided_RPPs = [2]
#
#
#
for i in range(0,len(list_of_number_of_divided_RPPs)):
    #
    counter = 0
    #
    ## ## ## Calculate the data of the RPPs
    data_of_RPPs_original = np.array([[ 4 , 70 , (70*0.15)**2 ] ,
                                      [11 , 50 , (50*0.15)**2 ]] )
    #
    r_correlation = 0.2
    num_of_smaller_RPPs_from_dividing_each_RPP = list_of_number_of_divided_RPPs[i]
    data_of_divided_RPPs , mean_of_divided_RPPs , cov_mtrx_of_divided_RPPs = Network.calc_mean_and_cov_mtrx_of_dividing_two_RPPs_to_smaller_ones( data_of_RPPs_original , r_correlation , num_of_smaller_RPPs_from_dividing_each_RPP)
    data_of_RPPs = data_of_divided_RPPs
    Cov_matrix_of_RPPs = cov_mtrx_of_divided_RPPs
    #
    #
    ## find NE
    My_Network = Network( compact_data_of_branches = compact_data_of_branches , index_of_slack_node =  index_of_slack_node , vector_of_nodal_demands = vector_of_nodal_demands , data_of_RPPs = data_of_RPPs , data_of_DA_conventional_generators =  data_of_DA_conventional_generators , data_of_RT_conventional_generators =  data_of_RT_conventional_generators , list_of_indices_of_allowed_lines_to_be_congested_in_DA_market = list_of_indices_of_allowed_lines_to_be_congested_in_DA_market , list_of_indices_of_allowed_lines_to_be_congested_in_RT_market = list_of_indices_of_allowed_lines_to_be_congested_in_RT_market , list_of_num_of_allowed_lines_to_be_congested_in_RT_market = list_of_num_of_allowed_lines_to_be_congested_in_RT_market , list_of_num_of_allowed_lines_to_be_congested_in_DA_market = list_of_num_of_allowed_lines_to_be_congested_in_DA_market , Cov_matrix_of_RPPs = Cov_matrix_of_RPPs )
    list_of_DA_commitments_at_NEs , list_of_congestions_at_NEs_in_DA_market , list_of_congestions_at_NEs_in_RT_market , list_of_DA_commitments_calculated_by_algorithm_ALL , list_of_congestions_in_DA_market_assumed_by_algorithm_ALL , list_of_congestions_in_DA_market_calculated_by_ISO_ALL , list_of_generation_of_DA_generators_ALL, list_of_expected_generation_of_RT_generators_ALL = My_Network.find_the_NEs___case_same_DA_and_RT_cong_pttrns()
    #
    DA_commitments_of_RPPs_at_current_NE = list_of_DA_commitments_at_NEs[0]
    #
    #
    Expected_RT_congestion_at_current_NE = list_of_congestions_at_NEs_in_RT_market[0]
    #
    if ('LMP_RT_avg' in locals()) or ('LMP_RT_avg' in globals()):
        5 + 6 # do nothing
    else:
        LMP_RT_avg = np.zeros((My_Network.num_of_nodes , len(list_of_number_of_divided_RPPs)))
        LMP_Diff__DA_and_Expected_RT = np.zeros((My_Network.num_of_nodes , len(list_of_number_of_divided_RPPs)))
    #
    #
    nodal_injections_temp, LMP_DA_temp, _2, _3 = My_Network.solve_DA_economic_dispatch_by_ISO(DA_commitments_of_RPPs_at_current_NE)
    #
    DA_generations = nodal_injections_temp[[int(temp_ttt) for temp_ttt in My_Network.data_of_DA_conventional_generators[:, 0].reshape(1 , My_Network.num_of_DA_conv_generators)[0, :].tolist()]]
    #
    #
    LMP_RT_sum_over_all_scenarios = np.zeros((My_Network.num_of_nodes , 1)).reshape((My_Network.num_of_nodes , 1))
    #
    check_result_of_DA_OPF = sum(DA_generations) - sum(vector_of_nodal_demands) + sum(DA_commitments_of_RPPs_at_current_NE)
    #
    samples = np.array(My_Network.generate_samples_for_the_case_of_multivariate_normal_distributions(num_of_samples))
    #
    for j in range(0,num_of_samples):
        current_realization_of_RPPs = samples[j,:]
        _1, LMP_RT_temp, list_of_lines_congested_or_not, DC_power_flow_has_feasible_solution = My_Network.solve_RT_economic_dispatch_by_ISO(current_realization_of_RPPs, DA_generations)
        #
        #
        if (DC_power_flow_has_feasible_solution != False):  # this means that  <<list_of_lines_congested_or_not>> is empty
            jj = 5+6
            if (sum([abs(int(ttt_temp)) for ttt_temp in (np.array(list_of_lines_congested_or_not) - np.array(Expected_RT_congestion_at_current_NE)).tolist()]) == 0):
                counter = counter + 1
                LMP_RT_sum_over_all_scenarios = np.add(LMP_RT_sum_over_all_scenarios, LMP_RT_temp.reshape((My_Network.num_of_nodes , 1)))
                
            else:
                pass
            #
        else:
            pass
        #
    #
    LMP_RT_avg[:,i] = (LMP_RT_sum_over_all_scenarios / counter).reshape(My_Network.num_of_nodes , )
    LMP_Diff__DA_and_Expected_RT[:,i] = np.subtract(LMP_DA_temp , LMP_RT_avg[:,i])
#
#
#
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[0,:].tolist(), label='node 1',
         linewidth = 1., 
         color = 'b', 
         linestyle = '--'
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         #marker = 'd',
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[1,:].tolist(), label='node 2', 
         linewidth = 1., 
         color = 'tab:orange', 
         linestyle = '--'
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         #marker = 'd',
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[2,:].tolist(), label='node 3', 
         linewidth = 1., 
         color = 'c', 
         linestyle = '--'
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         #marker = 'd',
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[3,:].tolist(),  label='node 4',
         linewidth = 1., 
         color = 'k', 
         linestyle = '--'
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         #marker = 'd',
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[4,:].tolist(),  label='node 5',
         linewidth = 1., 
         color = 'r', 
         linestyle = '--'
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         #marker = 'd',
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[5,:].tolist(),  label='node 6',
         linewidth = 1., 
         color = 'b', 
         markersize = 6, 
         markeredgewidth = 1,
         markerfacecolor = '.75',
         markeredgecolor = 'b',
         marker = 'x',
         markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[6,:].tolist(),  label='node 7',
         linewidth = 1., 
         color = 'tab:orange', 
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         #marker = 'd',
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[7,:].tolist(),  label='node 8',
         linewidth = 1., 
         color = 'k', 
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         #marker = 'd',
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[8,:].tolist(),  label='node 9',
         linewidth = 1., 
         color = 'r', 
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         marker = 10,
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[9,:].tolist(),  label='node 10',
         linewidth = 1., 
         color = 'b', 
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         marker = 10,
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[10,:].tolist(),  label='node 11',
         linewidth = 1., 
         color = 'g', 
         linestyle = '--'
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         #marker = 'd',
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[11,:].tolist(),  label='node 12',
         linewidth = 1., 
         color = 'k', 
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         marker = 10,
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[12,:].tolist(),  label='node 13',
         linewidth = 1., 
         color = 'g', 
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         marker = 'x',
         #markevery = 1
         )
#
plt.plot(list_of_number_of_divided_RPPs, LMP_Diff__DA_and_Expected_RT[13,:].tolist(),  label='node 14',
         linewidth = 1., 
         color = 'g', 
         #markersize = 6, 
         #markeredgewidth = 1,
         #markerfacecolor = '.75',
         #markeredgecolor = 'r',
         marker = 10,
         #markevery = 1
         )
#
#plt.ylim(0.6, 1.05)
plt.xlabel('Total Number of Competitive RPPs')
plt.ylabel('LMPs Difference = (DA LMP) - (Expected RT LMP)')
plt.legend(loc="upper left", bbox_to_anchor=[0, 0.3], ncol=4)
plt.show()



