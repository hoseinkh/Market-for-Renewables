####
# This code finds the NE for searching up to 1 line congested in the DA market!
#
#
#
#
import numpy as np
from tabulate import tabulate
from Network_Class import Network
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
RPP_Penetr = 0.10;
#
data_of_RPPs_original = np.array([[ 4 , 70 , (70*0.15)**2 ] ,
                                  [11 , 50 , (50*0.15)**2 ]] )
#
#
r_correlation = 0.2
#
# Calculate the data of the RPPs
num_of_smaller_RPPs_from_dividing_each_RPP = 1
data_of_divided_RPPs , mean_of_divided_RPPs , cov_mtrx_of_divided_RPPs = Network.calc_mean_and_cov_mtrx_of_dividing_two_RPPs_to_smaller_ones( data_of_RPPs_original , r_correlation , num_of_smaller_RPPs_from_dividing_each_RPP)
data_of_RPPs = data_of_divided_RPPs
Cov_matrix_of_RPPs = cov_mtrx_of_divided_RPPs
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
list_of_indices_of_allowed_lines_to_be_congested_in_DA_market = list(range(0,20))
list_of_num_of_allowed_lines_to_be_congested_in_DA_market = [0,1]
#
list_of_indices_of_allowed_lines_to_be_congested_in_RT_market = list(range(0,20))
list_of_num_of_allowed_lines_to_be_congested_in_RT_market = [0,1]
#
#
#
## ## ## ## ## ## ## ## ## SIMULATIONS ## ## ## ## ## ## ## ## ## ## ## ## ##

My_Network = Network( compact_data_of_branches = compact_data_of_branches , index_of_slack_node =  index_of_slack_node , vector_of_nodal_demands = vector_of_nodal_demands , data_of_RPPs = data_of_RPPs , data_of_DA_conventional_generators =  data_of_DA_conventional_generators , data_of_RT_conventional_generators =  data_of_RT_conventional_generators , list_of_indices_of_allowed_lines_to_be_congested_in_DA_market = list_of_indices_of_allowed_lines_to_be_congested_in_DA_market , list_of_indices_of_allowed_lines_to_be_congested_in_RT_market = list_of_indices_of_allowed_lines_to_be_congested_in_RT_market , list_of_num_of_allowed_lines_to_be_congested_in_RT_market = list_of_num_of_allowed_lines_to_be_congested_in_RT_market , list_of_num_of_allowed_lines_to_be_congested_in_DA_market = list_of_num_of_allowed_lines_to_be_congested_in_DA_market , Cov_matrix_of_RPPs = Cov_matrix_of_RPPs )
#
#
#
#
#
list_of_DA_commitments_at_NEs , list_of_congestions_at_NEs_in_DA_market , list_of_congestions_at_NEs_in_RT_market , list_of_DA_commitments_calculated_by_algorithm_ALL , list_of_congestions_in_DA_market_assumed_by_algorithm_ALL , list_of_congestions_in_DA_market_calculated_by_ISO_ALL , list_of_generation_of_DA_generators_ALL, list_of_expected_generation_of_RT_generators_ALL = My_Network.find_the_NEs___case_same_DA_and_RT_cong_pttrns()
#
#
#
# print RESULTS
num_of_NEs = len(list_of_DA_commitments_at_NEs)
if num_of_NEs == 0:
    print('There is no NE for the given netwrok and given set of pairs of DA and RT congestion patterns')
else:
    list_output_print = list()
    for i in range(0,num_of_NEs):
        list_output_print.append([list_of_congestions_at_NEs_in_DA_market[i] , list_of_DA_commitments_at_NEs[i]])
    #
    print('\n'*1 + ' '*30 + '*'*10 + 'Results' + '*'*10 + '\n'*3)
    print(tabulate(list_output_print, headers=['DA Congestion Pattern at NE', 'DA Commitments of RPPs at NE'], tablefmt='orgtbl'))



