# import sys
# sys.modules[__name__].__dict__.clear()
#
import gc
gc.collect()
#
import numpy as np
# import math
import itertools
# import pypi
import cvxopt
import cvxpy
import scipy
import scipy.optimize as sciopt
import copy
import math
# import scs
import matplotlib.pyplot as plt
#
import gurobipy
# import pygurobi
#
#
class Network:
    def __init__(self , compact_data_of_branches , index_of_slack_node , vector_of_nodal_demands = None , data_of_RPPs = None , data_of_DA_conventional_generators = None , data_of_RT_conventional_generators = None , list_of_indices_of_allowed_lines_to_be_congested_in_DA_market = None , list_of_num_of_allowed_lines_to_be_congested_in_DA_market = None , list_of_indices_of_allowed_lines_to_be_congested_in_RT_market = None , list_of_num_of_allowed_lines_to_be_congested_in_RT_market = None , Cov_matrix_of_RPPs = None ):
        self.compact_data_of_branches = copy.deepcopy(compact_data_of_branches)
        self.num_of_branches = copy.deepcopy(self.compact_data_of_branches.shape[0])
        self.num_of_nodes = int ( max(max(compact_data_of_branches[:,0]) , max(compact_data_of_branches[:,1])) ) + 1  # well some branch is connected to this node anyway
        (self.node_branch_matrix , self.branch_capacities , self.reactance_of_branches) = copy.deepcopy(self.extract_data_of_branches_from___compact_data_of_branches())
        self.index_of_slack_node = copy.deepcopy(index_of_slack_node)
        self.vector_of_nodal_demands = copy.deepcopy(np.array(vector_of_nodal_demands,dtype=float).tolist())
        self.data_of_RPPs = copy.deepcopy(data_of_RPPs)
        self.data_of_DA_conventional_generators = copy.deepcopy(data_of_DA_conventional_generators)
        self.data_of_RT_conventional_generators = copy.deepcopy(data_of_RT_conventional_generators)
        self.susceptance_matrix = copy.deepcopy(self.calc_susceptance_matrix())
        self.matrix_of_angles_to_line_flows = copy.deepcopy(self.calc_angles_to_line_flows_matrix())
        self.num_of_RPPs = copy.deepcopy(self.data_of_RPPs.shape[0])
        self.num_of_DA_conv_generators = copy.deepcopy(self.data_of_DA_conventional_generators.shape[0])
        self.num_of_RT_conv_generators = copy.deepcopy(self.data_of_RT_conventional_generators.shape[0])
        self.list_of_indices_of_nodes_with_DA_conv_generator = copy.deepcopy([int(temppp) for temppp in self.data_of_DA_conventional_generators[:,0]])
        self.list_of_indices_of_nodes_with_RT_conv_generator = copy.deepcopy([int(temppp) for temppp in self.data_of_RT_conventional_generators[:, 0]])
        self.list_of_indices_of_allowed_lines_to_be_congested_in_DA_market = copy.deepcopy(list_of_indices_of_allowed_lines_to_be_congested_in_DA_market)
        self.list_of_num_of_allowed_lines_to_be_congested_in_DA_market = copy.deepcopy(list_of_num_of_allowed_lines_to_be_congested_in_DA_market)
        self.list_of_indices_of_allowed_lines_to_be_congested_in_RT_market = copy.deepcopy(list_of_indices_of_allowed_lines_to_be_congested_in_RT_market)
        self.list_of_num_of_allowed_lines_to_be_congested_in_RT_market = copy.deepcopy(list_of_num_of_allowed_lines_to_be_congested_in_RT_market)
        self.PTDF_matrix = copy.deepcopy(self.calc_PTDF_matrix())
        self.Cov_matrix_of_RPPs = copy.deepcopy(Cov_matrix_of_RPPs)
        ## ## ##
        ## Explanation of the data structure
        #
        #
        # note that nodes are numbered starting from zero (not one)
        # Similarly, note that branches are numbered starting from zero (not one)
        #
        #
        # data_of_RPPs == is a num_of_RPPS x 4 matrix, where the row #i corresponds
        # ... to the RPP #i. the elements of this row are:
        # data_of_RPPs(i,0) = is the node that the RPP #i is connected to.
        # data_of_RPPs(i,1) = is the statistical mean of pdf of RPP #i.
        # data_of_RPPs(i,2) = is the statistical standard deviation of pdf of RPP #i.
        #
        #
        # data_of_DA_conventional_generators == is a num_of_DA_conventional_generators x 5 matrix, where the row #i corresponds
        # ... to the conventional generator #i. the elements of this row are:
        # data_of_DA_conventional_generators(i,0) = is the node that the DA conventional generator #i is connected to.
        # data_of_DA_conventional_generators(i,1) = is the parameter alpha_i of the cost function of the DA conventional generator #i.
        # data_of_DA_conventional_generators(i,2) = is the parameter beta_i of the cost function of the DA conventional generator #i.
        # data_of_DA_conventional_generators(i,3) = is the parameter gamma_i of the cost function of the DA conventional generator #i.
        # data_of_DA_conventional_generators(i,4) = is the capacity of the DA conventional generator #i.
        #
        #
        # data_of_RT_conventional_generators == is a num_of_RT_conventional_generators x 5 matrix, where the row #i corresponds
        # ... to the conventional generator #i. the elements of this row are:
        # data_of_RT_conventional_generators(i,0) = is the node that the RT conventional generator #i is connected to.
        # data_of_RT_conventional_generators(i,1) = is the parameter alpha_i of the cost function of the RT conventional generator #i.
        # data_of_RT_conventional_generators(i,2) = is the parameter beta_i of the cost function of the RT conventional generator #i.
        # data_of_RT_conventional_generators(i,3) = is the parameter gamma_i of the cost function of the RT conventional generator #i.
        # data_of_RT_conventional_generators(i,4) = is the capacity of the RT conventional generator #i.
        #
        #
        #  compact_data_of_branches = a matrix of size num_of_branches * 4, for which
        #  ... column #k of it corresponds to a line #k. note that this order of ...
        # ... branches is for our convention, yet this convention automatically ...
        # ... holds for all other codes. So, if we put a line in the row #6 of this ...
        # ... input matrix, then in all the codes, the branch #6 refers to this branch. The elements are
        # the element at row #i and column #0 is the head node of the branch #i.
        # the element at row #j and column #1 is the tail node of the branch #i.
        # the element at row #j and column #2 is the capacity of the branch #i.
        # the element at row #j and column #3 is the reactance of the branch #i.
        #
        # node_branch_matrix = the node_branch_matrix (nodes are corresponded ...)
        # (... to the rows, and branched are corresponded to the columns).
        # in column j of node_branch_matrix (corresponding to branch j), if the branch j ...
        # ... is coming out of the node m and go to the node n then the element [m,j] of this matrix ...
        # ... is +1 and the element [n,j] of this matrix is -1. The other elements of the j th column ...
        # ... of this matrix are zeros.
        #
        #
        # reactance_of_branches = a vector of size 1 x self.num_of_branches, which the element ...
        # ... #j of it is the reactance of branch #j.
        #
        #
        # branch_capacities = a vector of size 1 x self.num_of_branches, which the element ...
        # ... #j of it is the capacity of branch #j.
        #
        #
        #
        #
    #
    #
    #
    def convert_decimal_to_specific_base__returns_vector_or_string(self, decimal_form, target_base, length_of_target_base_form=None, return_vector=True, delete_first_zeros=True):
    # this function returns a string that includes all the converted version of the number "decimal_form" to the base of "target_base" ...
    # ... the length of the converted number is length_of_target_base_form
        if return_vector == True: # return a python list. Note that in this case the parameter delete_first_zeros is NOT important
            #
            if (decimal_form == 0) and (length_of_target_base_form != None):
                return [0] * length_of_target_base_form
            elif (decimal_form == 0) and (length_of_target_base_form == None):
                return [0]
            elif length_of_target_base_form == None:
                length_of_target_base_form = 0
                while True:
                    if decimal_form // (target_base ** length_of_target_base_form) != 0:
                        length_of_target_base_form += 1
                    else:
                        break
            else:
                pass
            #
            transferred_base_number__vector = [0] * length_of_target_base_form
            decimal_form_temp = decimal_form
            counter_temp = 0
            while True:
                next_digit = int(decimal_form_temp // (target_base ** (length_of_target_base_form - 1 - counter_temp)))
                # print('counter_temp  ==  {}'.format(counter_temp))
                transferred_base_number__vector[counter_temp] = next_digit
                #
                decimal_form_temp = int(decimal_form_temp % (target_base ** (length_of_target_base_form - 1 - counter_temp)))
                #
                counter_temp += 1
                if counter_temp == length_of_target_base_form:
                    break
                else:
                    pass
            #
            return transferred_base_number__vector
            #
        else: # return a string. Note that in this case the parameter delete_first_zeros is important
            if (not isinstance(target_base , int)) or (target_base <= 0): # the base is not a positive integer
                print("the base is not a positive integer")
            elif (length_of_target_base_form != None) and (decimal_form >= target_base**length_of_target_base_form): # the number can not be represented with this length
                print("the number can not be represented with this length")
            elif (length_of_target_base_form != None) and  ((not isinstance(length_of_target_base_form , int)) or (length_of_target_base_form <= 0)):
                print('either leave length_of_target_base_form as blank, or specify a positive integer')
                return None
            else: # we can represent the number
                if decimal_form == 0:
                    #
                    if delete_first_zeros:
                        return '0'
                    elif (not delete_first_zeros) and (length_of_target_base_form != None):
                        return '0'*length_of_target_base_form
                    else: # (not delete_first_zeros) and (length_of_target_base_form == None):
                        return '0'  # because we do not know what is the length_of_target_base_form
                    #
                else:
                    #
                    if length_of_target_base_form == None:
                        length_of_target_base_form = 0
                        while True:
                            if decimal_form // (target_base ** length_of_target_base_form) != 0:
                                length_of_target_base_form += 1
                            else:
                                break
                    else:
                        pass
                    #
                    transferred_base_number__string = ''
                    decimal_form_temp = decimal_form
                    counter_temp = 0
                    first_nonzero_digit_not_found = True
                    while True:
                        next_digit = decimal_form_temp // (target_base ** (length_of_target_base_form - 1 - counter_temp))
                        transferred_base_number__string = transferred_base_number__string + str(next_digit)
                        #
                        if (next_digit != 0) and first_nonzero_digit_not_found:
                            location_of_first_nonzero_digit_in_string = counter_temp
                            first_nonzero_digit_not_found = False
                        else:
                            pass
                        #
                        decimal_form_temp = decimal_form_temp % (target_base ** (length_of_target_base_form - 1 - counter_temp))
                        #
                        counter_temp += 1
                        if counter_temp == length_of_target_base_form:
                            break
                        else:
                            pass
                    #
                    if delete_first_zeros:
                        return transferred_base_number__string[location_of_first_nonzero_digit_in_string:]
                    else:
                        return transferred_base_number__string
    #
    #
    def extract_data_of_branches_from___compact_data_of_branches(self):
        #
        # forming node_branch_matrix
        self.node_branch_matrix = copy.deepcopy(np.zeros((self.num_of_nodes , self.num_of_branches)))
        for j in range(self.num_of_branches):
            head_of_branch_j = int(self.compact_data_of_branches[j,0])
            tail_of_branch_j = int(self.compact_data_of_branches[j, 1])
            #
            self.node_branch_matrix[head_of_branch_j , j] = 1
            self.node_branch_matrix[tail_of_branch_j , j] = -1
        #
        self.branch_capacities = copy.deepcopy(self.compact_data_of_branches[:,2])
        self.reactance_of_branches = copy.deepcopy(self.compact_data_of_branches[:, 3])
        #
        return self.node_branch_matrix , self.branch_capacities , self.reactance_of_branches
    #
    #
    def calc_angles_to_line_flows_matrix(self):
        matrix_of_angles_to_line_flows = np.zeros((self.num_of_branches , self.num_of_nodes))
        #
        for j in np.arange(0,self.num_of_branches,1,dtype=int):
            for i in np.arange(0, self.num_of_nodes,1,dtype=int):
                if self.node_branch_matrix[i,j] == 1:
                    head_of_cuurent_branch_j = i
                elif self.node_branch_matrix[i,j] == -1:
                    tail_of_cuurent_branch_j = i
                else:
                    pass # do nothing
            #
            matrix_of_angles_to_line_flows[ j , head_of_cuurent_branch_j] = (1 / self.reactance_of_branches[j])
            matrix_of_angles_to_line_flows[ j , tail_of_cuurent_branch_j] = (-1 / self.reactance_of_branches[j])
        #
        #
        return matrix_of_angles_to_line_flows
    #
    #
    def calc_susceptance_matrix(self):
        #
        #
        #  vector of net injections at nodes  =  susceptance_matrix * nodal angles
        #
        susceptance_matrix = copy.deepcopy(np.zeros((self.num_of_nodes , self.num_of_nodes) , dtype = float))
        #
        for j in range(self.num_of_branches):
            head_of_cuurent_branch_j = None
            tail_of_cuurent_branch_j = None
            for i in range(self.num_of_nodes):
                if self.node_branch_matrix[i,j] == 1:
                    head_of_cuurent_branch_j = i
                elif self.node_branch_matrix[i,j] == -1:
                    tail_of_cuurent_branch_j = i
                elif (head_of_cuurent_branch_j != None) and (tail_of_cuurent_branch_j != None):
                    break
                else:
                    pass # do nothing
            #
            susceptance_matrix[head_of_cuurent_branch_j, head_of_cuurent_branch_j] = susceptance_matrix[head_of_cuurent_branch_j, head_of_cuurent_branch_j] +  (1 / self.reactance_of_branches[j])
            susceptance_matrix[tail_of_cuurent_branch_j, tail_of_cuurent_branch_j] = susceptance_matrix[tail_of_cuurent_branch_j, tail_of_cuurent_branch_j] +  (1 / self.reactance_of_branches[j])
            susceptance_matrix[head_of_cuurent_branch_j, tail_of_cuurent_branch_j] = susceptance_matrix[head_of_cuurent_branch_j, tail_of_cuurent_branch_j] +  (-1 / self.reactance_of_branches[j])
            susceptance_matrix[tail_of_cuurent_branch_j, head_of_cuurent_branch_j] = susceptance_matrix[tail_of_cuurent_branch_j, head_of_cuurent_branch_j] +  (-1 / self.reactance_of_branches[j])
            #
        return susceptance_matrix
    #
    #
    #
    def generate_all_possible_congestion_patterns_for_length_k(self,k):
        # this function generates all the congestion patterns possible for k lines
        #
        # note that this function does not include those lines that are not congested
        # note that here all lines get congested (no zeros in the output)
        #
        # 1 is congestion in the direction defined in the definition
        # 2 is congestion in the reverse of the direction defined in the definition
        #
        #
        the_result_all = []
        for i in range(0,(2**k)):
            the_result_current = copy.deepcopy(self.convert_decimal_to_specific_base__returns_vector_or_string(i, 2, k, return_vector=True, delete_first_zeros=None))
            the_result_current = [x+1 for x in the_result_current]
            # print(the_result_current)
            the_result_all = the_result_all + [the_result_current]
        #
        return the_result_all
    #
    #
    #
    def choose_k_among_n_elements_of_a_list(self , input_list , k):
        #
        #
        # this function returns all choices of k elements of the list "input_list"
        #
        # it also removes the repetitve elements in the input_list, and considers only one instance for mupltiple occurance of single item
        #
        # note that the length of the input_list is n, and it should be no less than k
        #
        input_list_no_repitition = list(set(input_list))
        n = len(input_list)
        #
        if input_list_no_repitition == []:
            return []
        elif k == 0:
            return []
        elif not isinstance(k,int):
            print('k = {} should be an integer'.format(k))
            return None
        elif n < k:
            print('The length of the input list must be no less than k = {}'.format(k))
            return None
        elif n == k:
            return [input_list_no_repitition]
        else:
            #
            #
            final_result_list_of_tuples = list(itertools.combinations(tuple(input_list_no_repitition), k))
            final_result_list_of_lists = [list(t) for t in final_result_list_of_tuples]
            #
            return final_result_list_of_lists
    #
    #
    def find_all_allowed_congestions_patterns__both_congested_and_not_congested_lines_in_DA_market( self ):
        # matrix_of_all_critical_line_congestions
        #
        #
        # "matrix_of_all_critical_line_congestions_in_DA_market" is a matrix where the ith row ...
        # ... of this matrix is a  1 x num_of_branches vector of congestion of the lines
        #
        # self.list_of_indices_of_allowed_lines_to_be_congested_in_DA_market  is the list of allowed lines to be congested in the DA market
        #
        #
        # self.list_of_num_of_allowed_lines_to_be_congested_in_DA_market is the list of the lines that are candidates to be congested in the DA market
        #
        if self.list_of_num_of_allowed_lines_to_be_congested_in_DA_market == [0]:
            return [0]*self.num_of_branches
        else:
            #
            list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_DA_market = []
            #
            for i in self.list_of_num_of_allowed_lines_to_be_congested_in_DA_market:
                current_num_of_lines_to_be_congested_in_DA_market = i
                if current_num_of_lines_to_be_congested_in_DA_market == 0:
                    list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_DA_market = list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_DA_market + np.zeros((1,self.num_of_branches)).tolist()
                else:
                    list_of_cong_pttrns_for___current_num_of_lines_to_be_congested_in_DA_market = copy.deepcopy(self.generate_all_possible_congestion_patterns_for_length_k(current_num_of_lines_to_be_congested_in_DA_market))
                    list_of_all_combinations_of_indices_for___current_num_of_lines_to_be_congested_in_DA_market = copy.deepcopy(self.choose_k_among_n_elements_of_a_list(self.list_of_indices_of_allowed_lines_to_be_congested_in_DA_market , current_num_of_lines_to_be_congested_in_DA_market ))
                    #
                    #
                    for j in list_of_all_combinations_of_indices_for___current_num_of_lines_to_be_congested_in_DA_market:
                        current_indices_to_be_congested_for___current_num_of_lines_to_be_congested_in_DA_market = j
                        #
                        for t in list_of_cong_pttrns_for___current_num_of_lines_to_be_congested_in_DA_market:
                            current_cong_pttrn_for_congsted_lines_only_for___curr_num_of_cong_lines__and__curr_indices_in_DA_market = t
                            #
                            next_cong_pttrn_containing_both_congstd_and_non_congstd_in_DA_market = np.zeros((1,self.num_of_branches), dtype=int)
                            #
                            #
                            next_cong_pttrn_containing_both_congstd_and_non_congstd_in_DA_market[0,current_indices_to_be_congested_for___current_num_of_lines_to_be_congested_in_DA_market] = current_cong_pttrn_for_congsted_lines_only_for___curr_num_of_cong_lines__and__curr_indices_in_DA_market
                            next_cong_pttrn_containing_both_congstd_and_non_congstd_in_DA_market = next_cong_pttrn_containing_both_congstd_and_non_congstd_in_DA_market.tolist()
                            list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_DA_market = list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_DA_market + next_cong_pttrn_containing_both_congstd_and_non_congstd_in_DA_market
                        #
                    #
                #
            #
            return list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_DA_market
    #
    #
    #
    def find_all_allowed_congestions_patterns__both_congested_and_not_congested_lines_in_RT_market( self ):
        # matrix_of_all_critical_line_congestions
        #
        #
        # "matrix_of_all_critical_line_congestions_in_RT_market" is a matrix where the ith row ...
        # ... of this matrix is a  1 x num_of_branches vector of congestion of the lines
        #
        # self.list_of_indices_of_allowed_lines_to_be_congested_in_RT_market  is the list of allowed lines to be congested in the RT market
        #
        #
        # self.list_of_num_of_allowed_lines_to_be_congested_in_RT_market is the list of the lines that are candidates to be congested in the DA market
        #
        if self.list_of_num_of_allowed_lines_to_be_congested_in_RT_market == [0]:
            return [0]*self.num_of_branches
        else:
            #
            list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_RT_market = []
            #
            for i in self.list_of_num_of_allowed_lines_to_be_congested_in_RT_market:
                current_num_of_lines_to_be_congested_in_RT_market = i
                if current_num_of_lines_to_be_congested_in_RT_market == 0:
                    list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_RT_market = list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_RT_market + np.zeros((1,self.num_of_branches)).tolist()
                else:
                    list_of_cong_pttrns_for___current_num_of_lines_to_be_congested_in_RT_market = copy.deepcopy(self.generate_all_possible_congestion_patterns_for_length_k(current_num_of_lines_to_be_congested_in_RT_market))
                    list_of_all_combinations_of_indices_for___current_num_of_lines_to_be_congested_in_RT_market = copy.deepcopy(self.choose_k_among_n_elements_of_a_list(self.list_of_indices_of_allowed_lines_to_be_congested_in_RT_market , current_num_of_lines_to_be_congested_in_RT_market ))
                    #
                    #
                    for j in list_of_all_combinations_of_indices_for___current_num_of_lines_to_be_congested_in_RT_market:
                        current_indices_to_be_congested_for___current_num_of_lines_to_be_congested_in_RT_market = j
                        #
                        for t in list_of_cong_pttrns_for___current_num_of_lines_to_be_congested_in_RT_market:
                            current_cong_pttrn_for_congsted_lines_only_for___curr_num_of_cong_lines__and__curr_indices_in_RT_market = t
                            #
                            next_cong_pttrn_containing_both_congstd_and_non_congstd_in_RT_market = np.zeros((1,self.num_of_branches), dtype=int)
                            #
                            #
                            next_cong_pttrn_containing_both_congstd_and_non_congstd_in_RT_market[0,current_indices_to_be_congested_for___current_num_of_lines_to_be_congested_in_RT_market] = current_cong_pttrn_for_congsted_lines_only_for___curr_num_of_cong_lines__and__curr_indices_in_RT_market
                            next_cong_pttrn_containing_both_congstd_and_non_congstd_in_RT_market = next_cong_pttrn_containing_both_congstd_and_non_congstd_in_RT_market.tolist()
                            list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_RT_market = list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_RT_market + next_cong_pttrn_containing_both_congstd_and_non_congstd_in_RT_market
                        #
                    #
                #
            #
            return list_of_all_congestion_patterns__contain_both_congstd_and_not_congstd_in_RT_market
    #
    #
    #
    #
    #
    #
    def calc_right_null_space_vectors_of_a_matrix(self , A_input_matrix, eps=1e-15):
        # this function returns the null space of the matrix: A_input_matrix
        #
        A_input_matrix_modified = np.array(A_input_matrix)
        (num_of_rows_of_input_matrix , num_of_columns_of_input_matrix) = A_input_matrix_modified.shape
        # make the input matrix a square matrix either by adding rows or columns
        if num_of_rows_of_input_matrix < num_of_columns_of_input_matrix:
            A_input_matrix_modified = np.concatenate((A_input_matrix_modified,np.zeros(((num_of_columns_of_input_matrix - num_of_rows_of_input_matrix), A_input_matrix_modified.shape[1])).reshape((((num_of_columns_of_input_matrix - num_of_rows_of_input_matrix), A_input_matrix_modified.shape[1])))), axis=0)
        elif num_of_rows_of_input_matrix > num_of_columns_of_input_matrix:
            A_input_matrix_modified = np.concatenate((A_input_matrix_modified,np.zeros((A_input_matrix_modified.shape[1] , (num_of_rows_of_input_matrix - num_of_columns_of_input_matrix))).reshape(((A_input_matrix_modified.shape[1] , (num_of_rows_of_input_matrix - num_of_columns_of_input_matrix))))), axis=1)
        else:
            pass
        #
        ## Method 1 (this method apparently is more reliable)
        u, s, vh = scipy.linalg.svd(A_input_matrix_modified)
        null_mask = (s <= eps)
        null_space = scipy.compress(null_mask, vh, axis=0)
        return scipy.transpose(null_space)
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        ## method 2
        # eigen_values, eigen_vectors = np.linalg.eig(A_input_matrix_modified)
        # return eigen_vectors
    #
    #
    #
    def check_feasibility_of_a_set_of_inequality_and_equality_equations__using_Farkas_lemma(self , A_ineq , b_ineq , A_eq , b_eq):
        #
        # for regerence to the Farkas lemma, see: https://en.wikipedia.org/wiki/Farkas%27_lemma
        #
        #
        # this program checks to see whether the following set of inequality and equality equations are feasible ot not
        #
        #               A_ineq * x <= b_ineq
        #               A_eq * x  = b_eq
        #
        # to check this, we use the Farka's lemma, which states that:
        #  Either the system A*x = b (x can be positive or negative) has solution, ...
        #       .... or the system A'*y = 0 has a solution with b'*y < 0 and y >= 0.
        #
        #
        #
        #
        #
        #
        # output:  An integer representing the status of the feasibility of the set of inequalities and equalities.
        #                           0 : Optimization proceeding nominally.
        #                           1 : Iteration limit reached.
        #                           2 : Problem appears to be infeasible.
        #                           3 : Problem appears to be unbounded.
        #                           4 : Numerical difficulties encountered.
        #
        #
        #
        A_ineq_np = np.array(A_ineq)
        b_ineq_np = np.array(b_ineq)
        A_eq_np = np.array(A_eq)
        b_eq_np = np.array(b_eq)
        #
        #
        # determine the number of main variables:
        if A_ineq_np.size == A_ineq_np.shape[0]: # this means that there is only one inequality
            num_of_main_variables = A_ineq_np.size
            num_of_inequalities = 1
        else:
            num_of_main_variables = A_ineq_np.shape[1]
            num_of_inequalities = A_ineq_np.shape[0]
        #
        #
        # determine the number of slack variables:
        if A_eq_np.size == A_eq_np.shape[0]:  # this means that there is only one equality
            num_of_slack_variables = 1
            num_of_equalities = 1
        else:
            num_of_slack_variables = A_eq_np.shape[0]
            num_of_equalities = A_eq_np.shape[0]
        #
        #
        #
        left_hand_right_matrix_to_form_for_Farkas_lemma__part_1 = np.concatenate(  (  A_ineq_np, np.zeros((num_of_inequalities, num_of_slack_variables))), axis=1)
        left_hand_right_matrix_to_form_for_Farkas_lemma__part_2 = np.concatenate(  (  A_eq_np  ,  np.eye( num_of_equalities , num_of_slack_variables  )  )  ,  axis = 1 )
        left_hand_right_matrix_to_form_for_Farkas_lemma__part_3 = np.concatenate(  (  np.zeros((num_of_slack_variables, num_of_main_variables)) , -1*np.eye(num_of_slack_variables, num_of_slack_variables)), axis=1)
        left_hand_right_matrix_to_form_for_Farkas_lemma = np.concatenate((left_hand_right_matrix_to_form_for_Farkas_lemma__part_1 , left_hand_right_matrix_to_form_for_Farkas_lemma__part_2 , left_hand_right_matrix_to_form_for_Farkas_lemma__part_3) , axis=0 )
        #
        right_hand_right_matrix_to_form_for_Farkas_lemma = np.concatenate(( b_ineq_np , b_eq_np , np.zeros((num_of_slack_variables , 1)).reshape((num_of_slack_variables,1))) , axis=0)
        #
        null_space_of_the_matrix = copy.deepcopy(self.calc_right_null_space_vectors_of_a_matrix(left_hand_right_matrix_to_form_for_Farkas_lemma.transpose()))
        #
        # calc num_of_eigenvectors
        num_of_eigenvectors = null_space_of_the_matrix.shape[1]
        #
        #
        output_result = 'feasible'
        for i in range(num_of_eigenvectors):
            current_eigenvector = null_space_of_the_matrix[:,i]
            current_eigenvector = current_eigenvector.reshape((current_eigenvector.size , 1))
            #
            if (( right_hand_right_matrix_to_form_for_Farkas_lemma.transpose().dot(current_eigenvector) ) < 0) and (sum(sum(current_eigenvector >= 0))  == current_eigenvector.size ):
                # this means that our main set of equalities and inequalities is infeasible
                output_result = 'infeasible'
                break
            else:
                pass
            #
        #
        return output_result
    #
    #
    #
    def calc_PTDF_matrix(self):
        #
        #
        # this function returns a cell, for which the i th element of this cell contains the PTDF matrix for the ...
        # ... i th line, which is a matrix of this matrix returns the PTDF matrix, which is ...
        # ... of size  num_of_nodes * num_of_branches
        #
        # note that the element (i , j) of this PTDF corresponds to the share of 1 W injection at node i and withdraw ..
        # ... from slack node for which goes through line j.
        #
        # note that the slack node is defined by the user in index_of_slack_node
        #
        #
        #
        temp_susceptance_matrix = copy.deepcopy(self.susceptance_matrix)
        #
        temp_susceptance_matrix = np.delete(temp_susceptance_matrix , self.index_of_slack_node , axis=0)
        temp_susceptance_matrix = np.delete(temp_susceptance_matrix , self.index_of_slack_node, axis=1)
        # inverse_susceptance_matrix = temp_susceptance_matrix^(-1);
        inverse_susceptance_matrix = np.linalg.pinv(temp_susceptance_matrix)
        temp_column = np.zeros((self.num_of_nodes - 1, 1)) # the length is one smaller than the row, because we first added the column here
        temp_row = np.zeros((1, self.num_of_nodes))
        temp_row[:] = np.nan
        temp_column[:] = np.nan
        if self.index_of_slack_node == self.num_of_nodes - 1:
            inverse_susceptance_matrix = np.concatenate((inverse_susceptance_matrix[:, 0: self.index_of_slack_node], temp_column), axis=1)
            inverse_susceptance_matrix = np.concatenate((inverse_susceptance_matrix[0: self.index_of_slack_node , :], temp_row), axis=0)
        elif self.index_of_slack_node == 0:
            inverse_susceptance_matrix = np.concatenate((temp_column , inverse_susceptance_matrix[:, self.index_of_slack_node:]), axis=1)
            inverse_susceptance_matrix = np.concatenate((temp_row, inverse_susceptance_matrix[self.index_of_slack_node:, :]), axis=0)
        else:
            inverse_susceptance_matrix = np.concatenate((inverse_susceptance_matrix[:, 0: self.index_of_slack_node], temp_column, inverse_susceptance_matrix[:, self.index_of_slack_node:]), axis=1)
            inverse_susceptance_matrix = np.concatenate((inverse_susceptance_matrix[0: self.index_of_slack_node, :], temp_row, inverse_susceptance_matrix[self.index_of_slack_node:, :]), axis=0)
        #
        #
        #
        #
        PTDF_matrix = copy.deepcopy(np.zeros((self.num_of_nodes, self.num_of_branches)))
        #
        for i in range(self.num_of_nodes):
            for j in range(self.num_of_branches):
                head_of_branch_j = list(self.node_branch_matrix[:,j]).index(1)
                tail_of_branch_j = list(self.node_branch_matrix[:,j]).index(-1)
                if (head_of_branch_j != self.index_of_slack_node) and (tail_of_branch_j != self.index_of_slack_node) and (i != self.index_of_slack_node):
                    PTDF_matrix[i, j] = abs(self.susceptance_matrix[head_of_branch_j, tail_of_branch_j]) * ( inverse_susceptance_matrix[head_of_branch_j, i] - inverse_susceptance_matrix[tail_of_branch_j, i])
                elif (head_of_branch_j == self.index_of_slack_node) and (i != self.index_of_slack_node):
                    PTDF_matrix[i, j] = abs(self.susceptance_matrix[head_of_branch_j, tail_of_branch_j]) * ( - inverse_susceptance_matrix[tail_of_branch_j, i])
                elif (tail_of_branch_j == self.index_of_slack_node) and (i != self.index_of_slack_node):
                    PTDF_matrix[i, j] = abs(self.susceptance_matrix[head_of_branch_j, tail_of_branch_j]) * ( inverse_susceptance_matrix[head_of_branch_j, i])
                elif (i == self.index_of_slack_node):
                    PTDF_matrix[i, j] = 0
                else:
                    pass
            #
        #
        #
        return PTDF_matrix
    #
    #
    #
    def check_feasibility_of_a_set_of_inequality_and_equality_equations(self , A_ineq , b_ineq , A_eq , b_eq):
        # this program checks to see whether the following set of inequality and equality equations are feasible ot not
        #
        #               A_ineq * x <= b_ineq
        #               A_eq * x  = b_eq
        #
        #
        # it returns either 'feasible' or 'infeasible'
        #
        #
        #  solver can be: 'cvxpy'
        #                 'cvxopt'
        #
        # to check this, we import these equations as constraints into a linear optimization problem and see whether ...
        # ... it is feasible:
        #
        #               min q'*x
        #           s.t.
        #               A_ineq * x <= b_ineq
        #               A_eq * x  = b_eq
        #
        # note that the value of vector  c  does not matter because here we only look to see whether there is any ...
        # feasible solution or not.
        # we use the package   scipy.optimize.linprog   to solve the linear programming problem
        #
        #
        #
        # output:  An integer representing the status of the feasibility of the set of inequalities and equalities.
        #
        # solution_np['status'] =
        #                         0 : Optimization proceeding nominally.
        #                         1 : Iteration limit reached.
        #                         2 : Problem appears to be infeasible.
        #                         3 : Problem appears to be unbounded.
        #                         4 : Numerical difficulties encountered.
        #
        #
        A_ineq_np = np.array(A_ineq)
        b_ineq_np = np.array(b_ineq)
        A_eq_np = np.array(A_eq)
        b_eq_np = np.array(b_eq)
        #
        #
        # determine the number of variables:
        if A_ineq_np.size == A_ineq_np.shape[0]: # this means that there is only one inequality
            num_of_variables = A_ineq_np.size
        else:
            num_of_variables = A_ineq_np.shape[1]
        #
        #
        q = np.zeros(( num_of_variables , 1 ))
        #
        bounds = [(None,None)]*num_of_variables
        #
        #
        # method can be: method='interior-point'
        #                method='simplex'  (default)
        #
        #
        # solution_np = sciopt.linprog(q, A_ineq_np, b_ineq_np, A_eq_np, b_eq_np, bounds=None, method=None,callback=None, options=None)
        solution_np = sciopt.linprog(q, A_ineq_np, b_ineq_np, A_eq_np, b_eq_np, bounds , method= 'interior-point', callback=None, options=None)
        #
        #
        if (solution_np['status'] == 2) or (solution_np['status'] == 4):
            return 'infeasible'
        else:
            return 'feasible'
        #
        #
        #
    #
    #
    #
    def solve_DA_economic_dispatch_by_ISO(self, DA_commitments_of_RPPs):
        #
        # this function returns:
        # nodal_injections , LMPs , list_of_lines_congested_or_not , DC_power_flow_has_feasible_solution
        #
        #
        # DC_power_flow_has_feasible_solution = is a variable that takes ...
        # ... a value of True (if feasible) or False (if infeasible). If the rsultant DC power flow of the ...
        # ... combination of current DA commitment of the RPPs, the nodal demands, the line capacities, the ...
        # ... structure of the network, and the location of the DA conventional generators gives us a feasible ...
        # ... solution, then we will have a feasible solution, and current set of DA commitments ...
        # ... of the RPPs MAY be a pure NE. If there is no feasible solution to the DC ...
        # ... power flow, then definitely the current DA commitments are not a pure NE.
        #
        #
        #
        # list_of_lines_congested_or_not  is a 1*num_of_branches numpy vector where if the line #i is ...
        # ... congested in the direction of definition, the the i th element of this vector is 1, and if the ...
        # ... line #i is congested in the reverse direction of its definition,the i th element of this vector is -1, ...
        # ... and if the line #i is not congested, the i th element of this vector is 0.
        #
        #
        # note that this function considers the DA commitments of the RPPs as negative loads
        #
        #
        #
        #
        # this function calculates the optimal power generation of the DA generators and the corresponding ...
        # ... nodal angles of the DC OPF.
        #
        # the variables of the DC power flow optimization are:
        # x = [ nodal injections ; nodal angles]
        #
        # note that always the nodal angle of the reference node is zero (it is set in the equality constraint too)
        #
        #
        # we are trying to minimize 0.5 * xT * H * x + fT * x
        #
        # note that the vector of variables x consists of two set of variables:
        #   => the productions of DA generators
        #   => the nodal angles
        #
        # note that here we assume that there is one DA conv generators at each ...
        # node. then for those nodes that do not have any conv generators, we make ...
        # the corresponding parameters in H and f equal to zero.
        #
        # The variable vector x consists of 2*num_of_nodes variables: the first num_of_nodes ...
        # ... variables correspond to the DA conv injections, and the second num_of_nodes ...
        # ... variables correspond to the nodal angles.
        #
        #
        #
        ## -------------------------------------------------------------------------------------------------------------
        ## Calculating H and f (Quadratic and linear coefficient terms in the objective function of the OPF)
        H = copy.deepcopy( np.zeros(((self.num_of_nodes + self.num_of_nodes) , (self.num_of_nodes + self.num_of_nodes))) )
        f = copy.deepcopy( np.zeros(((self.num_of_nodes + self.num_of_nodes), 1)) )
        fixed_costs = 0
        ##
        ## forming the quadratic and linear matrix and vector in the objective function
        #
        for i in range(self.num_of_nodes):
            #
            sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i = 0
            sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i = 0
            #
            for j in range(self.num_of_DA_conv_generators):
                if int(self.data_of_DA_conventional_generators[j, 0]) == int(i):
                    sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i = sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i + 1/self.data_of_DA_conventional_generators[j, 1]
                    sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i = sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i + self.data_of_DA_conventional_generators[j, 2]/self.data_of_DA_conventional_generators[j, 1]
                    fixed_costs = fixed_costs + copy.deepcopy( self.data_of_DA_conventional_generators[j, 3] )
                else:
                    pass
            #
            if sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i != 0:
                H[i, i] = 1/sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i
                f[i, 0] = sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i / sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i
            else:
                pass
            #
        #
        #
        #
        #
        net_nodal_withdraws = copy.deepcopy(np.array(self.vector_of_nodal_demands , dtype=float))
        for i in range(self.num_of_RPPs):
            the_node_of_current_RPP = int(self.data_of_RPPs[i, 0])
            net_nodal_withdraws[the_node_of_current_RPP] = net_nodal_withdraws[the_node_of_current_RPP] - (np.array(DA_commitments_of_RPPs).reshape(1,self.num_of_RPPs))[0,i]
        #
        #
        #
        ## -------------------------------------------------------------------------------------------------------------
        # forming the A_eq and b_eq matrices for the optimization
        # KCL for all nodes
        # A_eq = [(-1 * np.eye(self.num_of_nodes, self.num_of_nodes)), self.susceptance_matrix]
        # b_eq = -1 * net_nodal_withdraws
        A_eq = copy.deepcopy( np.concatenate( ((-1 * np.eye(self.num_of_nodes, self.num_of_nodes)), self.susceptance_matrix) , axis = 1 ) )
        b_eq = copy.deepcopy( -1 * net_nodal_withdraws.reshape(self.num_of_nodes , 1) )
        #
        #
        ## setting the injection of the nodes without DA generator eqaul to zero
        if set(self.data_of_DA_conventional_generators[:,0]).issubset(set(list(range(0,self.num_of_nodes)))): # there exists some nodes without DA generator on them
            set_of_nodes_without_DA_generator = set(list(range(0,self.num_of_nodes))) - set(self.data_of_DA_conventional_generators[:,0])
            num_of_nodes_without_DA_generator = len( set_of_nodes_without_DA_generator )
            addtional_block_to__A_eq = np.zeros((num_of_nodes_without_DA_generator, 2 * self.num_of_nodes))
            addtional_block_to__b_eq = np.zeros((num_of_nodes_without_DA_generator, 1))
            temp_counter = 0
            for i in set_of_nodes_without_DA_generator: # the insert power at node i is zero because the there is no conventional DA generator at this node
                addtional_block_to__A_eq[temp_counter, i] = 1;
                addtional_block_to__b_eq[temp_counter, 0] = 0;
                temp_counter += 1;
                #
            #
            # A_eq = [A_eq ; addtional_block_to__A_eq];
            # b_eq = [b_eq ; addtional_block_to__b_eq];
            A_eq = np.concatenate((A_eq , addtional_block_to__A_eq) , axis = 0)
            b_eq = np.concatenate((b_eq, addtional_block_to__b_eq), axis=0)
        else:
            pass
        #
        #
        ## setting the angle of the slack node equal to zero node equal to zero
        last_row_of_A_eq = np.zeros((1, 2 * self.num_of_nodes));
        last_row_of_A_eq[0, (self.index_of_slack_node - self.num_of_nodes)] = 1;
        # A_eq = [A_eq ; last_row_of_A_eq];
        # b_eq = [b_eq ; 0];
        A_eq = np.concatenate((A_eq, last_row_of_A_eq), axis=0)
        b_eq = np.concatenate((b_eq, np.array([0]).reshape((1,1))), axis=0)
        #
        #
        #
        ## -------------------------------------------------------------------------------------------------------------
        ## Calculating A_ineq and b_ineq. Note that these inequalities are for:
        #  ... (1) line flows on the defined directions
        #  ... (2) line flows on the reverse directions
        #  ... (3) upper bounds of the variables
        #  ... (4) lower bounds of the variables
        #
        A_ineq = copy.deepcopy( np.concatenate( (  np.zeros((self.num_of_branches, self.num_of_nodes))  , self.matrix_of_angles_to_line_flows ) , axis = 1 ) )
        #
        # to consider the reverse power flow throgh lines too
        A_ineq = copy.deepcopy( np.concatenate( (A_ineq , -1*A_ineq) , axis = 0 ) )
        #
        b_ineq = copy.deepcopy( np.concatenate((self.branch_capacities.reshape((self.num_of_branches,1)) , self.branch_capacities.reshape((self.num_of_branches,1))) , axis = 0) )
        #
        #
        ## note that we need to set a lower bound for the generators to be no less than total commitment of the RPPs
        # this is because in our formulation, we did allow the generators to produce negative values too.
        #
            #
        ##  DC power flow
        # x0 = np.zeros((2 * num_of_nodes, 1)).reshape(2 * num_of_nodes, 1);
        # [x, fval, exitflag, output, lambda ] = quadprog(H, f, A, b, A_eq, b_eq, lb, ub, x0, opt_temp);
        # H_np      = np.array(H     ,dtype=float)
        # f_np      = np.array(f     , dtype=float)
        # A_ineq_np = np.array(A_ineq, dtype=float)
        # b_ineq_np = np.array(b_ineq, dtype=float)
        # A_eq_np   = np.array(A_eq  , dtype=float)
        # b_eq_np   = np.array(b_eq  , dtype=float)
        # # # # # # # # # # # # # # # # # # # #
        ## check the feasibility of the OPF:
        # check_flag = self.check_feasibility_of_a_set_of_inequality_and_equality_equations__using_Farkas_lemma(A_ineq_np, b_ineq_np, A_eq_np,b_eq_np)
        #
        # if check_flag == 'feasible':  # the set of inequality and equality equations are feasible. now we solve the OPF
        #
        #
        ##
        # remove formatting from the b_ineq and b_eq
        b_ineq_no_formatting = []
        if b_ineq.shape[0] >= b_ineq.shape[1]: # it is a vertical ndarray
            for i in range(b_ineq.shape[0]):
                b_ineq_no_formatting = b_ineq_no_formatting + [b_ineq[i,0]]
        else: # it is a horizontal ndarray
            for i in range(b_ineq.shape[1]):
                b_ineq_no_formatting = b_ineq_no_formatting + [b_ineq[0,i]]
        #
        b_eq_no_formatting = []
        if b_eq.shape[0] >= b_eq.shape[1]:  # it is a vertical ndarray
            for i in range(b_eq.shape[0]):
                b_eq_no_formatting = b_eq_no_formatting + [b_eq[i, 0]]
        else:  # it is a horizontal ndarray
            for i in range(b_eq.shape[1]):
                b_eq_no_formatting = b_eq_no_formatting + [b_eq[0, i]]
        #
        ##
        H_cvxpy = np.array(H, dtype=float)
        f_cvxpy = np.array(f, dtype=float).reshape((2*self.num_of_nodes , 1))
        A_ineq_cvxpy = np.array(A_ineq, dtype=float)
        b_ineq_cvxpy = np.array(b_ineq_no_formatting, dtype=float)
        A_eq_cvxpy = np.array(A_eq, dtype=float)
        b_eq_cvxpy = np.array(b_eq_no_formatting, dtype=float)
        # # # # # # # # # # # # # # # # # # # #
        number_of_variable = 2*self.num_of_nodes
        x_variables_cvxpy = cvxpy.Variable(number_of_variable)
        #
        obj_fun_cvxpy = 0.5 * cvxpy.quad_form(x_variables_cvxpy, H_cvxpy) + f_cvxpy.T * x_variables_cvxpy.T + fixed_costs
        #
        constraints_equl_and_inequl = [A_eq_cvxpy * x_variables_cvxpy.T == b_eq_cvxpy, A_ineq_cvxpy * x_variables_cvxpy.T <= b_ineq_cvxpy]
        # constraints_equl_and_inequl = [A_eq_cvxpy.dot(x_variables_cvxpy.T) == b_eq_cvxpy, A_ineq_cvxpy.dot(x_variables_cvxpy.T) <= b_ineq_cvxpy]
        # constraints_equl_and_inequl = [A_eq_cvxpy.__matmul__(x_variables_cvxpy.T) == b_eq_cvxpy, A_ineq_cvxpy.__matmul__(x_variables_cvxpy.T) <= b_ineq_cvxpy]
        # constraints_equl_and_inequl = [(A_eq_cvxpy.__matmul__(x_variables_cvxpy.T)).__eq__(b_eq_cvxpy) , (A_ineq_cvxpy.__matmul__(x_variables_cvxpy.T)).__le__(b_ineq_cvxpy)]
        # constraints_equl_and_inequl = [A_eq_cvxpy * x_variables_cvxpy.T == b_eq_cvxpy, A_ineq_cvxpy * x_variables_cvxpy.T <= b_ineq_cvxpy]
        #

        #
        problem__DA_dc_OPF = cvxpy.Problem(cvxpy.Minimize(obj_fun_cvxpy), constraints_equl_and_inequl)
        #
        try:
            value_of_obj_fun_at_optimal_point = problem__DA_dc_OPF.solve(solver=cvxpy.GUROBI, verbose=True , max_iter=400000,eps_abs=10**(-12), eps_rel=10**(-12))  # note that this command will solve the obtimization and returns the value ...
            # ... of the objective function at the optimal point
        except cvxpy.error.SolverError:
            DC_power_flow_has_feasible_solution = False  # it does not
            nodal_injections = np.array([])
            LMPs = np.array([])
            list_of_lines_congested_or_not = np.array([])
            return nodal_injections, LMPs, list_of_lines_congested_or_not, DC_power_flow_has_feasible_solution
        #
        # print("status:", problem__DA_dc_OPF.status)
        if problem__DA_dc_OPF.status == 'optimal':
            print("status:                                                               ", problem__DA_dc_OPF.status)
        else:
            print("status:", problem__DA_dc_OPF.status)
        #
        result_OPF_cvxpy = x_variables_cvxpy.value
        #
        #  solution to the optimization (optimal point)                        =   x_variables_cvxpy.value
        #  Lagrangian variables for equality constraint                        =   constraints_equl_and_inequl[0].dual_value
        #  Lagrangian variables for inequality constraint                      =   constraints_equl_and_inequl[1].dual_value
        #  value of objective function at optimal point in the primal problem  =   value_of_obj_fun_at_optimal_point
        #
        #
        #
        if problem__DA_dc_OPF.status == 'optimal':
            DC_power_flow_has_feasible_solution = True  # it does
        else:
            DC_power_flow_has_feasible_solution = False  # it is infeasible or unbounded
        #
        if DC_power_flow_has_feasible_solution == False:
            # print('The OPF solved by ISO does not have feasible solution')
            DC_power_flow_has_feasible_solution = False  # it does not
            nodal_injections = np.array([])
            LMPs = np.array([])
            list_of_lines_congested_or_not = np.array([])
        else: # DC power flow has a feasible solution
            #
            #
            ## computing the congestion pattern
            list_of_lines_congested_or_not = copy.deepcopy([0]*self.num_of_branches)
            the_evaluated_inequalities__must_be_small_and_nonpositive = A_ineq_cvxpy.__matmul__(x_variables_cvxpy.value) - b_ineq_cvxpy
            the_evaluated_inequalities__must_be_small_and_nonpositive = the_evaluated_inequalities__must_be_small_and_nonpositive.reshape(the_evaluated_inequalities__must_be_small_and_nonpositive.size , 1)
            #
            # print(the_evaluated_inequalities__must_be_small_and_nonpositive[:2*self.num_of_branches , 0])
            #
            #
            tolerance_parameter_for_considering_congestion = 10**(-3)
            for i in range(0,self.num_of_branches):
                if the_evaluated_inequalities__must_be_small_and_nonpositive[i] >= -1*tolerance_parameter_for_considering_congestion:
                    # line is congested in the defined direction
                    list_of_lines_congested_or_not[i] = 1
                elif the_evaluated_inequalities__must_be_small_and_nonpositive[self.num_of_branches + i] >= -1*tolerance_parameter_for_considering_congestion:
                    # line is congested in the reverse direction of the defined direction
                    list_of_lines_congested_or_not[i] = 2
                else:
                    pass
            #
            nodal_injections = x_variables_cvxpy.value[0:self.num_of_nodes]
            LMPs = constraints_equl_and_inequl[0].dual_value[0:self.num_of_nodes]
        #
        #
        #
        #
        return nodal_injections , LMPs , list_of_lines_congested_or_not , DC_power_flow_has_feasible_solution
        #
        #
        #
        #
    #
    #
    #
    def solve_RT_economic_dispatch_by_ISO(self , Realizations_of_the_RPPs , generations_of_the_DA_generators ):
        #
        # this function returns:
        # nodal_injections , LMPs , list_of_lines_congested_or_not , DC_power_flow_has_feasible_solution
        #
        #
        # DC_power_flow_has_feasible_solution = is a variable that takes ...
        # ... a value of True (if feasible) or False (if infeasible). If the rsultant DC power flow of the ...
        # ... combination of current DA commitment of the RPPs, the nodal demands, the line capacities, the ...
        # ... structure of the network, and the location of the DA conventional generators gives us a feasible ...
        # ... solution, then we will have a feasible solution, and current set of DA commitments ...
        # ... of the RPPs MAY be a pure NE. If there is no feasible solution to the DC ...
        # ... power flow, then definitely the current DA commitments are not a pure NE.
        #
        #
        #
        # list_of_lines_congested_or_not  is a 1*num_of_branches numpy vector where if the line #i is ...
        # ... congested in the direction of definition, the the i th element of this vector is 1, and if the ...
        # ... line #i is congested in the reverse direction of its definition,the i th element of this vector is -1, ...
        # ... and if the line #i is not congested, the i th element of this vector is 0.
        #
        #
        #
        #
        #
        #
        #
        # this function calculates the optimal power generation of the RT generators and the corresponding ...
        # ... nodal angles of the DC OPF.
        #
        # the variables of the DC power flow optimization are:
        # x = [ nodal injections ; nodal angles]
        #
        # note that always the nodal angle of the reference node is zero (it is set in the equality constraint too)
        #
        #
        # we are trying to minimize 0.5 * xT * H * x + fT * x
        #
        # note that the vector of variables x consists of two set of variables:
        #   => the productions of RT generators
        #   => the nodal angles
        #
        # note that here we assume that there is one RT conv generators at each ...
        # node. then for those nodes that do not have any conv generators, we make ...
        # the corresponding parameters in H and f equal to zero.
        #
        # The variable vector x consists of 2*num_of_nodes variables: the first num_of_nodes ...
        # ... variables correspond to the RT conv injections, and the second num_of_nodes ...
        # ... variables correspond to the nodal angles.
        #
        #
        #
        #
        ## -------------------------------------------------------------------------------------------------------------
        ## Calculating H and f (Quadratic and linear coefficient terms in the objective function of the OPF)
        H = copy.deepcopy(np.zeros(((self.num_of_nodes + self.num_of_nodes) , (self.num_of_nodes + self.num_of_nodes))))
        f = copy.deepcopy(np.zeros(((self.num_of_nodes + self.num_of_nodes), 1)))
        fixed_costs = 0
        ##
        ## forming the quadratic and linear matrix and vector in the objective function
        for i in range(self.num_of_nodes):
            #
            sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i = 0
            sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i = 0
            #
            for j in range(self.num_of_RT_conv_generators):
                if int(self.data_of_RT_conventional_generators[j, 0]) == int(i):
                    sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i = sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i + 1/self.data_of_RT_conventional_generators[j, 1]
                    sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i = sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i + self.data_of_RT_conventional_generators[j, 2]/self.data_of_RT_conventional_generators[j, 1]
                else:
                    pass
            #
            if sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i != 0:
                H[i, i] = 1/sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i
                f[i, 0] = sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i / sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i
                fixed_costs = fixed_costs + copy.deepcopy( self.data_of_RT_conventional_generators[j, 3] )
            else:
                pass
            #
            #
        #
        net_nodal_withdraws = copy.deepcopy(np.array(self.vector_of_nodal_demands, dtype=float))
        for i in range(self.num_of_RPPs):
            the_node_of_current_RPP = int(self.data_of_RPPs[i, 0])
            # net_nodal_withdraws[the_node_of_current_RPP] = net_nodal_withdraws[the_node_of_current_RPP] + DA_commitments_of_RPPs[i] - Realizations_of_the_RPPs[i]
            net_nodal_withdraws[the_node_of_current_RPP] = net_nodal_withdraws[the_node_of_current_RPP] - Realizations_of_the_RPPs[i]
        #
        for i in range(self.num_of_DA_conv_generators):
            the_node_of_current_DA_generator = int(self.data_of_DA_conventional_generators[i, 0])
            net_nodal_withdraws[the_node_of_current_DA_generator] = net_nodal_withdraws[the_node_of_current_DA_generator] - np.array(generations_of_the_DA_generators)[i]
        #
        #
        #
        ## -------------------------------------------------------------------------------------------------------------
        # forming the A_eq and b_eq matrices for the optimization
        # KCL for all nodes
        # A_eq = [(-1 * np.eye(self.num_of_nodes, self.num_of_nodes)), self.susceptance_matrix]
        # b_eq = -1 * net_nodal_withdraws
        A_eq = copy.deepcopy( np.concatenate( ((-1 * np.eye(self.num_of_nodes, self.num_of_nodes)), self.susceptance_matrix) , axis = 1 ) )
        b_eq = copy.deepcopy( -1 * net_nodal_withdraws.reshape(self.num_of_nodes , 1) )
        #
        #
        ## setting the injection of the nodes without RT generator eqaul to zero
        if set(self.data_of_RT_conventional_generators[:,0]).issubset(set(list(range(0,self.num_of_nodes)))): # there exists some nodes without RT generator on them
            set_of_nodes_without_RT_generator = set(list(range(0,self.num_of_nodes))) - set(self.data_of_RT_conventional_generators[:,0])
            num_of_nodes_without_RT_generator = len( set_of_nodes_without_RT_generator )
            addtional_block_to__A_eq = np.zeros((num_of_nodes_without_RT_generator, 2 * self.num_of_nodes))
            addtional_block_to__b_eq = np.zeros((num_of_nodes_without_RT_generator, 1))
            temp_counter = 0
            for i in set_of_nodes_without_RT_generator: # the insert power at node i is zero because the there is no conventional RT generator at this node
                addtional_block_to__A_eq[temp_counter, i] = 1;
                addtional_block_to__b_eq[temp_counter, 0] = 0;
                temp_counter += 1;
                #
            #
            # A_eq = [A_eq ; addtional_block_to__A_eq];
            # b_eq = [b_eq ; addtional_block_to__b_eq];
            A_eq = np.concatenate((A_eq , addtional_block_to__A_eq) , axis = 0)
            b_eq = np.concatenate((b_eq, addtional_block_to__b_eq), axis=0)
        else:
            pass
        #
        #
        ## setting the angle of the slack node equal to zero node equal to zero
        last_row_of_A_eq = np.zeros((1, 2 * self.num_of_nodes));
        last_row_of_A_eq[0, (self.index_of_slack_node - self.num_of_nodes)] = 1;
        # A_eq = [A_eq ; last_row_of_A_eq];
        # b_eq = [b_eq ; 0];
        A_eq = np.concatenate((A_eq, last_row_of_A_eq), axis=0)
        b_eq = np.concatenate((b_eq, np.array([0]).reshape((1,1))), axis=0)
        #
        #
        #
        ## -------------------------------------------------------------------------------------------------------------
        ## Calculating A_ineq and b_ineq. Note that these inequalities are for:
        #  ... (1) line flows on the defined directions
        #  ... (2) line flows on the reverse directions
        #  ... (3) upper bounds of the variables
        #  ... (4) lower bounds of the variables
        #
        A_ineq = copy.deepcopy(np.concatenate( (  np.zeros((self.num_of_branches, self.num_of_nodes))  , self.matrix_of_angles_to_line_flows ) , axis = 1 ))
        #
        # to consider the reverse power flow throgh lines too
        A_ineq = copy.deepcopy( np.concatenate( (A_ineq , -1*A_ineq) , axis = 0 ) )
        #
        b_ineq = copy.deepcopy( np.concatenate((self.branch_capacities.reshape((self.num_of_branches,1)) , self.branch_capacities.reshape((self.num_of_branches,1))) , axis = 0) )
        #
        #
        ## note that we need to set a lower bound for the generators to be no less than total commitment of the RPPs
        # this is because in our formulation, we did allow the generators to produce negative values too.
        #
        #
        #
        ##  DC power flow
        # x0 = np.zeros((2 * num_of_nodes, 1)).reshape(2 * num_of_nodes, 1);
        # [x, fval, exitflag, output, lambda ] = quadprog(H, f, A, b, A_eq, b_eq, lb, ub, x0, opt_temp);
        # H_np      = np.array(H     ,dtype=float)
        # f_np      = np.array(f     , dtype=float)
        # A_ineq_np = np.array(A_ineq, dtype=float)
        # b_ineq_np = np.array(b_ineq, dtype=float)
        # A_eq_np   = np.array(A_eq  , dtype=float)
        # b_eq_np   = np.array(b_eq  , dtype=float)
        # # # # # # # # # # # # # # # # # # # #

        # # # # # # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
        # begining of implementing with CVXPY
        ##
        # remove formatting from the b_ineq and b_eq
        b_ineq_no_formatting = []
        if b_ineq.shape[0] >= b_ineq.shape[1]: # it is a vertical ndarray
            for i in range(b_ineq.shape[0]):
                b_ineq_no_formatting = b_ineq_no_formatting + [b_ineq[i,0]]
        else: # it is a horizontal ndarray
            for i in range(b_ineq.shape[1]):
                b_ineq_no_formatting = b_ineq_no_formatting + [b_ineq[0,i]]
        #
        b_eq_no_formatting = []
        if b_eq.shape[0] >= b_eq.shape[1]:  # it is a vertical ndarray
            for i in range(b_eq.shape[0]):
                b_eq_no_formatting = b_eq_no_formatting + [b_eq[i, 0]]
        else:  # it is a horizontal ndarray
            for i in range(b_eq.shape[1]):
                b_eq_no_formatting = b_eq_no_formatting + [b_eq[0, i]]
        #
        ##
        H_cvxpy = np.array(H, dtype=float)
        f_cvxpy = np.array(f, dtype=float).reshape((2*self.num_of_nodes , 1))
        A_ineq_cvxpy = np.array(A_ineq, dtype=float)
        b_ineq_cvxpy = np.array(b_ineq_no_formatting, dtype=float)
        A_eq_cvxpy = np.array(A_eq, dtype=float)
        b_eq_cvxpy = np.array(b_eq_no_formatting, dtype=float)
        # # # # # # # # # # # # # # # # # # # #
        number_of_variable = 2*self.num_of_nodes
        x_variables_cvxpy = cvxpy.Variable(number_of_variable)
        #
        obj_fun_cvxpy = 0.5 * cvxpy.quad_form(x_variables_cvxpy, H_cvxpy) + f_cvxpy.T * x_variables_cvxpy.T + fixed_costs
        #
        constraints_equl_and_inequl = [A_eq_cvxpy * x_variables_cvxpy.T == b_eq_cvxpy, A_ineq_cvxpy * x_variables_cvxpy.T <= b_ineq_cvxpy]
        # constraints_equl_and_inequl = [A_eq_cvxpy.dot(x_variables_cvxpy.T) == b_eq_cvxpy, A_ineq_cvxpy.dot(x_variables_cvxpy.T) <= b_ineq_cvxpy]
        # constraints_equl_and_inequl = [A_eq_cvxpy.__matmul__(x_variables_cvxpy.T) == b_eq_cvxpy, A_ineq_cvxpy.__matmul__(x_variables_cvxpy.T) <= b_ineq_cvxpy]
        # constraints_equl_and_inequl = [(A_eq_cvxpy.__matmul__(x_variables_cvxpy.T)).__eq__(b_eq_cvxpy) , (A_ineq_cvxpy.__matmul__(x_variables_cvxpy.T)).__le__(b_ineq_cvxpy)]
        # constraints_equl_and_inequl = [A_eq_cvxpy * x_variables_cvxpy.T == b_eq_cvxpy, A_ineq_cvxpy * x_variables_cvxpy.T <= b_ineq_cvxpy]
        #

        #
        problem__RT_dc_OPF = cvxpy.Problem(cvxpy.Minimize(obj_fun_cvxpy), constraints_equl_and_inequl)
        #
        try:
            value_of_obj_fun_at_optimal_point = problem__RT_dc_OPF.solve(solver=cvxpy.GUROBI, verbose=True , max_iter=400000,eps_abs=10**(-12), eps_rel=10**(-12))  # note that this command will solve the obtimization and returns the value ...
            # ... of the objective function at the optimal point
        except cvxpy.error.SolverError:
            DC_power_flow_has_feasible_solution = False  # it does not
            nodal_injections = np.array([])
            LMPs = np.array([])
            list_of_lines_congested_or_not = np.array([])
            return nodal_injections , LMPs , list_of_lines_congested_or_not , DC_power_flow_has_feasible_solution
        #
        # print("status:", problem__RT_dc_OPF.status)
        if problem__RT_dc_OPF.status == 'optimal':
            print("status:                                                               ", problem__RT_dc_OPF.status)
        else:
            print("status:", problem__RT_dc_OPF.status)
        #
        result_OPF_cvxpy = x_variables_cvxpy.value
        #
        #  solution to the optimization (optimal point)                        =   x_variables_cvxpy.value
        #  Lagrangian variables for equality constraint                        =   constraints_equl_and_inequl[0].dual_value
        #  Lagrangian variables for inequality constraint                      =   constraints_equl_and_inequl[1].dual_value
        #  value of objective function at optimal point in the primal problem  =   value_of_obj_fun_at_optimal_point
        #
        #
        #
        if problem__RT_dc_OPF.status == 'optimal':
            DC_power_flow_has_feasible_solution = True  # it does
        else:
            DC_power_flow_has_feasible_solution = False  # it is infeasible or unbounded
        #
        if DC_power_flow_has_feasible_solution == False:
            # print('The OPF solved by ISO does not have feasible solution')
            DC_power_flow_has_feasible_solution = False  # it does not
            nodal_injections = np.array([])
            LMPs = np.array([])
            list_of_lines_congested_or_not = np.array([])
        else: # DC power flow has a feasible solution
            #
            #
            ## computing the congestion pattern
            list_of_lines_congested_or_not = copy.deepcopy( [0]*self.num_of_branches )
            the_evaluated_inequalities__must_be_small_and_nonpositive = A_ineq_cvxpy.__matmul__(x_variables_cvxpy.value) - b_ineq_cvxpy
            the_evaluated_inequalities__must_be_small_and_nonpositive = the_evaluated_inequalities__must_be_small_and_nonpositive.reshape(the_evaluated_inequalities__must_be_small_and_nonpositive.size , 1)
            #
            # print(the_evaluated_inequalities__must_be_small_and_nonpositive[:2*self.num_of_branches , 0])
            #
            #
            tolerance_parameter_for_considering_congestion = 10**(-6)
            for i in range(0,self.num_of_branches):
                if the_evaluated_inequalities__must_be_small_and_nonpositive[i] >= -1*tolerance_parameter_for_considering_congestion:
                    # line is congested in the defined direction
                    list_of_lines_congested_or_not[i] = 1
                elif the_evaluated_inequalities__must_be_small_and_nonpositive[self.num_of_branches + i] >= -1*tolerance_parameter_for_considering_congestion:
                    # line is congested in the reverse direction of the defined direction
                    list_of_lines_congested_or_not[i] = 2
                else:
                    pass
            #
            nodal_injections = x_variables_cvxpy.value[0:self.num_of_nodes]
            LMPs = constraints_equl_and_inequl[0].dual_value[0:self.num_of_nodes]
        #
        #
        #
        #
        return nodal_injections , LMPs , list_of_lines_congested_or_not , DC_power_flow_has_feasible_solution
    #
    #
    #
    def find_the_NEs___case_same_DA_and_RT_cong_pttrns( self ):
        #
        # this function finds the pure NEs assuming that the RT congestion pattern is the same as DA cong pttrn
        #
        #
        matrix_of_all_critical_line_congestions_in_DA_market = np.array(self.find_all_allowed_congestions_patterns__both_congested_and_not_congested_lines_in_DA_market())
        #
        if matrix_of_all_critical_line_congestions_in_DA_market.size == self.num_of_branches:
            matrix_of_all_critical_line_congestions_in_DA_market = matrix_of_all_critical_line_congestions_in_DA_market.reshape(1,self.num_of_branches)
            num_of_congestion_in_DA_market = 1
        else:
            num_of_congestion_in_DA_market = matrix_of_all_critical_line_congestions_in_DA_market.shape[0]
        #
        # # # # # # # # #
        #
        #
        list_of_DA_commitments_at_NEs = []
        list_of_congestions_at_NEs_in_DA_market = []
        list_of_congestions_at_NEs_in_RT_market = []
        #
        list_of_DA_commitments_calculated_by_algorithm_feasible_ALL = []
        list_of_congestions_in_DA_market_assumed_by_algorithm_feasible_ALL = []
        list_of_congestions_in_DA_market_calculated_by_ISO_feasible_ALL = []
        list_of_generation_of_DA_generators_ALL = []
        list_of_expected_generation_of_RT_generators_ALL = []
        #
        number_of_all_possible_NEs = num_of_congestion_in_DA_market
        num_of_true_NEs = 0
        for i in range(num_of_congestion_in_DA_market):
            #
            print('the percentage of process of checking NEs is {}'.format(i / number_of_all_possible_NEs))
            #
            current_DA_congestion_pattern = matrix_of_all_critical_line_congestions_in_DA_market[i,:]
            #
            current_DA_congestion_pattern = np.array([int(ttttt_temp) for ttttt_temp in current_DA_congestion_pattern.tolist()])
            #
            current_RT_congestion_pattern = current_DA_congestion_pattern
            #
            #
            DA_commitments_of_RPPs_for_current_cong_pattern, generation_of_DA_generators, expected_generation_of_RT_generator, _ , _ , _ , _ = self.find_DA_comtmnts_of_RPPs_for_curnt_DA_and_RT_cong_patterns( current_DA_congestion_pattern , current_RT_congestion_pattern)
            #
            # DA_commitments_of_RPPs_for_current_cong_pattern = np.zeros((1,10))
            #
            _ , _ , cong_pttrn_from_ISO_in_DA_market , does_DC_power_flow_has_feasible_solution_or_not = self.solve_DA_economic_dispatch_by_ISO( DA_commitments_of_RPPs_for_current_cong_pattern)
            #
            #
            if does_DC_power_flow_has_feasible_solution_or_not:
                #
                list_of_DA_commitments_calculated_by_algorithm_feasible_ALL = list_of_DA_commitments_calculated_by_algorithm_feasible_ALL + DA_commitments_of_RPPs_for_current_cong_pattern.tolist()
                list_of_congestions_in_DA_market_assumed_by_algorithm_feasible_ALL = list_of_congestions_in_DA_market_assumed_by_algorithm_feasible_ALL + [ current_DA_congestion_pattern.tolist() ]
                list_of_congestions_in_DA_market_calculated_by_ISO_feasible_ALL = list_of_congestions_in_DA_market_calculated_by_ISO_feasible_ALL + [ cong_pttrn_from_ISO_in_DA_market ]
                list_of_generation_of_DA_generators_ALL = list_of_generation_of_DA_generators_ALL + generation_of_DA_generators.tolist()
                list_of_expected_generation_of_RT_generators_ALL = list_of_expected_generation_of_RT_generators_ALL + expected_generation_of_RT_generator.tolist()
                #
                if int(sum(abs(cong_pttrn_from_ISO_in_DA_market - current_DA_congestion_pattern))) == 0:
                    # this means that there is a Nash Equilibrium
                    num_of_true_NEs += 1
                    list_of_DA_commitments_at_NEs = list_of_DA_commitments_at_NEs + DA_commitments_of_RPPs_for_current_cong_pattern.tolist()
                    list_of_congestions_at_NEs_in_DA_market = list_of_congestions_at_NEs_in_DA_market + [ list( current_DA_congestion_pattern ) ]
                    list_of_congestions_at_NEs_in_RT_market = list_of_congestions_at_NEs_in_RT_market + [list(current_RT_congestion_pattern)]
                else:
                    pass
                #
            else: # i.e. the DC power flow does not have any feasible solution, and we definitely have no pure NE.
                pass # do nothing because current DA commitments of the RPPs can not be accepted by the ISO.
        #
        #
        return list_of_DA_commitments_at_NEs , list_of_congestions_at_NEs_in_DA_market , list_of_congestions_at_NEs_in_RT_market , list_of_DA_commitments_calculated_by_algorithm_feasible_ALL , list_of_congestions_in_DA_market_assumed_by_algorithm_feasible_ALL , list_of_congestions_in_DA_market_calculated_by_ISO_feasible_ALL, list_of_generation_of_DA_generators_ALL, list_of_expected_generation_of_RT_generators_ALL
    #
    #
    #
    #
    #
    #
    def find_the_NEs___case_different_DA_and_RT_cong_pttrns( self ):
        #
        # this function finds the pure NEs assuming that the RT congestion pattern is different from the DA cong pttrn
        #
        #
        matrix_of_all_critical_line_congestions_in_DA_market = np.array(self.find_all_allowed_congestions_patterns__both_congested_and_not_congested_lines_in_DA_market())
        #
        if matrix_of_all_critical_line_congestions_in_DA_market.size == self.num_of_branches:
            matrix_of_all_critical_line_congestions_in_DA_market = matrix_of_all_critical_line_congestions_in_DA_market.reshape(1,self.num_of_branches)
            num_of_congestion_in_DA_market = 1
        else:
            num_of_congestion_in_DA_market = matrix_of_all_critical_line_congestions_in_DA_market.shape[0]
        #
        # # # # # # # # #
        matrix_of_all_critical_line_congestions_in_RT_market = np.array(self.find_all_allowed_congestions_patterns__both_congested_and_not_congested_lines_in_RT_market())
        #
        if matrix_of_all_critical_line_congestions_in_RT_market.size == self.num_of_branches:
            matrix_of_all_critical_line_congestions_in_RT_market = matrix_of_all_critical_line_congestions_in_RT_market.reshape(1, self.num_of_branches)
            num_of_congestion_in_RT_market = 1
        else:
            num_of_congestion_in_RT_market = matrix_of_all_critical_line_congestions_in_RT_market.shape[0]
        #
        list_of_DA_commitments_at_NEs = []
        list_of_congestions_at_NEs_in_DA_market = []
        list_of_congestions_at_NEs_in_RT_market = []
        #
        number_of_all_possible_NEs = num_of_congestion_in_DA_market * num_of_congestion_in_RT_market
        num_of_true_NEs = 0
        for i in range(num_of_congestion_in_DA_market):
            #
            #
            current_DA_congestion_pattern = matrix_of_all_critical_line_congestions_in_DA_market[i,:]
            #
            for j in range(num_of_congestion_in_RT_market):
                #
                print('the percentage of process of checking NEs is {}'.format(((i*num_of_congestion_in_RT_market) + j + 1) / number_of_all_possible_NEs))
                #
                current_RT_congestion_pattern = matrix_of_all_critical_line_congestions_in_RT_market[j, :]
                #
                # print('current_DA_congestion_pattern is {}'.format(current_DA_congestion_pattern))
                # print('current_RT_congestion_pattern is {}'.format(current_RT_congestion_pattern))
                #
                #
                #
                DA_commitments_of_RPPs_for_current_cong_pattern, _ , _ , _ , _ = self.find_DA_comtmnts_of_RPPs_for_curnt_DA_and_RT_cong_patterns( current_DA_congestion_pattern , current_RT_congestion_pattern)
                #
                # print('DA_commitments_of_RPPs_for_current_cong_pattern  =  {}'.format(DA_commitments_of_RPPs_for_current_cong_pattern))
                #
                _ , _ , cong_pttrn_from_ISO_in_DA_market , does_DC_power_flow_has_feasible_solution_or_not = self.solve_DA_economic_dispatch_by_ISO( DA_commitments_of_RPPs_for_current_cong_pattern)
                #
                #
                if does_DC_power_flow_has_feasible_solution_or_not:
                    if int(sum(abs(cong_pttrn_from_ISO_in_DA_market - current_DA_congestion_pattern))) == 0:
                        # this means that there is a Nash Equilibrium
                        num_of_true_NEs += 1
                        list_of_DA_commitments_at_NEs = list_of_DA_commitments_at_NEs + DA_commitments_of_RPPs_for_current_cong_pattern.tolist()
                        list_of_congestions_at_NEs_in_DA_market = list_of_congestions_at_NEs_in_DA_market + [ list( current_DA_congestion_pattern ) ]
                        list_of_congestions_at_NEs_in_RT_market = list_of_congestions_at_NEs_in_RT_market + [list(current_RT_congestion_pattern)]
                    else:
                        pass
                    #
                else: # i.e. the DC power flow does not have any feasible solution, and we definitely have no pure NE.
                    pass # do nothing because current DA commitments of the RPPs can not be accepted by the ISO.
        #
        #
        return list_of_DA_commitments_at_NEs , list_of_congestions_at_NEs_in_DA_market , list_of_congestions_at_NEs_in_RT_market
    #
    #
    #
    def find_DA_comtmnts_of_RPPs_for_curnt_DA_and_RT_cong_patterns( self, current_DA_congestion_pattern , current_RT_congestion_pattern):
        # this function calculates the DA commitments of the RPPs at the Nash Equilibrium ...
        #
        #
        # DA_commitments_of_RPPs_for_current_cong_pattern   is the DA commitments of the RPPs at the Nash Equilibrium.
        #
        #
        # vector_of_DA_nodal_demands = self.vector_of_nodal_demands
        vector_of_DA_nodal_demands = np.array(self.vector_of_nodal_demands , dtype=float).tolist()
        Total_DA_loads = 0
        for i in range(self.num_of_nodes):
            Total_DA_loads += vector_of_DA_nodal_demands[i]
        #
        #
        #
        if (sum([int(ttt_temp) for ttt_temp in (current_DA_congestion_pattern == [0]*self.num_of_branches)]) == self.num_of_branches) and (sum([int(ttt_temp) for ttt_temp in (current_RT_congestion_pattern == [0]*self.num_of_branches)]) == self.num_of_branches):
            # # there is no congestion in DA market and no congestion in the RT market
            #
            #
            #
            # # # # # # # # # # # # # #      DA Calculations      # # # # # # # # # # #
            #
            #
            #
            vector_of_inverse_alpha_g_DA = np.zeros((self.num_of_DA_conv_generators , 1)).reshape(self.num_of_DA_conv_generators , 1)
            for i in range(self.num_of_DA_conv_generators):
                vector_of_inverse_alpha_g_DA[i,0] = 1/(self.data_of_DA_conventional_generators[i,1])
            #
            vector_of_beta_over_alpha_g_DA = np.zeros((self.num_of_DA_conv_generators , 1)).reshape(self.num_of_DA_conv_generators , 1)
            for i in range(self.num_of_DA_conv_generators):
                vector_of_beta_over_alpha_g_DA[i,0] = (self.data_of_DA_conventional_generators[i,2])/(self.data_of_DA_conventional_generators[i,1])
            #
            diagonal_matrix_of_inverse_alpha_g_DA = np.zeros((self.num_of_DA_conv_generators , self.num_of_DA_conv_generators)).reshape(self.num_of_DA_conv_generators , self.num_of_DA_conv_generators)
            for i in range(self.num_of_DA_conv_generators):
                diagonal_matrix_of_inverse_alpha_g_DA[i,i] = 1/(self.data_of_DA_conventional_generators[i,1])
            #
            F__DA = ( (-1) / ( np.matmul(np.ones((1,self.num_of_DA_conv_generators)).reshape(1,self.num_of_DA_conv_generators) , vector_of_inverse_alpha_g_DA) ) ) * np.ones((self.num_of_nodes , self.num_of_RPPs)).reshape(self.num_of_nodes , self.num_of_RPPs)
            #
            F__DA_prime = ( 1 / ( np.matmul(np.ones((1,self.num_of_DA_conv_generators)).reshape(1,self.num_of_DA_conv_generators) , vector_of_inverse_alpha_g_DA) ) ) * (  np.matmul( np.ones((self.num_of_nodes , self.num_of_nodes)) , np.array(self.vector_of_nodal_demands).reshape(self.num_of_nodes , 1)  )   +   np.matmul( np.ones((self.num_of_nodes , self.num_of_DA_conv_generators)) , vector_of_beta_over_alpha_g_DA   )  )
            #
            #
            #
            K__DA = np.matmul(diagonal_matrix_of_inverse_alpha_g_DA , F__DA[self.list_of_indices_of_nodes_with_DA_conv_generator , :])
            K__DA = K__DA.reshape(self.num_of_DA_conv_generators , self.num_of_RPPs)
            #
            K__DA_prime = np.matmul(diagonal_matrix_of_inverse_alpha_g_DA , F__DA_prime[self.list_of_indices_of_nodes_with_DA_conv_generator , :]) - vector_of_beta_over_alpha_g_DA
            K__DA_prime = K__DA_prime.reshape(self.num_of_DA_conv_generators , 1)
            #
            #
            #
            #
            #
            #
            # # # # # # # # # # # # # #      RT Calculations      # # # # # # # # # # #
            #
            #
            vector_of_inverse_alpha_g_RT = np.zeros((self.num_of_RT_conv_generators, 1)).reshape(
                self.num_of_RT_conv_generators, 1)
            for i in range(self.num_of_RT_conv_generators):
                vector_of_inverse_alpha_g_RT[i, 0] = 1 / (self.data_of_RT_conventional_generators[i, 1])
            #
            vector_of_beta_over_alpha_g_RT = np.zeros((self.num_of_RT_conv_generators, 1)).reshape(
                self.num_of_RT_conv_generators, 1)
            for i in range(self.num_of_RT_conv_generators):
                vector_of_beta_over_alpha_g_RT[i, 0] = (self.data_of_RT_conventional_generators[i, 2]) / (
                self.data_of_RT_conventional_generators[i, 1])
            #
            diagonal_matrix_of_inverse_alpha_g_RT = np.zeros( (self.num_of_RT_conv_generators, self.num_of_RT_conv_generators)).reshape( self.num_of_RT_conv_generators, self.num_of_RT_conv_generators)
            for i in range(self.num_of_RT_conv_generators):
                diagonal_matrix_of_inverse_alpha_g_RT[i, i] = 1 / (self.data_of_RT_conventional_generators[i, 1])
            #
            #
            F__RT = ((-1) / (np.matmul(np.ones((1, self.num_of_RT_conv_generators)).reshape(1,self.num_of_RT_conv_generators) , vector_of_inverse_alpha_g_RT))) * np.matmul(np.ones((self.num_of_nodes , self.num_of_DA_conv_generators)).reshape(self.num_of_nodes , self.num_of_DA_conv_generators) , K__DA)
            #
            F__RT_prime = ((-1) / (np.matmul(np.ones((1, self.num_of_RT_conv_generators)).reshape(1,self.num_of_RT_conv_generators) , vector_of_inverse_alpha_g_RT))) * np.ones((self.num_of_nodes , self.num_of_RPPs)).reshape(self.num_of_nodes , self.num_of_RPPs)
            #
            F__RT_double_prime__part_1 = (1 / (np.matmul(np.ones((1, self.num_of_RT_conv_generators)).reshape(1,self.num_of_RT_conv_generators) , vector_of_inverse_alpha_g_RT))) * np.matmul( np.ones((self.num_of_nodes , self.num_of_nodes)).reshape(self.num_of_nodes , self.num_of_nodes) , np.array(self.vector_of_nodal_demands).reshape(self.num_of_nodes , 1)  )
            #
            F__RT_double_prime__part_2 = (1 / (np.matmul(np.ones((1, self.num_of_RT_conv_generators)).reshape(1,self.num_of_RT_conv_generators) , vector_of_inverse_alpha_g_RT))) * np.matmul( np.ones((self.num_of_nodes , self.num_of_DA_conv_generators)).reshape(self.num_of_nodes , self.num_of_DA_conv_generators) , K__DA_prime  )
            #
            F__RT_double_prime__part_3 = (1 / (np.matmul(np.ones((1, self.num_of_RT_conv_generators)).reshape(1,self.num_of_RT_conv_generators) , vector_of_inverse_alpha_g_RT))) * np.matmul( np.ones((self.num_of_nodes, self.num_of_RT_conv_generators)).reshape(self.num_of_nodes, self.num_of_RT_conv_generators),vector_of_beta_over_alpha_g_RT)
            #
            F__RT_double_prime = F__RT_double_prime__part_1 - F__RT_double_prime__part_2 + F__RT_double_prime__part_3
            #
            #
            #
            K__RT              = np.matmul(diagonal_matrix_of_inverse_alpha_g_RT,F__RT[self.list_of_indices_of_nodes_with_RT_conv_generator, :])
            K__RT              = K__RT.reshape(self.num_of_RT_conv_generators, self.num_of_RPPs)
            K__RT_prime        = np.matmul(diagonal_matrix_of_inverse_alpha_g_RT,F__RT_prime[self.list_of_indices_of_nodes_with_RT_conv_generator, :])
            K__RT_prime        = K__RT_prime.reshape(self.num_of_RT_conv_generators , self.num_of_RPPs)
            K__RT_double_prime = np.matmul(diagonal_matrix_of_inverse_alpha_g_RT,F__RT_double_prime[self.list_of_indices_of_nodes_with_RT_conv_generator, :]) - vector_of_beta_over_alpha_g_RT
            K__RT_double_prime = K__RT_double_prime.reshape(self.num_of_RT_conv_generators , 1)
            #
            #
            #
            # # # # # # # # # # # #      final Calculations       # # # # # # # # # # #
            #
            #
            #
            #
            #
            Zeta_DA = np.zeros((self.num_of_RPPs, self.num_of_nodes))
            for i in range(self.num_of_RPPs):
                for j in range(self.num_of_nodes):
                    if j == self.data_of_RPPs[i, 0]:
                        Zeta_DA[i, j] = 1
                        break
                    else:
                        pass
                    #
                #
            #
            #
            Zeta_RT = Zeta_DA
            #
            #
            temp_vector_DA = [0] * self.num_of_RPPs
            for i in range(self.num_of_RPPs):
                index_of_node_of_current_RPP_i = int(self.data_of_RPPs[i, 0])
                temp_vector_DA[i] = F__DA[index_of_node_of_current_RPP_i, i]
            #
            #
            Zeta_DA_prime = np.diag(temp_vector_DA)
            #
            #
            temp_vector_RT = [0] * self.num_of_RPPs
            for i in range(self.num_of_RPPs):
                index_of_node_of_current_RPP_i = int(self.data_of_RPPs[i, 0])
                temp_vector_RT[i] = F__RT[index_of_node_of_current_RPP_i, i]
            #
            #
            Zeta_RT_prime = np.diag(temp_vector_RT)
            #
            #
            #
            #
            10 + 5
            ##
            # Delta_left = np.matmul(Zeta_DA, F__DA) + Zeta_DA_prime - Zeta_RT_prime - np.matmul(Zeta_RT, F__RT_prime)
            Delta_left = np.matmul(Zeta_DA, F__DA) + Zeta_DA_prime - Zeta_RT_prime - np.matmul(Zeta_RT, F__RT)
            #
            Delta_right = -1 * np.matmul(Zeta_DA, F__DA_prime) + np.matmul(Zeta_RT, F__RT_double_prime) + np.matmul(
                (np.matmul(Zeta_RT, F__RT_prime) - Zeta_RT_prime), self.data_of_RPPs[:, 1].reshape(self.num_of_RPPs, 1))
            #
            #
            # DA_commitments_of_RPPs_at_NE_for_current_cong_pattern = (Delta_left^(-1))*Delta_right
            DA_commitments_of_RPPs_at_NE_for_current_cong_pattern = np.matmul(np.linalg.pinv(Delta_left),
                                                                              Delta_right).reshape(1, self.num_of_RPPs)
            #
            #
            generation_of_DA_generators = np.matmul(K__DA, np.array(DA_commitments_of_RPPs_at_NE_for_current_cong_pattern).reshape(self.num_of_RPPs , 1)) + K__DA_prime
            #
            expected_generation_of_RT_generator = np.matmul(K__RT, np.array(DA_commitments_of_RPPs_at_NE_for_current_cong_pattern).reshape(self.num_of_RPPs , 1)) + np.matmul(K__RT_prime, np.array(self.data_of_RPPs[:,1]).reshape(self.num_of_RPPs , 1)) + K__RT_double_prime
            # LMP = np.matmul(F__DA,DA_commitments_of_RPPs_at_NE_for_current_cong_pattern.reshape(self.num_of_RPPs,1)) +  F__DA_prime
            # return DA_commitments_of_RPPs_at_NE_for_current_cong_pattern , F__DA , F__DA_prime , F__RT , F__RT_prime
            return DA_commitments_of_RPPs_at_NE_for_current_cong_pattern, generation_of_DA_generators, expected_generation_of_RT_generator, None, None, None, None
            #
            #
        elif (sum([int(ttt_temp) for ttt_temp in (current_DA_congestion_pattern == [0]*self.num_of_branches)]) == self.num_of_branches) and (sum([int(ttt_temp) for ttt_temp in (current_RT_congestion_pattern == [0]*self.num_of_branches)]) != self.num_of_branches):
            # we have no line congestion in DA market, but some lines are congested in the RT market
            #
            #
            # # # # # # # # # # # # # #      DA Calculations      # # # # # # # # # # #
            #
            #
            #
            vector_of_inverse_alpha_g_DA = np.zeros((self.num_of_DA_conv_generators, 1)).reshape(
                self.num_of_DA_conv_generators, 1)
            for i in range(self.num_of_DA_conv_generators):
                vector_of_inverse_alpha_g_DA[i, 0] = 1 / (self.data_of_DA_conventional_generators[i, 1])
            #
            vector_of_beta_over_alpha_g_DA = np.zeros((self.num_of_DA_conv_generators, 1)).reshape(self.num_of_DA_conv_generators, 1)
            for i in range(self.num_of_DA_conv_generators):
                vector_of_beta_over_alpha_g_DA[i, 0] = (self.data_of_DA_conventional_generators[i, 2]) / (self.data_of_DA_conventional_generators[i, 1])
            #
            diagonal_matrix_of_inverse_alpha_g_DA = np.zeros(
                (self.num_of_DA_conv_generators, self.num_of_DA_conv_generators)).reshape(self.num_of_DA_conv_generators, self.num_of_DA_conv_generators)
            for i in range(self.num_of_DA_conv_generators):
                diagonal_matrix_of_inverse_alpha_g_DA[i, i] = 1 / (self.data_of_DA_conventional_generators[i, 1])
            #
            F__DA = ((-1) / ( np.matmul(np.ones((1, self.num_of_DA_conv_generators)).reshape(1, self.num_of_DA_conv_generators), vector_of_inverse_alpha_g_DA))) * np.ones((self.num_of_nodes, self.num_of_RPPs)).reshape( self.num_of_nodes, self.num_of_RPPs)
            F__DA = F__DA.reshape(self.num_of_nodes , self.num_of_RPPs)
            #
            F__DA_prime = (1 / ( np.matmul(np.ones((1, self.num_of_DA_conv_generators)).reshape(1, self.num_of_DA_conv_generators), vector_of_inverse_alpha_g_DA))) * (np.matmul(np.ones((self.num_of_nodes, self.num_of_nodes)), np.array(self.vector_of_nodal_demands).reshape( self.num_of_nodes, 1)) + np.matmul( np.ones((self.num_of_nodes, self.num_of_DA_conv_generators)), vector_of_beta_over_alpha_g_DA))
            F__DA_prime = F__DA_prime.reshape(self.num_of_nodes, 1)
            #
            K__DA = np.matmul(diagonal_matrix_of_inverse_alpha_g_DA, F__DA[self.list_of_indices_of_nodes_with_DA_conv_generator, :])
            K__DA = K__DA.reshape(self.num_of_DA_conv_generators , self.num_of_RPPs)
            #
            K__DA_prime = np.matmul(diagonal_matrix_of_inverse_alpha_g_DA, F__DA_prime[self.list_of_indices_of_nodes_with_DA_conv_generator, :]) - vector_of_beta_over_alpha_g_DA
            K__DA_prime = K__DA_prime.reshape(self.num_of_DA_conv_generators, 1)
            #
            #
            #
            ## Calculation of E_g, E_c, and E_d
            E_g__DA = np.zeros((self.num_of_nodes, self.num_of_DA_conv_generators))
            for i in range(self.num_of_DA_conv_generators):
                index_of_connecting_node_to_DA_generator = int(self.data_of_DA_conventional_generators[i, 0])
                E_g__DA[index_of_connecting_node_to_DA_generator, i] = 1
            #
            #
            #
            E_c__DA = np.zeros((self.num_of_nodes, self.num_of_RPPs))
            for i in range(self.num_of_RPPs):
                index_of_connecting_node_to_RPP = int(self.data_of_RPPs[i, 0])
                E_c__DA[index_of_connecting_node_to_RPP, i] = 1
            #
            #
            # because for all nodes we put a load there.
            E_d__DA = np.eye(self.num_of_nodes, self.num_of_nodes)
            #
            #
            #
            #
            #
            # # # # # # # # # # # # # #      RT Calculations      # # # # # # # # # # #
            #
            #
            # Note that no demand is in the RT market, and only deviations of the RT generators must be provided
            vector_of_RT_nodal_demands = np.zeros((self.num_of_nodes, 1)).transpose()
            Total_RT_pure_loads__no_RPP_deviation_considered = 0
            #
            RT_lines_congested_or_not = [0] * self.num_of_branches
            for i in range(self.num_of_branches):
                if current_RT_congestion_pattern[i] == 1:
                    RT_lines_congested_or_not[i] = 1
                elif current_RT_congestion_pattern[i] == 2:
                    RT_lines_congested_or_not[i] = -1
                else:
                    pass
                #
            #
            #
            #
            #
            ## Calculation of E_g, E_c, and E_d
            E_g__RT = np.zeros((self.num_of_nodes, self.num_of_RT_conv_generators))
            for i in range(self.num_of_RT_conv_generators):
                index_of_connecting_node_to_RT_generator = int(self.data_of_RT_conventional_generators[i, 0])
                E_g__RT[index_of_connecting_node_to_RT_generator, i] = 1
            #
            #
            #
            E_c__RT = np.zeros((self.num_of_nodes, self.num_of_RPPs))
            for i in range(self.num_of_RPPs):
                index_of_connecting_node_to_RPP = int(self.data_of_RPPs[i, 0])
                E_c__RT[index_of_connecting_node_to_RPP, i] = 1
            #
            #
            # because for all nodes we put a load there.
            E_d__RT = np.eye(self.num_of_nodes, self.num_of_nodes)
            #
            #
            #
            #
            #
            PTDF_matrix = self.calc_PTDF_matrix()
            PTDF_matrix = PTDF_matrix.transpose()
            #
            if sum([abs(ttt_temp) for ttt_temp in RT_lines_congested_or_not]) == 0:  # no line is congested. Note that this case should not happen because there is another part of the code related to this
                indices_of_RT_congested_lines = []
                num_of_RT_congested_lines = 0
            else:
                indices_of_RT_congested_lines = np.nonzero(np.array(RT_lines_congested_or_not))[0]  # is a list
                num_of_RT_congested_lines = len(indices_of_RT_congested_lines)
            #
            #
            A__RT = PTDF_matrix[indices_of_RT_congested_lines, :]
            A__RT = np.array(A__RT)
            #
            #
            #
            ## Calc A_g, A_c and A_d and T
            A_g__RT = np.matmul(A__RT, E_g__RT)
            A_c__RT = np.matmul(A__RT, E_c__RT)
            A_d__RT = np.matmul(A__RT, E_d__RT)
            #
            A_gDA__at__RT = np.matmul(A__RT, E_g__DA)
            A_dDA__at__RT = np.matmul(A__RT, E_d__DA)
            #
            T__RT = np.multiply(
                np.array([self.branch_capacities[ttt_temp] for ttt_temp in indices_of_RT_congested_lines]),
                np.array([RT_lines_congested_or_not[ttt_temp] for ttt_temp in indices_of_RT_congested_lines]))
            #
            #
            ## Calc H__RT and b__RT
            # H__RT = np.zeros((self.num_of_RT_conv_generators, self.num_of_RT_conv_generators))
            # b__RT = np.zeros((self.num_of_RT_conv_generators, 1))
            # #
            # for i in range(self.num_of_RT_conv_generators):
            #     H__RT[i, i] = H__RT[i, i] + 2 * self.data_of_RT_conventional_generators[i, 1]
            #     b__RT[i, 0] = b__RT[i, 0] + self.data_of_RT_conventional_generators[i, 2]
            #
            H__RT = np.zeros( (self.num_of_nodes , self.num_of_nodes) )
            b__RT = np.zeros( ( self.num_of_nodes , 1 ) )
            #
            for i in range(self.num_of_nodes):
                #
                sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i = 0
                sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i = 0
                #
                for j in range(self.num_of_RT_conv_generators):
                    if int(self.data_of_RT_conventional_generators[j, 0]) == int(i):
                        sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i = sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i + 1 / self.data_of_RT_conventional_generators[j, 1]
                        sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i = sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i + self.data_of_RT_conventional_generators[j, 2] / self.data_of_RT_conventional_generators[j, 1]
                    else:
                        pass
                #
                if sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i != 0:
                    H__RT[i, i] = 1 / sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i
                    b__RT[i, 0] = sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i / sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i
                else:
                    pass
                #
            #
            H__RT = np.diag(H__RT[self.list_of_indices_of_nodes_with_RT_conv_generator, self.list_of_indices_of_nodes_with_RT_conv_generator])
            b__RT = b__RT[self.list_of_indices_of_nodes_with_RT_conv_generator, 0]
            #
            #
            #
            ## froming matrix Z__RT and W__RT
            #
            Z__RT___upper_block = np.concatenate((H__RT, A_g__RT[:, :self.num_of_RT_conv_generators].transpose(),
                                                  np.ones((self.num_of_RT_conv_generators, 1))), axis=1)
            Z__RT___middle_block = np.concatenate((A_g__RT,
                                                   np.zeros((num_of_RT_congested_lines, num_of_RT_congested_lines)),
                                                   np.zeros((num_of_RT_congested_lines, 1))), axis=1)
            Z__RT___lower_block = np.concatenate((np.ones((1, self.num_of_RT_conv_generators)),
                                                  np.zeros((1, num_of_RT_congested_lines)),
                                                  np.array([0]).reshape(1, 1)), axis=1)
            #
            Z__RT = np.concatenate((Z__RT___upper_block, Z__RT___middle_block, Z__RT___lower_block), axis=0)
            #
            #
            #
            # W__RT = Z__RT^(-1);
            W__RT = np.linalg.pinv(Z__RT)
            #
            #
            #
            # K__RT = -1 *  np.matmul(  np.matmul( W__RT[:self.num_of_RT_conv_generators , self.num_of_RT_conv_generators : (self.num_of_RT_conv_generators + num_of_RT_congested_lines)]  ,  A_g__DA )  ,  K__DA )     +    np.matmul( W__RT[:self.num_of_RT_conv_generators , (self.num_of_RT_conv_generators + num_of_RT_congested_lines)].reshape(self.num_of_RT_conv_generators,1)  ,   np.ones((self.num_of_RPPs , 1)).transpose().reshape(1,self.num_of_RPPs)  )
            K__RT = -1 * np.matmul(np.matmul(W__RT[:self.num_of_RT_conv_generators, self.num_of_RT_conv_generators: (self.num_of_RT_conv_generators + num_of_RT_congested_lines)], A_gDA__at__RT),K__DA) + np.matmul(W__RT[:self.num_of_RT_conv_generators, (self.num_of_RT_conv_generators + num_of_RT_congested_lines)].reshape(self.num_of_RT_conv_generators, 1), np.ones((self.num_of_RPPs, 1)).transpose().reshape(1,self.num_of_RPPs))
            #
            #
            K__RT_prime___part_1 = -1 * np.matmul(W__RT[0:self.num_of_RT_conv_generators, self.num_of_RT_conv_generators:(self.num_of_RT_conv_generators + num_of_RT_congested_lines)], A_c__RT)
            K__RT_prime___part_1 = K__RT_prime___part_1.reshape(self.num_of_RT_conv_generators , self.num_of_RPPs)
            K__RT_prime___part_2 = -1 * np.matmul(W__RT[0:self.num_of_RT_conv_generators,(self.num_of_RT_conv_generators + num_of_RT_congested_lines)].reshape(self.num_of_RT_conv_generators, 1), np.ones((self.num_of_RPPs, 1)).transpose().reshape(1,self.num_of_RPPs))
            K__RT_prime___part_2 = K__RT_prime___part_2.reshape(self.num_of_RT_conv_generators, self.num_of_RPPs)
            K__RT_prime = K__RT_prime___part_1 + K__RT_prime___part_2
            #
            #
            #
            K__RT_double_prime___part_1 = -1 * np.matmul(W__RT[:self.num_of_RT_conv_generators, :self.num_of_RT_conv_generators], b__RT)
            K__RT_double_prime___part_1 = K__RT_double_prime___part_1.reshape(self.num_of_RT_conv_generators , 1)
            K__RT_double_prime___part_2___part_1 = W__RT[:self.num_of_RT_conv_generators,self.num_of_RT_conv_generators:(self.num_of_RT_conv_generators + num_of_RT_congested_lines)]
            K__RT_double_prime___part_2___part_1 = K__RT_double_prime___part_2___part_1.reshape(self.num_of_RT_conv_generators , num_of_RT_congested_lines)
            K__RT_double_prime___part_2___part_2 = np.matmul(A_dDA__at__RT,np.array(vector_of_DA_nodal_demands).reshape(self.num_of_nodes, 1)) - np.matmul(A_gDA__at__RT,K__DA_prime) + T__RT.reshape(num_of_RT_congested_lines, 1) + np.matmul(A_d__RT,vector_of_RT_nodal_demands.reshape(self.num_of_nodes, 1))
            K__RT_double_prime___part_2___part_2 = K__RT_double_prime___part_2___part_2.reshape(num_of_RT_congested_lines , 1)
            K__RT_double_prime___part_2 = np.matmul(K__RT_double_prime___part_2___part_1,K__RT_double_prime___part_2___part_2)
            K__RT_double_prime___part_2 = K__RT_double_prime___part_2.reshape(self.num_of_RT_conv_generators, 1)
            K__RT_double_prime___part_3 = W__RT[0:self.num_of_RT_conv_generators,(self.num_of_RT_conv_generators + num_of_RT_congested_lines)].reshape(self.num_of_RT_conv_generators, 1) * Total_RT_pure_loads__no_RPP_deviation_considered
            K__RT_double_prime___part_3 = K__RT_double_prime___part_3.reshape(self.num_of_RT_conv_generators, 1)
            #
            K__RT_double_prime = K__RT_double_prime___part_1 + K__RT_double_prime___part_2 + K__RT_double_prime___part_3
            #
            #
            ## Calculation of  M__RT  and  M__RT_prime
            #
            10 + 5
            #
            M__RT___part_1 = np.matmul(np.matmul(A_d__RT.transpose(), W__RT[:self.num_of_RT_conv_generators,self.num_of_RT_conv_generators:(self.num_of_RT_conv_generators + num_of_RT_congested_lines)].transpose()),H__RT.transpose())
            M__RT___part_1 = M__RT___part_1.reshape(self.num_of_nodes , self.num_of_RT_conv_generators)
            M__RT___part_2 = np.matmul(np.matmul(np.ones((self.num_of_nodes, 1)).reshape(self.num_of_nodes, 1),W__RT[:self.num_of_RT_conv_generators, (self.num_of_RT_conv_generators + num_of_RT_congested_lines)].transpose().reshape(1, self.num_of_RT_conv_generators)), H__RT.transpose())
            M__RT___part_2 = M__RT___part_2.reshape(self.num_of_nodes, self.num_of_RT_conv_generators)
            M__RT = M__RT___part_1 + M__RT___part_2
            #
            #
            #
            M__RT_prime___part_1 = np.matmul(np.matmul(A_d__RT.transpose(), W__RT[:self.num_of_RT_conv_generators, self.num_of_RT_conv_generators:( self.num_of_RT_conv_generators + num_of_RT_congested_lines)].transpose()), b__RT)
            M__RT_prime___part_1 = M__RT_prime___part_1.reshape(self.num_of_nodes , 1)
            M__RT_prime___part_2 = np.matmul(np.matmul(np.ones((self.num_of_nodes, 1)).reshape(self.num_of_nodes, 1), W__RT[:self.num_of_RT_conv_generators, ( self.num_of_RT_conv_generators + num_of_RT_congested_lines)].transpose().reshape(1, self.num_of_RT_conv_generators)), b__RT)
            M__RT_prime___part_2 = M__RT_prime___part_2.reshape(self.num_of_nodes, 1)
            M__RT_prime = M__RT_prime___part_1 + M__RT_prime___part_2
            #
            #
            #
            ## Calculation of  F__DA  and  F__DA_prime , F__RT , F__RT_prime , F__RT_double_prime
            #
            #
            #
            F__RT = np.matmul(M__RT, K__RT)
            F__RT_prime = np.matmul(M__RT, K__RT_prime)
            F__RT_double_prime = np.matmul(M__RT, K__RT_double_prime) + M__RT_prime
            #
            #
            #
            #
            # # # # # # # # # # # #      final Calculations       # # # # # # # # # # #
            #
            #
            #
            Zeta_DA = np.zeros((self.num_of_RPPs, self.num_of_nodes))
            for i in range(self.num_of_RPPs):
                for j in range(self.num_of_nodes):
                    if j == self.data_of_RPPs[i, 0]:
                        Zeta_DA[i, j] = 1
                        break
                    else:
                        pass
                    #
                #
            #
            #
            Zeta_RT = Zeta_DA
            #
            #
            temp_vector_DA = [0] * self.num_of_RPPs
            for i in range(self.num_of_RPPs):
                index_of_node_of_current_RPP_i = int(self.data_of_RPPs[i, 0])
                temp_vector_DA[i] = F__DA[index_of_node_of_current_RPP_i, i]
            #
            #
            Zeta_DA_prime = np.diag(temp_vector_DA)
            #
            #
            temp_vector_RT = [0] * self.num_of_RPPs
            for i in range(self.num_of_RPPs):
                index_of_node_of_current_RPP_i = int(self.data_of_RPPs[i, 0])
                temp_vector_RT[i] = F__RT[index_of_node_of_current_RPP_i, i]
            #
            #
            Zeta_RT_prime = np.diag(temp_vector_RT)
            #
            #
            #
            #
            10 + 5
            ##
            # Delta_left = np.matmul(Zeta_DA, F__DA) + Zeta_DA_prime - Zeta_RT_prime - np.matmul(Zeta_RT, F__RT_prime)
            Delta_left = np.matmul(Zeta_DA, F__DA) + Zeta_DA_prime - Zeta_RT_prime - np.matmul(Zeta_RT, F__RT)
            #
            Delta_right = -1 * np.matmul(Zeta_DA, F__DA_prime) + np.matmul(Zeta_RT, F__RT_double_prime) + np.matmul(
                (np.matmul(Zeta_RT, F__RT_prime) - Zeta_RT_prime), self.data_of_RPPs[:, 1].reshape(self.num_of_RPPs, 1))
            #
            #
            # DA_commitments_of_RPPs_at_NE_for_current_cong_pattern = (Delta_left^(-1))*Delta_right
            DA_commitments_of_RPPs_at_NE_for_current_cong_pattern = np.matmul(np.linalg.pinv(Delta_left), Delta_right).reshape(1, self.num_of_RPPs)
            #
            #
            generation_of_DA_generators = np.matmul(K__DA, np.array(DA_commitments_of_RPPs_at_NE_for_current_cong_pattern).reshape(self.num_of_RPPs, 1)) + K__DA_prime
            #
            expected_generation_of_RT_generator = np.matmul(K__RT, np.array(DA_commitments_of_RPPs_at_NE_for_current_cong_pattern).reshape(self.num_of_RPPs, 1)) + np.matmul(K__RT_prime,np.array(self.data_of_RPPs[:, 1]).reshape(self.num_of_RPPs, 1)) + K__RT_double_prime
            #
            # return DA_commitments_of_RPPs_at_NE_for_current_cong_pattern , F__DA , F__DA_prime , F__RT , F__RT_prime
            return DA_commitments_of_RPPs_at_NE_for_current_cong_pattern, generation_of_DA_generators, expected_generation_of_RT_generator, None, None, None, None
            #
            #
            #
        elif (sum([int(ttt_temp) for ttt_temp in (current_DA_congestion_pattern == [0]*self.num_of_branches)]) != self.num_of_branches) and (sum([int(ttt_temp) for ttt_temp in (current_RT_congestion_pattern == [0]*self.num_of_branches)]) == self.num_of_branches):
            # means that we have some congestion in DA market but no congestion in RT market
            #
            DA_lines_congested_or_not = [0] * self.num_of_branches
            for i in range(self.num_of_branches):
                if current_DA_congestion_pattern[i] == 1:
                    DA_lines_congested_or_not[i] = 1
                elif current_DA_congestion_pattern[i] == 2:
                    DA_lines_congested_or_not[i] = -1
                else:
                    pass
            #
            #
            #
            #
            ## Calculation of E_g, E_c, and E_d
            E_g__DA = np.zeros((self.num_of_nodes, self.num_of_DA_conv_generators))
            for i in range(self.num_of_DA_conv_generators):
                index_of_connecting_node_to_DA_generator = int(self.data_of_DA_conventional_generators[i, 0])
                E_g__DA[index_of_connecting_node_to_DA_generator, i] = 1
            #
            #
            #
            E_c__DA = np.zeros((self.num_of_nodes, self.num_of_RPPs))
            for i in range(self.num_of_RPPs):
                index_of_connecting_node_to_RPP = int(self.data_of_RPPs[i, 0])
                E_c__DA[index_of_connecting_node_to_RPP, i] = 1
            #
            #
            # because for all nodes we put a load there.
            E_d__DA = np.eye(self.num_of_nodes, self.num_of_nodes)
            #
            #
            #
            #
            #
            ##
            ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###
            PTDF_matrix = self.calc_PTDF_matrix()
            PTDF_matrix = PTDF_matrix.transpose()
            #
            if sum([abs(ttt_temp) for ttt_temp in
                    DA_lines_congested_or_not]) == 0:  # no line is congested. Note that this case should not happen because there is another part of the code related to this
                indices_of_DA_congested_lines = []
                num_of_DA_congested_lines = 0
            else:
                indices_of_DA_congested_lines = np.nonzero(np.array(DA_lines_congested_or_not))[0]  # is a list
                num_of_DA_congested_lines = len(indices_of_DA_congested_lines)
            #
            #
            A__DA = PTDF_matrix[indices_of_DA_congested_lines, :]
            A__DA = np.array(A__DA)
            #
            #
            #
            ## Calc A_g__DA, A_c__DA and A_d and T__DA
            A_g__DA = np.matmul(A__DA, E_g__DA)
            A_c__DA = np.matmul(A__DA, E_c__DA)
            A_d__DA = np.matmul(A__DA, E_d__DA)
            #
            #
            T__DA = np.multiply(np.array(self.branch_capacities)[indices_of_DA_congested_lines],
                                np.array(DA_lines_congested_or_not)[indices_of_DA_congested_lines].transpose())
            #
            #
            #
            ## Calc H__DA and b__DA
            # H__DA = np.zeros((self.num_of_DA_conv_generators, self.num_of_DA_conv_generators))
            # b__DA = np.zeros((self.num_of_DA_conv_generators, 1))
            # #
            # for i in range(self.num_of_DA_conv_generators):
            #     H__DA[i, i] = H__DA[i, i] + 2 * self.data_of_DA_conventional_generators[i, 1]
            #     b__DA[i, 0] = b__DA[i, 0] + self.data_of_DA_conventional_generators[i, 2]
            #
            H__DA = np.zeros((self.num_of_nodes, self.num_of_nodes))
            b__DA = np.zeros((self.num_of_nodes, 1))
            #
            for i in range(self.num_of_nodes):
                #
                sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i = 0
                sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i = 0
                #
                for j in range(self.num_of_DA_conv_generators):
                    if int(self.data_of_DA_conventional_generators[j, 0]) == int(i):
                        sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i = sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i + 1 / self.data_of_DA_conventional_generators[j, 1]
                        sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i = sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i + self.data_of_DA_conventional_generators[j, 2] / self.data_of_DA_conventional_generators[j, 1]
                    else:
                        pass
                #
                if sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i != 0:
                    H__DA[i, i] = 1 / sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i
                    b__DA[i, 0] = sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i / sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i
                else:
                    pass
                #
            #
            H__DA = np.diag(H__DA[self.list_of_indices_of_nodes_with_DA_conv_generator , self.list_of_indices_of_nodes_with_DA_conv_generator])
            b__DA = b__DA[self.list_of_indices_of_nodes_with_DA_conv_generator , 0]
            #
            #
            ## froming matrix Z__DA and W__DA
            Z__DA___upper_block = np.concatenate((H__DA, A_g__DA[:, 0:self.num_of_DA_conv_generators].transpose(),
                                                  np.ones((self.num_of_DA_conv_generators, 1))), axis=1)
            Z__DA___middle_block = np.concatenate((A_g__DA,
                                                   np.zeros((num_of_DA_congested_lines, num_of_DA_congested_lines)),
                                                   np.zeros((num_of_DA_congested_lines, 1))), axis=1)
            Z__DA___lower_block = np.concatenate((np.ones((1, self.num_of_DA_conv_generators)),
                                                  np.zeros((1, num_of_DA_congested_lines)),
                                                  np.array([0]).reshape(1, 1)), axis=1)
            #
            Z__DA = np.concatenate((Z__DA___upper_block, Z__DA___middle_block, Z__DA___lower_block), axis=0)
            #
            #
            #
            # W__DA = Z__DA^(-1);
            W__DA = np.linalg.pinv(Z__DA)
            #
            #
            #
            K__DA = -1 * np.matmul(W__DA[0:self.num_of_DA_conv_generators, (self.num_of_DA_conv_generators): ( self.num_of_DA_conv_generators + num_of_DA_congested_lines)], A_c__DA) - np.matmul( W__DA[0: self.num_of_DA_conv_generators, (self.num_of_DA_conv_generators + num_of_DA_congested_lines)].reshape(self.num_of_DA_conv_generators, 1), np.ones((self.num_of_RPPs, 1)).transpose())
            #
            #
            #
            #
            K__DA_prime___part_1 = -1 * np.matmul( W__DA[0:self.num_of_DA_conv_generators, 0:self.num_of_DA_conv_generators], b__DA)
            K__DA_prime___part_1 = K__DA_prime___part_1.reshape(self.num_of_DA_conv_generators, 1)
            K__DA_prime___part_2 = np.matmul(W__DA[0: self.num_of_DA_conv_generators, self.num_of_DA_conv_generators: ( self.num_of_DA_conv_generators + num_of_DA_congested_lines)], ( np.matmul(A_d__DA, vector_of_DA_nodal_demands).reshape( num_of_DA_congested_lines, 1) + T__DA.reshape( num_of_DA_congested_lines, 1)))
            K__DA_prime___part_2 = K__DA_prime___part_2.reshape(self.num_of_DA_conv_generators, 1)
            K__DA_prime___part_3 = W__DA[0: self.num_of_DA_conv_generators, (self.num_of_DA_conv_generators + num_of_DA_congested_lines)].reshape( self.num_of_DA_conv_generators, 1) * Total_DA_loads
            K__DA_prime___part_3 = K__DA_prime___part_3.reshape(self.num_of_DA_conv_generators, 1)
            #
            K__DA_prime = K__DA_prime___part_1 + K__DA_prime___part_2 + K__DA_prime___part_3
            #
            ## Calculation of  M__DA  and  M__DA_prime
            #
            #
            #
            M__DA = np.matmul(np.matmul(A_d__DA.transpose(), W__DA[0:self.num_of_DA_conv_generators, self.num_of_DA_conv_generators: ( self.num_of_DA_conv_generators + num_of_DA_congested_lines)].transpose()), H__DA.transpose()) + np.matmul(np.matmul(np.ones((self.num_of_nodes, 1)), (
            W__DA[0: self.num_of_DA_conv_generators, (self.num_of_DA_conv_generators + num_of_DA_congested_lines)]).transpose().reshape(1, self.num_of_DA_conv_generators)), H__DA.transpose())
            #
            #
            M__DA_prime = np.matmul(np.matmul(A_d__DA.transpose(), W__DA[0:self.num_of_DA_conv_generators, self.num_of_DA_conv_generators: ( self.num_of_DA_conv_generators + num_of_DA_congested_lines)].transpose()), b__DA) + np.matmul( np.matmul(np.ones((self.num_of_nodes, 1)).reshape(self.num_of_nodes, 1), W__DA[0:self.num_of_DA_conv_generators, (self.num_of_DA_conv_generators + num_of_DA_congested_lines)].transpose().reshape(1, self.num_of_DA_conv_generators)), b__DA)
            M__DA_prime = M__DA_prime.reshape(self.num_of_nodes , 1)
            #
            #
            ## Calculation of  F__DA  and  F__DA_prime
            #
            F__DA = np.matmul(M__DA, K__DA)
            F__DA_prime = np.matmul(M__DA, K__DA_prime) + M__DA_prime
            #
            #
            #
            # # # # # # # # # # # # # #      RT Calculations      # # # # # # # # # # #
            #
            #
            #
            vector_of_inverse_alpha_g_RT = np.zeros((self.num_of_RT_conv_generators, 1)).reshape(
                self.num_of_RT_conv_generators, 1)
            for i in range(self.num_of_RT_conv_generators):
                vector_of_inverse_alpha_g_RT[i, 0] = 1 / (self.data_of_RT_conventional_generators[i, 1])
            #
            vector_of_beta_over_alpha_g_RT = np.zeros((self.num_of_RT_conv_generators, 1)).reshape(
                self.num_of_RT_conv_generators, 1)
            for i in range(self.num_of_RT_conv_generators):
                vector_of_beta_over_alpha_g_RT[i, 0] = (self.data_of_RT_conventional_generators[i, 2]) / ( self.data_of_RT_conventional_generators[i, 1])
            #
            diagonal_matrix_of_inverse_alpha_g_RT = np.zeros(
                (self.num_of_RT_conv_generators, self.num_of_RT_conv_generators)).reshape(
                self.num_of_RT_conv_generators, self.num_of_RT_conv_generators)
            for i in range(self.num_of_RT_conv_generators):
                diagonal_matrix_of_inverse_alpha_g_RT[i, i] = -1 / (self.data_of_RT_conventional_generators[i, 1])
            #
            #
            F__RT = ((-1) / ( np.matmul(np.ones((1, self.num_of_RT_conv_generators)).reshape(1, self.num_of_RT_conv_generators), vector_of_inverse_alpha_g_RT))) * np.matmul( np.ones((self.num_of_nodes, self.num_of_DA_conv_generators)).reshape(self.num_of_nodes, self.num_of_DA_conv_generators), K__DA)
            #
            F__RT_prime = ((-1) / ( np.matmul(np.ones((1, self.num_of_RT_conv_generators)).reshape(1, self.num_of_RT_conv_generators), vector_of_inverse_alpha_g_RT))) * np.ones((self.num_of_nodes, self.num_of_RPPs)).reshape( self.num_of_nodes, self.num_of_RPPs)
            #
            F__RT_double_prime__part_1 = (1 / ( np.matmul(np.ones((1, self.num_of_RT_conv_generators)).reshape(1, self.num_of_RT_conv_generators), vector_of_inverse_alpha_g_RT))) * np.matmul( np.ones((self.num_of_nodes, self.num_of_nodes)).reshape(self.num_of_nodes, self.num_of_nodes), np.array(self.vector_of_nodal_demands).reshape(self.num_of_nodes, 1))
            #
            F__RT_double_prime__part_2 = (1 / ( np.matmul(np.ones((1, self.num_of_RT_conv_generators)).reshape(1, self.num_of_RT_conv_generators), vector_of_inverse_alpha_g_RT))) * np.matmul( np.ones((self.num_of_nodes, self.num_of_DA_conv_generators)).reshape(self.num_of_nodes, self.num_of_DA_conv_generators), K__DA_prime)
            #
            F__RT_double_prime__part_3 = (1 / ( np.matmul(np.ones((1, self.num_of_RT_conv_generators)).reshape(1, self.num_of_RT_conv_generators), vector_of_inverse_alpha_g_RT))) * np.matmul( np.ones((self.num_of_nodes, self.num_of_RT_conv_generators)).reshape(self.num_of_nodes, self.num_of_RT_conv_generators), vector_of_beta_over_alpha_g_RT)
            #
            F__RT_double_prime = F__RT_double_prime__part_1 - F__RT_double_prime__part_2 + F__RT_double_prime__part_3
            #
            #
            #
            K__RT              = np.matmul(diagonal_matrix_of_inverse_alpha_g_RT,F__RT[self.list_of_indices_of_nodes_with_RT_conv_generator, :])
            K__RT              = K__RT.reshape(self.num_of_RT_conv_generators, self.num_of_RPPs)
            K__RT_prime        = np.matmul(diagonal_matrix_of_inverse_alpha_g_RT,F__RT_prime[self.list_of_indices_of_nodes_with_RT_conv_generator, :])
            K__RT_prime        = K__RT_prime.reshape(self.num_of_RT_conv_generators , self.num_of_RPPs)
            K__RT_double_prime = np.matmul(diagonal_matrix_of_inverse_alpha_g_RT,F__RT_double_prime[self.list_of_indices_of_nodes_with_RT_conv_generator, :]) - vector_of_beta_over_alpha_g_RT
            K__RT_double_prime = K__RT_double_prime.reshape(self.num_of_RT_conv_generators , 1)
            #
            #
            # # # # # # # # # # # #      final Calculations       # # # # # # # # # # #
            #
            #
            #
            #
            #
            Zeta_DA = np.zeros((self.num_of_RPPs, self.num_of_nodes))
            for i in range(self.num_of_RPPs):
                for j in range(self.num_of_nodes):
                    if j == self.data_of_RPPs[i, 0]:
                        Zeta_DA[i, j] = 1
                        break
                    else:
                        pass
                    #
                #
            #
            #
            Zeta_RT = Zeta_DA
            #
            #
            temp_vector_DA = [0] * self.num_of_RPPs
            for i in range(self.num_of_RPPs):
                index_of_node_of_current_RPP_i = int(self.data_of_RPPs[i, 0])
                temp_vector_DA[i] = F__DA[index_of_node_of_current_RPP_i, i]
            #
            #
            Zeta_DA_prime = np.diag(temp_vector_DA)
            #
            #
            temp_vector_RT = [0] * self.num_of_RPPs
            for i in range(self.num_of_RPPs):
                index_of_node_of_current_RPP_i = int(self.data_of_RPPs[i, 0])
                temp_vector_RT[i] = F__RT[index_of_node_of_current_RPP_i, i]
            #
            #
            Zeta_RT_prime = np.diag(temp_vector_RT)
            #
            #
            #
            #
            10 + 5
            ##
            # Delta_left = np.matmul(Zeta_DA, F__DA) + Zeta_DA_prime - Zeta_RT_prime - np.matmul(Zeta_RT, F__RT_prime)
            Delta_left = np.matmul(Zeta_DA, F__DA) + Zeta_DA_prime - Zeta_RT_prime - np.matmul(Zeta_RT, F__RT)
            #
            Delta_right = -1 * np.matmul(Zeta_DA, F__DA_prime) + np.matmul(Zeta_RT, F__RT_double_prime) + np.matmul( (np.matmul(Zeta_RT, F__RT_prime) - Zeta_RT_prime), self.data_of_RPPs[:, 1].reshape(self.num_of_RPPs, 1))
            #
            #
            # DA_commitments_of_RPPs_at_NE_for_current_cong_pattern = (Delta_left^(-1))*Delta_right
            DA_commitments_of_RPPs_at_NE_for_current_cong_pattern = np.matmul(np.linalg.pinv(Delta_left), Delta_right).reshape(1, self.num_of_RPPs)
            #
            #
            generation_of_DA_generators = np.matmul(K__DA, np.array(DA_commitments_of_RPPs_at_NE_for_current_cong_pattern).reshape(self.num_of_RPPs, 1)) + K__DA_prime
            #
            expected_generation_of_RT_generator = np.matmul(K__RT, np.array(DA_commitments_of_RPPs_at_NE_for_current_cong_pattern).reshape(self.num_of_RPPs, 1)) + np.matmul(K__RT_prime,np.array(self.data_of_RPPs[:, 1]).reshape(self.num_of_RPPs, 1)) + K__RT_double_prime
            #
            # return DA_commitments_of_RPPs_at_NE_for_current_cong_pattern , F__DA , F__DA_prime , F__RT , F__RT_prime
            return DA_commitments_of_RPPs_at_NE_for_current_cong_pattern, generation_of_DA_generators, expected_generation_of_RT_generator, None, None, None, None
            #
            #
        else: # meaning that some lines are congested
            ## ## ## forming the vector "DA_lines_congested_or_not"
            # note that in the vector "current_congestion_pattern", congestion in the opposite direction ...
            # ... is shown with 2, however in the code, the congection in opposite direction is shown with -1;
            #
            DA_lines_congested_or_not = [0] * self.num_of_branches
            for i in range(self.num_of_branches):
                if current_DA_congestion_pattern[i] == 1:
                    DA_lines_congested_or_not[i] = 1
                elif current_DA_congestion_pattern[i] == 2:
                    DA_lines_congested_or_not[i] = -1
                else:
                    pass
            #
            #
            #
            #
            ## Calculation of E_g, E_c, and E_d
            E_g__DA = np.zeros((self.num_of_nodes , self.num_of_DA_conv_generators))
            for i in range(self.num_of_DA_conv_generators):
                index_of_connecting_node_to_DA_generator = int(self.data_of_DA_conventional_generators[i,0])
                E_g__DA[index_of_connecting_node_to_DA_generator , i] = 1
            #
            #
            #
            E_c__DA = np.zeros((self.num_of_nodes , self.num_of_RPPs))
            for i in range(self.num_of_RPPs):
                index_of_connecting_node_to_RPP = int(self.data_of_RPPs[i,0])
                E_c__DA[index_of_connecting_node_to_RPP , i] = 1
            #
            #
            # because for all nodes we put a load there.
            E_d__DA = np.eye(self.num_of_nodes , self.num_of_nodes)
            #
            #
            #
            #
            #
            ##
            ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###
            PTDF_matrix = self.calc_PTDF_matrix()
            PTDF_matrix = PTDF_matrix.transpose()
            #
            if sum([abs(ttt_temp) for ttt_temp in DA_lines_congested_or_not]) == 0:  # no line is congested. Note that this case should not happen because there is another part of the code related to this
                indices_of_DA_congested_lines = []
                num_of_DA_congested_lines = 0
            else:
                indices_of_DA_congested_lines = np.nonzero(np.array(DA_lines_congested_or_not))[0] # is a list
                num_of_DA_congested_lines = len(indices_of_DA_congested_lines)
            #
            #
            A__DA = PTDF_matrix[indices_of_DA_congested_lines,:]
            A__DA = np.array(A__DA)
            #
            #
            #
            ## Calc A_g__DA, A_c__DA and A_d and T__DA
            A_g__DA = np.matmul(A__DA , E_g__DA)
            A_c__DA = np.matmul(A__DA , E_c__DA)
            A_d__DA = np.matmul(A__DA , E_d__DA)
            #
            #
            T__DA = np.multiply(np.array(self.branch_capacities)[indices_of_DA_congested_lines] , np.array(DA_lines_congested_or_not)[indices_of_DA_congested_lines].transpose())
            #
            #
            #
            ## Calc H__DA and b__DA
            # H__DA = np.zeros((self.num_of_DA_conv_generators , self.num_of_DA_conv_generators))
            # b__DA = np.zeros((self.num_of_DA_conv_generators , 1))
            # #
            # for i in range(self.num_of_DA_conv_generators):
            #     H__DA[i,i] = H__DA[i,i] + 2*self.data_of_DA_conventional_generators[i,1]
            #     b__DA[i,0] = b__DA[i,0] + self.data_of_DA_conventional_generators[i,2]
            #
            H__DA = np.zeros((self.num_of_nodes, self.num_of_nodes))
            b__DA = np.zeros((self.num_of_nodes, 1))
            #
            for i in range(self.num_of_nodes):
                #
                sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i = 0
                sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i = 0
                #
                for j in range(self.num_of_DA_conv_generators):
                    if int(self.data_of_DA_conventional_generators[j, 0]) == int(i):
                        sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i = sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i + 1 / self.data_of_DA_conventional_generators[j, 1]
                        sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i = sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i + self.data_of_DA_conventional_generators[j, 2] / self.data_of_DA_conventional_generators[j, 1]
                    else:
                        pass
                #
                if sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i != 0:
                    H__DA[i, i] = 1 / sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i
                    b__DA[i, 0] = sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i / sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i
                else:
                    pass
                #
            #
            H__DA = np.diag(H__DA[self.list_of_indices_of_nodes_with_DA_conv_generator, self.list_of_indices_of_nodes_with_DA_conv_generator])
            b__DA = b__DA[self.list_of_indices_of_nodes_with_DA_conv_generator, 0]
            #
            #
            #
            ## froming matrix Z__DA and W__DA
            Z__DA___upper_block  =  np.concatenate( ( H__DA , A_g__DA[:,0:self.num_of_DA_conv_generators].transpose() , np.ones((self.num_of_DA_conv_generators,1)) ) , axis = 1 )
            Z__DA___middle_block =  np.concatenate( ( A_g__DA , np.zeros((num_of_DA_congested_lines , num_of_DA_congested_lines)) , np.zeros((num_of_DA_congested_lines , 1) ) ) , axis = 1 )
            Z__DA___lower_block  =  np.concatenate( ( np.ones((1 , self.num_of_DA_conv_generators)) , np.zeros((1,num_of_DA_congested_lines)) , np.array([0]).reshape(1,1) ) , axis = 1 )
            #
            Z__DA = np.concatenate( ( Z__DA___upper_block , Z__DA___middle_block , Z__DA___lower_block ) , axis=0 )
            #
            #
            #
            # W__DA = Z__DA^(-1);
            W__DA = np.linalg.pinv(Z__DA)
            #
            #
            #
            K__DA = -1 * np.matmul(  W__DA[0:self.num_of_DA_conv_generators , (self.num_of_DA_conv_generators) : (self.num_of_DA_conv_generators + num_of_DA_congested_lines)]  ,  A_c__DA   ) - np.matmul( W__DA[0: self.num_of_DA_conv_generators, (self.num_of_DA_conv_generators + num_of_DA_congested_lines)].reshape(self.num_of_DA_conv_generators,1)  ,  np.ones((self.num_of_RPPs , 1)).transpose() )
            K__DA = K__DA.reshape(self.num_of_DA_conv_generators , self.num_of_RPPs)
            #
            #
            #
            # K__DA_prime = -1 * W__DA(1:num_of_DA_conv_generators , 1:num_of_DA_conv_generators) * b__DA ...
            #     + W__DA(1:num_of_DA_conv_generators , (num_of_DA_conv_generators + 1) : (num_of_DA_conv_generators + num_of_DA_congested_lines))*(A_d__DA * vector_of_DA_nodal_demands' + T__DA) ...
            #     + W__DA(1:num_of_DA_conv_generators , (num_of_DA_conv_generators + num_of_DA_congested_lines + 1)) * (sum(vector_of_DA_nodal_demands'));
            #
            #
            K__DA_prime___part_1 = -1 * np.matmul( W__DA[0:self.num_of_DA_conv_generators, 0:self.num_of_DA_conv_generators]  ,  b__DA )
            K__DA_prime___part_1 = K__DA_prime___part_1.reshape(self.num_of_DA_conv_generators , 1)
            K__DA_prime___part_2 = np.matmul( W__DA[0: self.num_of_DA_conv_generators, self.num_of_DA_conv_generators: ( self.num_of_DA_conv_generators + num_of_DA_congested_lines)]   ,   (np.matmul( A_d__DA , vector_of_DA_nodal_demands).reshape(num_of_DA_congested_lines , 1) + T__DA.reshape(num_of_DA_congested_lines,1) )  )
            K__DA_prime___part_2 = K__DA_prime___part_2.reshape(self.num_of_DA_conv_generators,1)
            K__DA_prime___part_3 =  W__DA[0: self.num_of_DA_conv_generators, (self.num_of_DA_conv_generators + num_of_DA_congested_lines)].reshape(self.num_of_DA_conv_generators, 1) * Total_DA_loads
            K__DA_prime___part_3 = K__DA_prime___part_3.reshape(self.num_of_DA_conv_generators, 1)
            #
            K__DA_prime = K__DA_prime___part_1 + K__DA_prime___part_2 + K__DA_prime___part_3
            #
            ## Calculation of  M__DA  and  M__DA_prime
            #
            #
            #
            M__DA =  np.matmul( np.matmul( A_d__DA.transpose()  ,   W__DA[0:self.num_of_DA_conv_generators , self.num_of_DA_conv_generators : (self.num_of_DA_conv_generators + num_of_DA_congested_lines)].transpose() ) ,  H__DA.transpose() )    +      np.matmul( np.matmul( np.ones((self.num_of_nodes, 1))  ,  (W__DA[0 : self.num_of_DA_conv_generators, (self.num_of_DA_conv_generators + num_of_DA_congested_lines)]).transpose().reshape(1,self.num_of_DA_conv_generators) ) ,    H__DA.transpose()  )
            M__DA = M__DA.reshape(self.num_of_nodes , self.num_of_DA_conv_generators)
            #
            #
            M__DA_prime =   np.matmul( np.matmul(  A_d__DA.transpose()  ,  W__DA[0:self.num_of_DA_conv_generators , self.num_of_DA_conv_generators : (self.num_of_DA_conv_generators + num_of_DA_congested_lines)].transpose() ) ,  b__DA  )   +  np.matmul( np.matmul( np.ones((self.num_of_nodes , 1)).reshape(self.num_of_nodes , 1)  ,  W__DA[0:self.num_of_DA_conv_generators , (self.num_of_DA_conv_generators + num_of_DA_congested_lines)].transpose().reshape(1,self.num_of_DA_conv_generators) ) ,    b__DA )
            M__DA_prime = M__DA_prime.reshape(self.num_of_nodes , 1)
            #
            #
            ## Calculation of  F__DA  and  F__DA_prime
            #
            F__DA =  np.matmul( M__DA , K__DA )
            F__DA_prime =  np.matmul( M__DA , K__DA_prime )  +   M__DA_prime
            #
            #
            #
            # # # # # # # # # # # # # #      RT Calculations      # # # # # # # # # # #
            #
            #
            #
            ## forming the vector "RT_lines_congested_or_not"
            # note that in the vector "current_congestion_pattern", congestion in the opposite direction ...
            # ... is shown with 2, however in the code, the congection in opposite direction is shown with -1;
            #
            # Note that no demand is in the RT market, and only deviations of the RT generators must be provided
            vector_of_RT_nodal_demands = np.zeros((self.num_of_nodes , 1)).transpose()
            Total_RT_pure_loads__no_RPP_deviation_considered = 0
            #
            RT_lines_congested_or_not = [0]*self.num_of_branches
            for i in range(self.num_of_branches):
                if current_RT_congestion_pattern[i] == 1:
                    RT_lines_congested_or_not[i] = 1
                elif current_RT_congestion_pattern[i] == 2:
                    RT_lines_congested_or_not[i] = -1
                else:
                    pass
                #
            #
            #
            #
            #
            ## Calculation of E_g, E_c, and E_d
            E_g__RT = np.zeros((  self.num_of_nodes , self.num_of_RT_conv_generators ))
            for i in range(self.num_of_RT_conv_generators):
                index_of_connecting_node_to_RT_generator = int(self.data_of_RT_conventional_generators[i,0])
                E_g__RT[index_of_connecting_node_to_RT_generator , i] = 1
            #
            #
            #
            E_c__RT = np.zeros((self.num_of_nodes , self.num_of_RPPs))
            for i in range(self.num_of_RPPs):
                index_of_connecting_node_to_RPP = int(self.data_of_RPPs[i,0])
                E_c__RT[index_of_connecting_node_to_RPP , i] = 1
            #
            #
            # because for all nodes we put a load there.
            E_d__RT = np.eye(self.num_of_nodes , self.num_of_nodes)
            #
            #
            #
            #
            #
            PTDF_matrix = self.calc_PTDF_matrix()
            PTDF_matrix = PTDF_matrix.transpose()
            #
            if sum([abs(ttt_temp) for ttt_temp in RT_lines_congested_or_not]) == 0:  # no line is congested. Note that this case should not happen because there is another part of the code related to this
                indices_of_RT_congested_lines = []
                num_of_RT_congested_lines = 0
            else:
                indices_of_RT_congested_lines = np.nonzero(np.array(RT_lines_congested_or_not))[0]  # is a list
                num_of_RT_congested_lines = len(indices_of_RT_congested_lines)
            #
            #
            A__RT = PTDF_matrix[indices_of_RT_congested_lines, :]
            A__RT = np.array(A__RT)
            #
            #
            #
            ## Calc A_g, A_c and A_d and T
            A_g__RT = np.matmul(A__RT , E_g__RT)
            A_c__RT = np.matmul(A__RT , E_c__RT)
            A_d__RT = np.matmul(A__RT , E_d__RT)
            #
            A_gDA__at__RT = np.matmul(A__RT , E_g__DA)
            A_dDA__at__RT = np.matmul(A__RT , E_d__DA)
            # T__RT = np.multiply( self.branch_capacities[indices_of_RT_congested_lines] , RT_lines_congested_or_not[indices_of_RT_congested_lines] )
            # T__RT = np.multiply(self.branch_capacities[list(indices_of_RT_congested_lines)] , RT_lines_congested_or_not[list(indices_of_RT_congested_lines)])
            # T__RT = np.multiply([self.branch_capacities[ttt_temp] for ttt_temp in indices_of_RT_congested_lines] , [RT_lines_congested_or_not[ttt_temp] for ttt_temp in indices_of_RT_congested_lines])
            T__RT = np.multiply( np.array([self.branch_capacities[ttt_temp] for ttt_temp in indices_of_RT_congested_lines]) , np.array([RT_lines_congested_or_not[ttt_temp] for ttt_temp in indices_of_RT_congested_lines]))
            #
            #
            ## Calc H__RT and b__RT
            # H__RT = np.zeros( (self.num_of_RT_conv_generators , self.num_of_RT_conv_generators) )
            # b__RT = np.zeros( ( self.num_of_RT_conv_generators , 1 ) )
            # #
            # for i in range(self.num_of_RT_conv_generators):
            #     H__RT[i,i] = H__RT[i,i] + 2*self.data_of_RT_conventional_generators[i,1]
            #     b__RT[i,0] = b__RT[i,0] + self.data_of_RT_conventional_generators[i,2]
            # #
            #
            H__RT = np.zeros( (self.num_of_nodes , self.num_of_nodes) )
            b__RT = np.zeros( ( self.num_of_nodes , 1 ) )
            #
            for i in range(self.num_of_nodes):
                #
                sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i = 0
                sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i = 0
                #
                for j in range(self.num_of_RT_conv_generators):
                    if int(self.data_of_RT_conventional_generators[j, 0]) == int(i):
                        sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i = sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i + 1 / self.data_of_RT_conventional_generators[j, 1]
                        sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i = sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i + self.data_of_RT_conventional_generators[j, 2] / self.data_of_RT_conventional_generators[j, 1]
                    else:
                        pass
                #
                if sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i != 0:
                    H__RT[i, i] = 1 / sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i
                    b__RT[i, 0] = sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i / sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i
                else:
                    pass
                #
            #
            H__RT = np.diag(H__RT[self.list_of_indices_of_nodes_with_RT_conv_generator, self.list_of_indices_of_nodes_with_RT_conv_generator])
            b__RT = b__RT[self.list_of_indices_of_nodes_with_RT_conv_generator, 0]
            #
            #
            ## froming matrix Z__RT and W__RT
            #
            Z__RT___upper_block  = np.concatenate( ( H__RT , A_g__RT[: , :self.num_of_RT_conv_generators].transpose()  , np.ones((self.num_of_RT_conv_generators,1)) ) , axis=1 )
            Z__RT___middle_block = np.concatenate(  ( A_g__RT ,  np.zeros((num_of_RT_congested_lines , num_of_RT_congested_lines))  ,  np.zeros((num_of_RT_congested_lines , 1))  ) , axis = 1  )
            Z__RT___lower_block  = np.concatenate(  ( np.ones((1 , self.num_of_RT_conv_generators)) , np.zeros((1,num_of_RT_congested_lines)) ,  np.array([0]).reshape(1,1) )  , axis=1 )
            #
            Z__RT = np.concatenate( ( Z__RT___upper_block , Z__RT___middle_block , Z__RT___lower_block ) , axis = 0)
            #
            #
            #
            # W__RT = Z__RT^(-1);
            W__RT = np.linalg.pinv(Z__RT)
            #
            #
            #
            # K__RT = -1 *  np.matmul(  np.matmul( W__RT[:self.num_of_RT_conv_generators , self.num_of_RT_conv_generators : (self.num_of_RT_conv_generators + num_of_RT_congested_lines)]  ,  A_g__DA )  ,  K__DA )     +    np.matmul( W__RT[:self.num_of_RT_conv_generators , (self.num_of_RT_conv_generators + num_of_RT_congested_lines)].reshape(self.num_of_RT_conv_generators,1)  ,   np.ones((self.num_of_RPPs , 1)).transpose().reshape(1,self.num_of_RPPs)  )
            K__RT = -1 * np.matmul(np.matmul(W__RT[:self.num_of_RT_conv_generators, self.num_of_RT_conv_generators: ( self.num_of_RT_conv_generators + num_of_RT_congested_lines)], A_gDA__at__RT), K__DA) + np.matmul( W__RT[:self.num_of_RT_conv_generators, (self.num_of_RT_conv_generators + num_of_RT_congested_lines)].reshape(self.num_of_RT_conv_generators, 1), np.ones((self.num_of_RPPs, 1)).transpose().reshape(1, self.num_of_RPPs))
            K__RT = K__RT.reshape(self.num_of_RT_conv_generators , self.num_of_RPPs)
            #
            #
            K__RT_prime___part_1 =  -1 * np.matmul(  W__RT[0:self.num_of_RT_conv_generators , self.num_of_RT_conv_generators:(self.num_of_RT_conv_generators + num_of_RT_congested_lines)]  ,  A_c__RT  )
            K__RT_prime___part_1 = K__RT_prime___part_1.reshape(self.num_of_RT_conv_generators , self.num_of_RPPs)
            K__RT_prime___part_2 =  -1 * np.matmul(  W__RT[0:self.num_of_RT_conv_generators , (self.num_of_RT_conv_generators + num_of_RT_congested_lines)].reshape(self.num_of_RT_conv_generators , 1)  ,  np.ones((self.num_of_RPPs , 1)).transpose().reshape(1,self.num_of_RPPs)  )
            K__RT_prime___part_2 = K__RT_prime___part_2.reshape(self.num_of_RT_conv_generators, self.num_of_RPPs)
            K__RT_prime = K__RT_prime___part_1 + K__RT_prime___part_2
            #
            # HHHHHH
            # K__RT_double_prime___part_1 = -1* np.matmul( W__RT[:self.num_of_RT_conv_generators , :self.num_of_RT_conv_generators] , b__RT  )
            # K__RT_double_prime___part_2___part_1 = W__RT[:self.num_of_RT_conv_generators , self.num_of_RT_conv_generators:(self.num_of_RT_conv_generators + num_of_RT_congested_lines)]
            # K__RT_double_prime___part_2___part_2 = np.matmul(A_d__DA , vector_of_DA_nodal_demands.reshape(self.num_of_nodes,1) ) - np.matmul(A_g__DA , K__DA_prime) + T__RT + np.matmul( A_d__RT , vector_of_RT_nodal_demands.reshape(self.num_of_nodes , 1) )
            # K__RT_double_prime___part_2 = np.matmul(K__RT_double_prime___part_2___part_1 , K__RT_double_prime___part_2___part_2)
            # K__RT_double_prime___part_3 = W__RT[0:self.num_of_RT_conv_generators , (self.num_of_RT_conv_generators + num_of_RT_congested_lines)].reshape(self.num_of_RT_conv_generators , 1)  *  sum(vector_of_RT_nodal_demands)
            #
            K__RT_double_prime___part_1 = -1 * np.matmul( W__RT[:self.num_of_RT_conv_generators, :self.num_of_RT_conv_generators], b__RT)
            K__RT_double_prime___part_1 = K__RT_double_prime___part_1.reshape(self.num_of_RT_conv_generators , 1)
            K__RT_double_prime___part_2___part_1 = W__RT[:self.num_of_RT_conv_generators, self.num_of_RT_conv_generators:( self.num_of_RT_conv_generators + num_of_RT_congested_lines)]
            K__RT_double_prime___part_2___part_1 = K__RT_double_prime___part_2___part_1.reshape(self.num_of_RT_conv_generators , num_of_RT_congested_lines)
            K__RT_double_prime___part_2___part_2 = np.matmul(A_dDA__at__RT, np.array(vector_of_DA_nodal_demands).reshape(self.num_of_nodes, 1)) - np.matmul(A_gDA__at__RT, K__DA_prime) + T__RT.reshape(num_of_RT_congested_lines,1) + np.matmul( A_d__RT, vector_of_RT_nodal_demands.reshape(self.num_of_nodes, 1))
            K__RT_double_prime___part_2___part_2 = K__RT_double_prime___part_2___part_2.reshape(num_of_RT_congested_lines, 1)
            K__RT_double_prime___part_2 = np.matmul(K__RT_double_prime___part_2___part_1, K__RT_double_prime___part_2___part_2)
            K__RT_double_prime___part_3 = W__RT[0:self.num_of_RT_conv_generators, (self.num_of_RT_conv_generators + num_of_RT_congested_lines)].reshape( self.num_of_RT_conv_generators, 1) * Total_RT_pure_loads__no_RPP_deviation_considered
            K__RT_double_prime___part_3 = K__RT_double_prime___part_3.reshape(self.num_of_RT_conv_generators, 1)

            #
            K__RT_double_prime = K__RT_double_prime___part_1 + K__RT_double_prime___part_2 + K__RT_double_prime___part_3
            #
            #
            ## Calculation of  M__RT  and  M__RT_prime
            #
            10 + 5
            #
            M__RT___part_1 =  np.matmul( np.matmul(  A_d__RT.transpose() ,  W__RT[:self.num_of_RT_conv_generators , self.num_of_RT_conv_generators:(self.num_of_RT_conv_generators + num_of_RT_congested_lines)].transpose() ) ,  H__RT.transpose()  )
            M__RT___part_1 = M__RT___part_1.reshape(self.num_of_nodes , self.num_of_RT_conv_generators)
            M__RT___part_2 = np.matmul( np.matmul( np.ones((self.num_of_nodes , 1)).reshape(self.num_of_nodes , 1)  ,  W__RT[:self.num_of_RT_conv_generators , (self.num_of_RT_conv_generators + num_of_RT_congested_lines)].transpose().reshape(1,self.num_of_RT_conv_generators) ) , H__RT.transpose() )
            M__RT___part_2 = M__RT___part_2.reshape(self.num_of_nodes, self.num_of_RT_conv_generators)
            M__RT = M__RT___part_1  +  M__RT___part_2
            #
            #
            #
            # M__RT_prime___part_1 = np.matmul( np.matmul(  A_d__RT.transpose()  ,  W__RT[:self.num_of_RT_conv_generators , self.num_of_RT_conv_generators:(self.num_of_RT_conv_generators + num_of_RT_congested_lines) ].transpose() ) ,  H__RT.transpose()  )
            # M__RT_prime___part_2 = np.matmul( np.matmul(  np.ones((self.num_of_nodes , 1)).reshape(self.num_of_nodes , 1)  ,  W__RT[:self.num_of_RT_conv_generators , (self.num_of_RT_conv_generators + num_of_RT_congested_lines) ].transpose().reshape(1,self.num_of_RT_conv_generators) ) ,  H__RT.transpose()  )
            # M__RT_prime = M__RT_prime___part_1  +  M__RT_prime___part_2
            M__RT_prime___part_1 = np.matmul(np.matmul(A_d__RT.transpose(), W__RT[:self.num_of_RT_conv_generators,self.num_of_RT_conv_generators:(self.num_of_RT_conv_generators + num_of_RT_congested_lines)].transpose()) , b__RT)
            M__RT_prime___part_1 = M__RT_prime___part_1.reshape(self.num_of_nodes , 1)
            M__RT_prime___part_2 = np.matmul(np.matmul(np.ones((self.num_of_nodes, 1)).reshape(self.num_of_nodes, 1),W__RT[:self.num_of_RT_conv_generators, (self.num_of_RT_conv_generators + num_of_RT_congested_lines)].transpose().reshape(1, self.num_of_RT_conv_generators)) , b__RT)
            M__RT_prime___part_2 = M__RT_prime___part_2.reshape(self.num_of_nodes, 1)
            M__RT_prime = M__RT_prime___part_1 + M__RT_prime___part_2
            #
            #
            #
            ## Calculation of  F__DA  and  F__DA_prime , F__RT , F__RT_prime , F__RT_double_prime
            #
            F__DA = np.matmul(M__DA, K__DA)
            F__DA_prime = np.matmul(M__DA, K__DA_prime) + M__DA_prime
            #
            F__RT = np.matmul( M__RT , K__RT )
            F__RT_prime = np.matmul( M__RT , K__RT_prime )
            F__RT_double_prime = np.matmul(M__RT, K__RT_double_prime) + M__RT_prime
            #
            #
            #
            #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # #      final Calculations       # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            #
            #
            #
            #
            Zeta_DA = np.zeros((self.num_of_RPPs , self.num_of_nodes))
            for i in range(self.num_of_RPPs):
                for j in range(self.num_of_nodes):
                    if j == self.data_of_RPPs[i,0]:
                        Zeta_DA[i,j] = 1
                        break
                    else:
                        pass
                    #
                #
            #
            #
            Zeta_RT = Zeta_DA
            #
            #
            temp_vector_DA = [0]*self.num_of_RPPs
            for i in range(self.num_of_RPPs):
                index_of_node_of_current_RPP_i = int(self.data_of_RPPs[i,0])
                temp_vector_DA[i] = F__DA[index_of_node_of_current_RPP_i , i]
            #
            #
            Zeta_DA_prime = np.diag(temp_vector_DA)
            #
            #
            temp_vector_RT = [0]*self.num_of_RPPs
            for i in range(self.num_of_RPPs):
                index_of_node_of_current_RPP_i = int(self.data_of_RPPs[i,0])
                temp_vector_RT[i] = F__RT[index_of_node_of_current_RPP_i , i]
            #
            #
            Zeta_RT_prime = np.diag(temp_vector_RT)
            #
            #
            #
            #
            10 + 5
            ##
            # Delta_left = np.matmul( Zeta_DA , F__DA ) + Zeta_DA_prime - Zeta_RT_prime - np.matmul( Zeta_RT , F__RT_prime )
            Delta_left = np.matmul(Zeta_DA, F__DA) + Zeta_DA_prime - Zeta_RT_prime - np.matmul(Zeta_RT, F__RT)
            #
            Delta_right = -1 *  np.matmul( Zeta_DA , F__DA_prime) + np.matmul( Zeta_RT ,  F__RT_double_prime )   +   np.matmul(  (np.matmul( Zeta_RT , F__RT_prime )   -   Zeta_RT_prime)  ,  self.data_of_RPPs[:,1].reshape(self.num_of_RPPs , 1)  )
            #
            #
            # DA_commitments_of_RPPs_at_NE_for_current_cong_pattern = (Delta_left^(-1))*Delta_right
            DA_commitments_of_RPPs_at_NE_for_current_cong_pattern = np.matmul( np.linalg.pinv(Delta_left)  ,  Delta_right ).reshape(1,self.num_of_RPPs)
            #
            #
            generation_of_DA_generators = np.matmul(K__DA, np.array(DA_commitments_of_RPPs_at_NE_for_current_cong_pattern).reshape(self.num_of_RPPs, 1)) + K__DA_prime
            #
            expected_generation_of_RT_generator = np.matmul(K__RT, np.array(DA_commitments_of_RPPs_at_NE_for_current_cong_pattern).reshape(self.num_of_RPPs, 1)) + np.matmul(K__RT_prime,np.array(self.data_of_RPPs[:, 1]).reshape(self.num_of_RPPs, 1)) + K__RT_double_prime
            #
            # return DA_commitments_of_RPPs_at_NE_for_current_cong_pattern , F__DA , F__DA_prime , F__RT , F__RT_prime
            return DA_commitments_of_RPPs_at_NE_for_current_cong_pattern, generation_of_DA_generators, expected_generation_of_RT_generator , None , None , None , None
    #
    #
    #
    def generate_samples_for_the_case_of_multivariate_normal_distributions(self , num_of_samples):
        #
        # returns a ndarray of shape         num_of_samples * num_of_random_variables
        # note that here  num_of_random_variables is equal to the num_of_RPPs
        #
        # the following line makes it always generate the same set of random variables.
        np.random.seed(0)
        #
        mean_of_RPPs = self.data_of_RPPs[:,1]
        Cov_matrix = np.array(self.Cov_matrix_of_RPPs)
        matrix_of_random_samples = np.random.multivariate_normal( mean_of_RPPs , Cov_matrix , num_of_samples )
        return matrix_of_random_samples
    #
    #
    #
    def social_optimum_calculations(self , num_of_scenarios , penalty_factor = 5000):
        # this function is for the calculation of the social welfare calculations and to make the
        #
        generated_scenarios = np.array(self.generate_samples_for_the_case_of_multivariate_normal_distributions(num_of_scenarios)).reshape((num_of_scenarios , self.num_of_RPPs))
#        generated_scenarios = np.array([[70,50]]).reshape(1,2)
        # each row of generated_scenarios corresponds to one scenario
        #
        #
        #
        # this function returns:
        # nodal_injections , LMPs , list_of_lines_congested_or_not , DC_power_flow_has_feasible_solution
        #
        #
        # DC_power_flow_has_feasible_solution = is a variable that takes ...
        # ... a value of True (if feasible) or False (if infeasible). If the rsultant DC power flow of the ...
        # ... combination of current DA commitment of the RPPs, the nodal demands, the line capacities, the ...
        # ... structure of the network, and the location of the DA conventional generators gives us a feasible ...
        # ... solution, then we will have a feasible solution, and current set of DA commitments ...
        # ... of the RPPs MAY be a pure NE. If there is no feasible solution to the DC ...
        # ... power flow, then definitely the current DA commitments are not a pure NE.
        #
        #
        #
        # list_of_lines_congested_or_not  is a 1*num_of_branches numpy vector where if the line #i is ...
        # ... congested in the direction of definition, the the i th element of this vector is 1, and if the ...
        # ... line #i is congested in the reverse direction of its definition,the i th element of this vector is -1, ...
        # ... and if the line #i is not congested, the i th element of this vector is 0.
        #
        #
        # note that this function considers the realizations of the RPPs as negative loads.
        # note that for the social optimum calculations (i.e. this function), no DA commitments are required
        #
        #
        #
        # this function calculates the optimal power generation of the DA generators and RT generators and the  ...
        # ... corresponding total expected cost function!
        #
        # the variables of the DC power flow optimization are:
        # x = [ nodal injections for DA generators ; nodal injections for the RT generators ; total nodal injections ; nodal angles ; labmbda_positive_lines ; lambda_negative_lines]
        #
        # nodal injections for DA generators ==> is the injection for the DA conv generators. for the nodes without DA conv generators, the corresponding elements in this vector is zero
        # (length of these variables isL num_of_nodes)
        #
        # nodal injections for RT generators ==> is the injection for the RT conv generators for each scenario. for the nodes without RT conv generators, the corresponding elements in this vector is zero at each scenario!
        # (note that the size of this vector is: num_of_nodes * num_of_scenarios )
        #
        # total nodal injections ==> this set of variables is the sum of the previous two vectors. This set of variables represent the net injections in each node
        #
        #
        #
        # note that always the nodal angle of the reference node is zero (it is set in the equality constraint too)
        #
        #
        # we are trying to minimize 0.5 * xT * H * x + fT * x
        #
        #
        #
        # note that here we assume that there is one DA conv generators at each ...
        # node and one RT conv generators at each node. then for those nodes that do not have any DA conv generators, we make ...
        # the corresponding elements in the 'nodal injections for DA generators' to zero, and for the nodes without RT conv generators, ...
        # ... we make the corresponding elements in the 'nodal injections for RT generators' set to zero!
        #
        #
        # The variable vector x consists of 2*num_of_nodes variables: the first num_of_nodes ...
        # ... variables correspond to the DA conv injections, and the second num_of_nodes ...
        # ... variables correspond to the nodal angles.
        #
        #
        #
        ## -------------------------------------------------------------------------------------------------------------
        ## Calculating H and f (Quadratic and linear coefficient terms in the objective function of the OPF)
        H = copy.deepcopy(np.zeros(((self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios ), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios)))))
        f = copy.deepcopy(np.zeros(((self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios ), 1))).reshape(( (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios) , 1))
        #
        # explanation of the size of f (and H)
        # first set of variables consists of (num_of_nodes) variables which correspond to the nodal injections corresponding to the DA conv generators
        # second set of variables consists of (self.num_of_nodes * num_of_scenarios) variables which correspond to the nodal injections corresponding to the RT conv generators at all the scenarios
        # third set of variables consists of (self.num_of_nodes * num_of_scenarios) variables which correspond to the net nodal injections corresponding to both DA and RT conv generators at all the scenarios (if we copy the first set of variables for each scenario and then add it to the second set of variables, we get this vector)
        # forth set of variables consists of (self.num_of_branches * num_of_scenarios) variables which is the slack variables corresponding to the + line flow inequlalities at all scenarios.
        # fifth set of variables consists of (self.num_of_branches * num_of_scenarios) variables which is the slack variables corresponding to the - line flow inequlalities at all scenarios.
        # sixth set of variables consists of (self.num_of_nodes * num_of_scenarios) variables which correspinds to the nodal angles at all the scenarios
        #
        #
        #
        #
        vector_of_probabilities = (1 / num_of_scenarios) * np.ones((1, num_of_scenarios))
        #
        ##
        ## forming the quadratic and linear matrix and vector in the objective function
        # for the DA generators
        H_DA_temp = copy.deepcopy(np.zeros((self.num_of_nodes, self.num_of_nodes)))
        f_DA_temp = copy.deepcopy(np.zeros((self.num_of_nodes, 1))).reshape((self.num_of_nodes , 1))
        for i in range(self.num_of_nodes):
            #
            sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i = 0
            sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i = 0
            #
            for j in range(self.num_of_DA_conv_generators):
                if int(self.data_of_DA_conventional_generators[j, 0]) == int(i):
                    sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i = sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i + 1 / self.data_of_DA_conventional_generators[j, 1]
                    sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i = sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i + self.data_of_DA_conventional_generators[j, 2] / self.data_of_DA_conventional_generators[j, 1]
                else:
                    pass
            #
            if sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i != 0:
                H_DA_temp[i, i] = 1 / sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i
                f_DA_temp[i, 0] = sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i / sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i
            else:
                pass
            #
        #
        #
        #
        # for the RT generators
        H_RT_temp = copy.deepcopy(np.zeros((self.num_of_nodes , self.num_of_nodes)))
        f_RT_temp = copy.deepcopy(np.zeros((self.num_of_nodes , 1))).reshape((self.num_of_nodes , 1))
        #
        for i in range(self.num_of_nodes):
            #
            sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i = 0
            sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i = 0
            #
            for j in range(self.num_of_RT_conv_generators):
                if int(self.data_of_RT_conventional_generators[j, 0]) == int(i):
                    sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i = sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i + 1 / self.data_of_RT_conventional_generators[j, 1]
                    sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i = sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i + self.data_of_RT_conventional_generators[j, 2] / self.data_of_RT_conventional_generators[j, 1]
                else:
                    pass
            #
            if sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i != 0:
                H_RT_temp[i, i] = 1 / sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i
                f_RT_temp[i, 0] = sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i / sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i
            else:
                pass
            #
        #
        #
        # # # # Here we put the H_DA_temp and f_DA_temp in the main matrices H and f
        H[0:self.num_of_nodes , 0:self.num_of_nodes] = H_DA_temp
        # f[0:self.num_of_nodes, 0] = f_DA_temp
        f[0:self.num_of_nodes] = f_DA_temp
        #
        #
        # # # # Here we put the H_RT_temp and f_RT_temp in the main matrices H and f
        for i in range(0,num_of_scenarios):
            H[(i+1)*self.num_of_nodes:(i+2)*self.num_of_nodes , (i+1)*self.num_of_nodes:(i+2)*self.num_of_nodes] = vector_of_probabilities[0,i] * H_RT_temp
            # f[(i+1)*self.num_of_nodes:(i+2)*self.num_of_nodes, 0] = vector_of_probabilities[0,i] * f_RT_temp
            f[(i + 1) * self.num_of_nodes:(i + 2) * self.num_of_nodes] = vector_of_probabilities[0, i] * f_RT_temp
        #
        #
        #
        #
        #
        # # # add the penalty elements
        # (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)
        f[(self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios)) : (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) )] = penalty_factor * np.ones(( ((self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios)) , 1 )) #.reshape(((self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) , ))
        #
        #
        #
        ## -------------------------------------------------------------------------------------------------------------
        # forming the A_eq and b_eq matrices for the optimization
        #
        # set the net nodal generation (note that this is not the net nodal injection)
        # this is the net generation made by the DA and RT generators
        # A_eq_part_1 and b_eq_part_1 corresponds to the equation: P_DA + P_RT - net_nodal_injection_by_generators = 0
        A_eq_part_1 = np.zeros( ( ( self.num_of_nodes * num_of_scenarios ) , (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios) ) )
        b_eq_part_1 = np.zeros(((self.num_of_nodes * num_of_scenarios), 1 ))
        #
        for i in range(0,num_of_scenarios):
            A_eq_part_1[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  , 0 : self.num_of_nodes ] = np.eye( self.num_of_nodes  , self.num_of_nodes)   #
            A_eq_part_1[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  , (i+1) * self.num_of_nodes : (i + 2) * self.num_of_nodes ] = np.eye( self.num_of_nodes , self.num_of_nodes)
            A_eq_part_1[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  ,  self.num_of_nodes + ( num_of_scenarios * self.num_of_nodes ) + i * self.num_of_nodes : self.num_of_nodes + ( num_of_scenarios * self.num_of_nodes ) + (i + 1) * self.num_of_nodes ] = -1 * np.eye(self.num_of_nodes, self.num_of_nodes)
        #
        #
        #
        A_eq = copy.deepcopy(A_eq_part_1)
        b_eq = copy.deepcopy(b_eq_part_1)
        #
        #
        #
        #
        # A_eq_pqrt_2 and b_eq_part_2
        # here we write the code for the KCL equations for the nodes at each scenario
        #
        A_eq_part_2 = np.zeros(((self.num_of_nodes * num_of_scenarios), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_eq_part_2 = np.zeros(((self.num_of_nodes * num_of_scenarios), 1))
        #
        for i in range(0,num_of_scenarios):
            A_eq_part_2[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  ,  ( self.num_of_nodes + (num_of_scenarios * self.num_of_nodes) +  i * self.num_of_nodes ) : ( self.num_of_nodes + (num_of_scenarios * self.num_of_nodes) +  (i + 1) * self.num_of_nodes )  ] = -1 * np.eye( self.num_of_nodes  , self.num_of_nodes)   #
            A_eq_part_2[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  ,  ( self.num_of_nodes + (num_of_scenarios * self.num_of_nodes) + (num_of_scenarios * self.num_of_nodes) + (num_of_scenarios * self.num_of_branches) + (num_of_scenarios * self.num_of_branches) +  i * self.num_of_nodes ) : ( self.num_of_nodes + (num_of_scenarios * self.num_of_nodes) + (num_of_scenarios * self.num_of_nodes) + (num_of_scenarios * self.num_of_branches) + (num_of_scenarios * self.num_of_branches) +  (i + 1) * self.num_of_nodes )  ] =  self.susceptance_matrix.reshape(( self.num_of_nodes , self.num_of_nodes ))
        #
        #
        for i in range(0,num_of_scenarios):
            net_nodal_withdraws_temp = copy.deepcopy(np.array(self.vector_of_nodal_demands, dtype=float))
            for j in range(self.num_of_RPPs):
                the_node_of_current_RPP = int(self.data_of_RPPs[j, 0])
                net_nodal_withdraws_temp[the_node_of_current_RPP] = net_nodal_withdraws_temp[the_node_of_current_RPP] - generated_scenarios[i, j]
                #
            #
            b_eq_part_2[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  , 0] = -1 * net_nodal_withdraws_temp
        #
        #
        #
        A_eq = np.concatenate((A_eq, A_eq_part_2), axis=0)
        b_eq = np.concatenate((b_eq, b_eq_part_2), axis=0)
        #
        #
        #
        ## setting the injection of the nodes without DA generator eqaul to zero
        if not set(set(list(range(0, self.num_of_nodes)))).issubset(self.data_of_DA_conventional_generators[:, 0]):  # there exists some nodes without DA generator on them
            set_of_nodes_without_DA_generator = set(list(range(0, self.num_of_nodes))) - set(self.data_of_DA_conventional_generators[:, 0])
            num_of_nodes_without_DA_generator = len(set_of_nodes_without_DA_generator)
            A_eq_part_3__DA = np.zeros((num_of_nodes_without_DA_generator, ((self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios))))
            b_eq_part_3__DA = np.zeros((num_of_nodes_without_DA_generator, 1))
            #
            temp_counter = 0
            for i in set_of_nodes_without_DA_generator:  # the insert power at node i is zero because the there is no conventional DA generator at this node
                A_eq_part_3__DA[temp_counter, i] = 1
                b_eq_part_3__DA[temp_counter, 0] = 0
                temp_counter += 1
            #
            #
            # A_eq = [A_eq ; addtional_block_to__A_eq];
            # b_eq = [b_eq ; addtional_block_to__b_eq];
            A_eq = np.concatenate((A_eq, A_eq_part_3__DA), axis=0)
            b_eq = np.concatenate((b_eq, b_eq_part_3__DA), axis=0)
            #
            #
        else:
            pass
            #
        #
        #
        #
        #
        ## setting the injection of the nodes without DA generator eqaul to zero
        if not set(set(list(range(0, self.num_of_nodes)))).issubset(self.data_of_RT_conventional_generators[:,0]):  # there exists some nodes without DA generator on them
            set_of_nodes_without_RT_generator = set(list(range(0, self.num_of_nodes))) - set(self.data_of_RT_conventional_generators[:, 0])
            num_of_nodes_without_RT_generator = len(set_of_nodes_without_RT_generator)
            A_eq_part_3__RT = np.zeros(((num_of_nodes_without_RT_generator * num_of_scenarios), ((self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios))))
            b_eq_part_3__RT = np.zeros(((num_of_nodes_without_RT_generator * num_of_scenarios), 1))
            #
            for j in range(0,num_of_scenarios):
                temp_counter = 0
                for i in set_of_nodes_without_RT_generator:  # the insert power at node i is zero because the there is no conventional DA generator at this node
                    A_eq_part_3__RT[ ( (j * num_of_nodes_without_RT_generator) +  temp_counter ) , ( self.num_of_nodes + (j * self.num_of_nodes) + i) ] = 1
                    b_eq_part_3__RT[ ( (j * num_of_nodes_without_RT_generator) +  temp_counter ) , 0] = 0
                    temp_counter += 1
            #
            #
            # A_eq = [A_eq ; addtional_block_to__A_eq];
            # b_eq = [b_eq ; addtional_block_to__b_eq];
            A_eq = np.concatenate((A_eq, A_eq_part_3__RT), axis=0)
            b_eq = np.concatenate((b_eq, b_eq_part_3__RT), axis=0)
            #
            #
        else:
            pass
            #
        #
        #
        #
        #
        #
        #
        #
        ## setting the angle of the slack node equal to zero for each scenario
        #
        A_eq_part_5 = np.zeros(( num_of_scenarios , (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_eq_part_5 = np.zeros(( num_of_scenarios , 1 ))
        #
        for i in range(0,num_of_scenarios):
            A_eq_part_5[i , (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + self.num_of_branches * num_of_scenarios + i * self.num_of_nodes + self.index_of_slack_node) ] = 1
            b_eq_part_5[i] = 0
        #
        A_eq = np.concatenate((A_eq, A_eq_part_5), axis=0)
        b_eq = np.concatenate((b_eq, b_eq_part_5), axis=0)
        #
        #
        #
        #
        #
        # # this part is for calculating the expected cost at a NE, and hence we fix the generation of DA conventional generators
        #if False:
            #A_eq_part_6 = np.zeros((3, (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
            #b_eq_part_6 = np.zeros((3, 1))
            ## For NE #1
            #A_eq_part_6[0, 7] = 1
            #b_eq_part_6[0] = 193.14900208
            #A_eq_part_6[1, 8] = 1
            #b_eq_part_6[1] = 135.19975516
            #A_eq_part_6[2, 11] = 1
            #b_eq_part_6[2] = 70.92913888
            #
            #
            #A_eq = np.concatenate((A_eq, A_eq_part_6), axis=0)
            #b_eq = np.concatenate((b_eq, b_eq_part_6), axis=0)
        #else:
            #pass
        #
        #
        #
        # this part is solely for debugging and should not be a part of the program
        # here the dual variables are set to zero
        # A_eq_part_7_positive = np.zeros(((num_of_scenarios*self.num_of_branches), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        # A_eq_part_7_positive[:,(self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios)):(self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios))] = np.eye( num_of_scenarios * self.num_of_branches , num_of_scenarios * self.num_of_branches )
        # b_eq_part_7_positive = np.zeros(((num_of_scenarios * self.num_of_branches), 1))
        # A_eq = np.concatenate((A_eq, A_eq_part_7_positive), axis=0)
        # b_eq = np.concatenate((b_eq, b_eq_part_7_positive), axis=0)
        # #
        # A_eq_part_7_negative = np.zeros(((num_of_scenarios * self.num_of_branches), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        # A_eq_part_7_negative[:,(self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios)):(self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios))] = np.eye( num_of_scenarios * self.num_of_branches, num_of_scenarios * self.num_of_branches )
        # b_eq_part_7_negative = np.zeros(((num_of_scenarios * self.num_of_branches), 1))
        # A_eq = np.concatenate((A_eq, A_eq_part_7_negative), axis=0)
        # b_eq = np.concatenate((b_eq, b_eq_part_7_negative), axis=0)
        #
        ## -------------------------------------------------------------------------------------------------------------
        ## note that in this stochastic programming, there is no inequality equation
        #
        #
        #
        ## ## ## ## ## ## setting the slack variables for the + and - line flow limit inequalities
        # for + line flows
        A_ineq_lines_positive = np.zeros(((self.num_of_branches * num_of_scenarios), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_ineq_lines_positive = np.zeros(((self.num_of_branches * num_of_scenarios), 1)).reshape(((self.num_of_branches * num_of_scenarios), 1))
        #
        for i in range(0, num_of_scenarios):
            A_ineq_lines_positive[(i * self.num_of_branches): ((i + 1) * self.num_of_branches), (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + i * self.num_of_branches): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + (i + 1) * self.num_of_branches)] = -1 * np.eye(self.num_of_branches, self.num_of_branches)
            A_ineq_lines_positive[(i * self.num_of_branches): ((i + 1) * self.num_of_branches), (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + self.num_of_branches * num_of_scenarios + i * self.num_of_nodes): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + self.num_of_branches * num_of_scenarios + (i + 1) * self.num_of_nodes)] = +1 * self.matrix_of_angles_to_line_flows
            #
            b_ineq_lines_positive[(i * self.num_of_branches): ((i + 1) * self.num_of_branches)] = copy.deepcopy(self.branch_capacities.reshape((self.num_of_branches, 1)))
        #
        #
        A_ineq = A_ineq_lines_positive
        b_ineq = b_ineq_lines_positive
        #
        #
        #
        ##
        # for - line flows
        A_ineq_lines_negative = np.zeros(((self.num_of_branches * num_of_scenarios), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_ineq_lines_negative = np.zeros(((self.num_of_branches * num_of_scenarios), 1)).reshape(((self.num_of_branches * num_of_scenarios), 1))
        #
        #
        for i in range(0, num_of_scenarios):
            A_ineq_lines_negative[(i * self.num_of_branches): ((i + 1) * self.num_of_branches), (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + i * self.num_of_branches): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + (i + 1) * self.num_of_branches)] = -1 * np.eye(self.num_of_branches, self.num_of_branches)
            A_ineq_lines_negative[(i * self.num_of_branches): ((i + 1) * self.num_of_branches), (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + self.num_of_branches * num_of_scenarios + i * self.num_of_nodes): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + self.num_of_branches * num_of_scenarios + (i + 1) * self.num_of_nodes)] = -1 * self.matrix_of_angles_to_line_flows
            #
            b_ineq_lines_negative[(i * self.num_of_branches): ((i + 1) * self.num_of_branches)] = copy.deepcopy(self.branch_capacities.reshape((self.num_of_branches, 1)))
        #
        #
        A_ineq = np.concatenate((A_ineq, A_ineq_lines_negative), axis=0)
        b_ineq = np.concatenate((b_ineq, b_ineq_lines_negative), axis=0)
        #
        #
        # # #
        A_ineq___big_submtrx_for_positive_dual_lines_direct_flow = np.zeros(((num_of_scenarios * self.num_of_branches), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_ineq___big_submtrx_for_positive_dual_lines_direct_flow = np.zeros(((num_of_scenarios * self.num_of_branches) , 1))
        for i in range(0, num_of_scenarios):
            A_ineq_positive_dual_lines_direct_flow = np.zeros(((self.num_of_branches), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
            A_ineq_positive_dual_lines_direct_flow[:, (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + i * self.num_of_branches): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + (i + 1) * self.num_of_branches)] = -1*np.eye((self.num_of_branches) , (self.num_of_branches))
            b_ineq_positive_dual_lines_direct_flow = np.zeros((self.num_of_branches , 1))
            #
            A_ineq___big_submtrx_for_positive_dual_lines_direct_flow[i*self.num_of_branches:(i+1)*self.num_of_branches , :] = A_ineq_positive_dual_lines_direct_flow
            b_ineq___big_submtrx_for_positive_dual_lines_direct_flow[i * self.num_of_branches:(i + 1) * self.num_of_branches, :] = b_ineq_positive_dual_lines_direct_flow
            #
        A_ineq = np.concatenate((A_ineq, A_ineq___big_submtrx_for_positive_dual_lines_direct_flow), axis=0)
        b_ineq = np.concatenate((b_ineq, b_ineq___big_submtrx_for_positive_dual_lines_direct_flow), axis=0)
        #
        #
        #
        A_ineq___big_submtrx_for_negative_dual_lines_direct_flow = np.zeros(((num_of_scenarios * self.num_of_branches),(self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_ineq___big_submtrx_for_negative_dual_lines_direct_flow = np.zeros(((num_of_scenarios * self.num_of_branches) , 1))
        for i in range(0, num_of_scenarios):
            A_ineq_negative_dual_lines_direct_flow = np.zeros(((self.num_of_branches), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
            A_ineq_negative_dual_lines_direct_flow[:, (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + i * self.num_of_branches): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + (i + 1) * self.num_of_branches)] = -1 * np.eye((self.num_of_branches),(self.num_of_branches))
            b_ineq_negative_dual_lines_direct_flow = np.zeros((self.num_of_branches, 1))
            #
            A_ineq___big_submtrx_for_negative_dual_lines_direct_flow[i * self.num_of_branches:(i + 1) * self.num_of_branches, :] = A_ineq_negative_dual_lines_direct_flow
            b_ineq___big_submtrx_for_negative_dual_lines_direct_flow[i * self.num_of_branches:(i + 1) * self.num_of_branches, :] = b_ineq_negative_dual_lines_direct_flow
            #
        A_ineq = np.concatenate((A_ineq, A_ineq___big_submtrx_for_negative_dual_lines_direct_flow), axis=0)
        b_ineq = np.concatenate((b_ineq, b_ineq___big_submtrx_for_negative_dual_lines_direct_flow), axis=0)
        ####
        H_cvxpy = np.array(H, dtype=float)
        f_cvxpy = np.array(f, dtype=float).reshape(((self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios) , 1))
        A_ineq_cvxpy = np.array(A_ineq, dtype=float)
        b_ineq_cvxpy = np.array(b_ineq, dtype=float).reshape(b_ineq.shape[0] , )
        A_eq_cvxpy = np.array(A_eq, dtype=float)
        b_eq_cvxpy = np.array(b_eq, dtype=float)
        b_eq_cvxpy = b_eq_cvxpy.reshape(len(b_eq_cvxpy) , )
        # b_eq_cvxpy = np.array(b_eq_no_formatting, dtype=float)
        # # # # # # # # # # # # # # # # # # # #
        number_of_variable = (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)
        x_variables_cvxpy = cvxpy.Variable(number_of_variable)
        #
        obj_fun_cvxpy = 0.5 * cvxpy.quad_form(x_variables_cvxpy, H_cvxpy) + f_cvxpy.T * x_variables_cvxpy.T
        #
        # constraints_equl_and_inequl = [A_eq_cvxpy * x_variables_cvxpy.T == b_eq_cvxpy]
        #
        constraints_equl_and_inequl = [A_eq_cvxpy * x_variables_cvxpy.T == b_eq_cvxpy ,
                                        A_ineq_cvxpy * x_variables_cvxpy.T <= b_ineq_cvxpy]
        #
        #
        problem__DA_dc_OPF = cvxpy.Problem(cvxpy.Minimize(obj_fun_cvxpy), constraints_equl_and_inequl)
        #
        try:
            # ... of the objective function at the optimal point
            value_of_obj_fun_at_optimal_point = problem__DA_dc_OPF.solve(solver=cvxpy.GUROBI, verbose=True , max_iter=400000,eps_abs=10**(-12), eps_rel=10**(-12))
#            value_of_obj_fun_at_optimal_point = problem__DA_dc_OPF.solve(solver=cvxpy.GUROBI, verbose=True , max_iter=400000,eps_abs=10**(-8), eps_rel=10**(-8))
#            value_of_obj_fun_at_optimal_point = problem__DA_dc_OPF.solve(verbose=True , max_iter=400000,eps_abs=10**(-12), eps_rel=10**(-12))
            #
        except cvxpy.error.SolverError:
            value_of_obj_fun_at_optimal_point = None
            DC_power_flow_has_feasible_solution = False  # it does not
            DA_nodal_injections = np.array([])
            #
            return value_of_obj_fun_at_optimal_point , DA_nodal_injections , DC_power_flow_has_feasible_solution , _ , _
        #
        #
        if problem__DA_dc_OPF.status == 'optimal':
            DC_power_flow_has_feasible_solution = True
            #
            #
            DA_nodal_injections = x_variables_cvxpy.value[0:self.num_of_nodes]
            #
            # to check the values of different parameters
            RT_nodal_injections_for_scenarios = x_variables_cvxpy.value[self.num_of_nodes:self.num_of_nodes + num_of_scenarios * self.num_of_nodes]
            net_nodal_injections_of_generators_for_scenarios = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios * self.num_of_nodes: self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes]
            slack_variables_for_flows_in_defined_direction = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes: self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches]
            slack_variables_for_flows_in_reverse_of_defined_direction = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches: self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches + num_of_scenarios * self.num_of_branches]
            nodal_angles_for_scenarios = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches + num_of_scenarios * self.num_of_branches: self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches + num_of_scenarios * self.num_of_branches + num_of_scenarios * self.num_of_nodes]
            #
                #
        else:
            DC_power_flow_has_feasible_solution = 'Wierd situation! Please check the code'
            #
            #
        #
        DA_nodal_injections = x_variables_cvxpy.value[0:self.num_of_nodes]
        #
        #
        #
        #
        dual_variables_for_directed_line_flows = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios*self.num_of_nodes + num_of_scenarios*self.num_of_nodes : self.num_of_nodes + num_of_scenarios*self.num_of_nodes + num_of_scenarios*self.num_of_nodes + num_of_scenarios*self.num_of_branches]
        #
        dual_variables_for_reverse_of_directed_line_flows = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches : self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches + num_of_scenarios * self.num_of_branches]
        #
        #
        # if the line is not congested, then the corresponding value should be negative ( = flow - Line_Capacity)
        extra_flow_of_lines_in_defined_direction_with_slack_variables = np.add(np.matmul( A_ineq_lines_positive , x_variables_cvxpy.value.reshape(len(x_variables_cvxpy.value),1) ) , -1*b_ineq_lines_positive.reshape(len(b_ineq_lines_positive),1))
        #
        #
        # if the line is not congested, then the corresponding value should be negative ( = flow - Line_Capacity)
        extra_flow_of_lines_in_reverse_of_defined_direction_with_slack_variables = np.add(np.matmul( A_ineq_lines_negative , x_variables_cvxpy.value.reshape(len(x_variables_cvxpy.value),1) ) , -1*b_ineq_lines_negative.reshape(len(b_ineq_lines_negative),1))
        #
        #
        #
        # if the line is not congested, then the corresponding value should be negative ( = flow - Line_Capacity)
        extra_flow_of_lines_in_defined_direction_without_slack_variables = np.add(extra_flow_of_lines_in_defined_direction_with_slack_variables.reshape(len(extra_flow_of_lines_in_defined_direction_with_slack_variables),1) , dual_variables_for_directed_line_flows.reshape(len(dual_variables_for_directed_line_flows),1))
        #
        #
        # if the line is not congested, then the corresponding value should be negative ( = flow - Line_Capacity)
        extra_flow_of_lines_in_reverse_of_defined_direction_without_slack_variables = np.add(extra_flow_of_lines_in_reverse_of_defined_direction_with_slack_variables.reshape(len(extra_flow_of_lines_in_reverse_of_defined_direction_with_slack_variables),1), dual_variables_for_reverse_of_directed_line_flows.reshape(len(dual_variables_for_reverse_of_directed_line_flows),1))
        #
        #
        #
        #
        #
        # # # here we plot the slack variables for analyzing purposes
        # fig = plt.figure(figsize=(12, 8))
        # plt.scatter(range(0,num_of_scenarios*self.num_of_branches) , slack_variables_for_flows_in_defined_direction , label = 'Slack Var - flow in def direction')
        # plt.scatter(range(0, num_of_scenarios * self.num_of_branches), slack_variables_for_flows_in_reverse_of_defined_direction , label = 'Slack Var - flow in reverse of def direction')
        # plt.plot(range(0, num_of_scenarios * self.num_of_branches), np.zeros((num_of_scenarios * self.num_of_branches) , ) , 'k' , label = 'Zero line')
        # plt.title('Slack variables of transmission line constraints for penalty = {0} and # scenarios = {1}'.format(penalty_factor , num_of_scenarios))
        # plt.legend(loc = "lower right")
        # plt.show()
        #
        #
        # here we find the congestion pattern for each scenario
        matrix_of_congestion_pttrns_for_scenarios = np.zeros(( num_of_scenarios  , self.num_of_branches ))
        #
        threshold_cong = 10**(-3)
        for i in range(0,num_of_scenarios):
            print('i = {}'.format(i))
            for j in range(0,self.num_of_branches):
                if extra_flow_of_lines_in_defined_direction_without_slack_variables[  i*self.num_of_branches + j  ] >= -1*threshold_cong:
                    matrix_of_congestion_pttrns_for_scenarios[i,j] = 1
                elif extra_flow_of_lines_in_reverse_of_defined_direction_without_slack_variables[ i*self.num_of_branches + j  ] >= -1*threshold_cong:
                    matrix_of_congestion_pttrns_for_scenarios[i,j] = 2
                else:
                    pass
                #
            #
        #
        #
        #
        #
        # This part is for the calculation of the dual variables
        LMPs_for_all_scenarios = constraints_equl_and_inequl[0].dual_value[num_of_scenarios * self.num_of_nodes:2*num_of_scenarios * self.num_of_nodes]
        #
        LMP_sum_for_all_scenarios = np.zeros((self.num_of_nodes , 1)).reshape(self.num_of_nodes , 1)
        for i in range(0,num_of_scenarios):
            LMP_sum_for_all_scenarios = np.add( LMP_sum_for_all_scenarios , LMPs_for_all_scenarios[i*self.num_of_nodes : (i+1)*self.num_of_nodes].reshape(self.num_of_nodes , 1) )
        #
        LMP_average = LMP_sum_for_all_scenarios # note that the objective function already included the probability of each scenario@
        #
        #
#        DA_commitments_at_NE = [42.4811 , 42.4811 , 22.1016 , 22.1016]
#        node_of_RPPs = [4 , 4 , 11 , 11]
#        DA_LMPs = [12.9111, 12.9111, 5.14984, 5.14984]
#        expected_payoff_of_RPPs = [0,0,0,0]
#        for j in range(0,num_of_scenarios):
#            expected_payoff_of_RPPs[0] = DA_LMPs[0] * DA_commitments_at_NE[0] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[0]] * (generated_scenarios[j, 0] - DA_commitments_at_NE[0])
#            expected_payoff_of_RPPs[1] = DA_LMPs[1] * DA_commitments_at_NE[1] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[1]] * (generated_scenarios[j, 1] - DA_commitments_at_NE[1])
#            expected_payoff_of_RPPs[2] = DA_LMPs[2] * DA_commitments_at_NE[2] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[2]] * (generated_scenarios[j, 2] - DA_commitments_at_NE[2])
#            expected_payoff_of_RPPs[3] = DA_LMPs[3] * DA_commitments_at_NE[3] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[3]] * (generated_scenarios[j, 3] - DA_commitments_at_NE[3])
        #
        #
#        DA_commitments_at_NE = [77.270 , 46.095]
#        node_of_RPPs = [4 , 11]
#        DA_LMPs = [13.1127, 5.07846]
#        expected_payoff_of_RPPs = [0, 0]
#        for j in range(0, num_of_scenarios):
#            expected_payoff_of_RPPs[0] = DA_LMPs[0] * DA_commitments_at_NE[0] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[0]] * (generated_scenarios[j, 0] - DA_commitments_at_NE[0])
#            expected_payoff_of_RPPs[1] = DA_LMPs[1] * DA_commitments_at_NE[1] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[1]] * (generated_scenarios[j, 1] - DA_commitments_at_NE[1])
        #
        #
        return value_of_obj_fun_at_optimal_point , DA_nodal_injections , matrix_of_congestion_pttrns_for_scenarios , DC_power_flow_has_feasible_solution , LMP_average
    #
    #
    #
    def Expected_System_Cost_at_NE(self , DA_Generations_at_NE , num_of_scenarios , penalty_factor = 5000):
        # this function is for the calculation of the social welfare calculations and to make the
        #
        generated_scenarios = np.array(self.generate_samples_for_the_case_of_multivariate_normal_distributions(num_of_scenarios)).reshape((num_of_scenarios , self.num_of_RPPs))
#        generated_scenarios = np.array([[70,50]]).reshape(1,2)
        # each row of generated_scenarios corresponds to one scenario
        #
        # 'DA_Generations_at_NE' is the power dispatches of DA generators at NE. 
        #
        # this function returns:
        # nodal_injections , LMPs , list_of_lines_congested_or_not , DC_power_flow_has_feasible_solution
        #
        #
        # DC_power_flow_has_feasible_solution = is a variable that takes ...
        # ... a value of True (if feasible) or False (if infeasible). If the rsultant DC power flow of the ...
        # ... combination of current DA commitment of the RPPs, the nodal demands, the line capacities, the ...
        # ... structure of the network, and the location of the DA conventional generators gives us a feasible ...
        # ... solution, then we will have a feasible solution, and current set of DA commitments ...
        # ... of the RPPs MAY be a pure NE. If there is no feasible solution to the DC ...
        # ... power flow, then definitely the current DA commitments are not a pure NE.
        #
        #
        #
        # list_of_lines_congested_or_not  is a 1*num_of_branches numpy vector where if the line #i is ...
        # ... congested in the direction of definition, the the i th element of this vector is 1, and if the ...
        # ... line #i is congested in the reverse direction of its definition,the i th element of this vector is -1, ...
        # ... and if the line #i is not congested, the i th element of this vector is 0.
        #
        #
        # note that this function considers the realizations of the RPPs as negative loads.
        # note that for the social optimum calculations (i.e. this function), no DA commitments are required
        #
        #
        #
        # this function calculates the optimal power generation of the DA generators and RT generators and the  ...
        # ... corresponding total expected cost function!
        #
        # the variables of the DC power flow optimization are:
        # x = [ nodal injections for DA generators ; nodal injections for the RT generators ; total nodal injections ; nodal angles ; labmbda_positive_lines ; lambda_negative_lines]
        #
        # nodal injections for DA generators ==> is the injection for the DA conv generators. for the nodes without DA conv generators, the corresponding elements in this vector is zero
        # (length of these variables isL num_of_nodes)
        #
        # nodal injections for RT generators ==> is the injection for the RT conv generators for each scenario. for the nodes without RT conv generators, the corresponding elements in this vector is zero at each scenario!
        # (note that the size of this vector is: num_of_nodes * num_of_scenarios )
        #
        # total nodal injections ==> this set of variables is the sum of the previous two vectors. This set of variables represent the net injections in each node
        #
        #
        #
        # note that always the nodal angle of the reference node is zero (it is set in the equality constraint too)
        #
        #
        # we are trying to minimize 0.5 * xT * H * x + fT * x
        #
        #
        #
        # note that here we assume that there is one DA conv generators at each ...
        # node and one RT conv generators at each node. then for those nodes that do not have any DA conv generators, we make ...
        # the corresponding elements in the 'nodal injections for DA generators' to zero, and for the nodes without RT conv generators, ...
        # ... we make the corresponding elements in the 'nodal injections for RT generators' set to zero!
        #
        #
        # The variable vector x consists of 2*num_of_nodes variables: the first num_of_nodes ...
        # ... variables correspond to the DA conv injections, and the second num_of_nodes ...
        # ... variables correspond to the nodal angles.
        #
        #
        #
        ## -------------------------------------------------------------------------------------------------------------
        ## Calculating H and f (Quadratic and linear coefficient terms in the objective function of the OPF)
        H = copy.deepcopy(np.zeros(((self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios ), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios)))))
        f = copy.deepcopy(np.zeros(((self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios ), 1))).reshape(( (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios) , 1))
        #
        # explanation of the size of f (and H)
        # first set of variables consists of (num_of_nodes) variables which correspond to the nodal injections corresponding to the DA conv generators
        # second set of variables consists of (self.num_of_nodes * num_of_scenarios) variables which correspond to the nodal injections corresponding to the RT conv generators at all the scenarios
        # third set of variables consists of (self.num_of_nodes * num_of_scenarios) variables which correspond to the net nodal injections corresponding to both DA and RT conv generators at all the scenarios (if we copy the first set of variables for each scenario and then add it to the second set of variables, we get this vector)
        # forth set of variables consists of (self.num_of_branches * num_of_scenarios) variables which is the slack variables corresponding to the + line flow inequlalities at all scenarios.
        # fifth set of variables consists of (self.num_of_branches * num_of_scenarios) variables which is the slack variables corresponding to the - line flow inequlalities at all scenarios.
        # sixth set of variables consists of (self.num_of_nodes * num_of_scenarios) variables which correspinds to the nodal angles at all the scenarios
        #
        #
        #
        #
        vector_of_probabilities = (1 / num_of_scenarios) * np.ones((1, num_of_scenarios))
        #
        ##
        ## forming the quadratic and linear matrix and vector in the objective function
        # for the DA generators
        H_DA_temp = copy.deepcopy(np.zeros((self.num_of_nodes, self.num_of_nodes)))
        f_DA_temp = copy.deepcopy(np.zeros((self.num_of_nodes, 1))).reshape((self.num_of_nodes , 1))
        for i in range(self.num_of_nodes):
            #
            sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i = 0
            sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i = 0
            #
            for j in range(self.num_of_DA_conv_generators):
                if int(self.data_of_DA_conventional_generators[j, 0]) == int(i):
                    sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i = sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i + 1 / self.data_of_DA_conventional_generators[j, 1]
                    sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i = sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i + self.data_of_DA_conventional_generators[j, 2] / self.data_of_DA_conventional_generators[j, 1]
                else:
                    pass
            #
            if sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i != 0:
                H_DA_temp[i, i] = 1 / sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i
                f_DA_temp[i, 0] = sum_of_beta_over_alpha_of_DA_conv_gen_located_at_node_i / sum_of_iverse_of_alpha_of_DA_conv_gen_located_at_node_i
            else:
                pass
            #
        #
        #
        #
        # for the RT generators
        H_RT_temp = copy.deepcopy(np.zeros((self.num_of_nodes , self.num_of_nodes)))
        f_RT_temp = copy.deepcopy(np.zeros((self.num_of_nodes , 1))).reshape((self.num_of_nodes , 1))
        #
        for i in range(self.num_of_nodes):
            #
            sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i = 0
            sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i = 0
            #
            for j in range(self.num_of_RT_conv_generators):
                if int(self.data_of_RT_conventional_generators[j, 0]) == int(i):
                    sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i = sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i + 1 / self.data_of_RT_conventional_generators[j, 1]
                    sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i = sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i + self.data_of_RT_conventional_generators[j, 2] / self.data_of_RT_conventional_generators[j, 1]
                else:
                    pass
            #
            if sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i != 0:
                H_RT_temp[i, i] = 1 / sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i
                f_RT_temp[i, 0] = sum_of_beta_over_alpha_of_RT_conv_gen_located_at_node_i / sum_of_iverse_of_alpha_of_RT_conv_gen_located_at_node_i
            else:
                pass
            #
        #
        #
        # # # # Here we put the H_DA_temp and f_DA_temp in the main matrices H and f
        H[0:self.num_of_nodes , 0:self.num_of_nodes] = H_DA_temp
        # f[0:self.num_of_nodes, 0] = f_DA_temp
        f[0:self.num_of_nodes] = f_DA_temp
        #
        #
        # # # # Here we put the H_RT_temp and f_RT_temp in the main matrices H and f
        for i in range(0,num_of_scenarios):
            H[(i+1)*self.num_of_nodes:(i+2)*self.num_of_nodes , (i+1)*self.num_of_nodes:(i+2)*self.num_of_nodes] = vector_of_probabilities[0,i] * H_RT_temp
            # f[(i+1)*self.num_of_nodes:(i+2)*self.num_of_nodes, 0] = vector_of_probabilities[0,i] * f_RT_temp
            f[(i + 1) * self.num_of_nodes:(i + 2) * self.num_of_nodes] = vector_of_probabilities[0, i] * f_RT_temp
        #
        #
        #
        #
        #
        # # # add the penalty elements
        # (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)
        f[(self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios)) : (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) )] = penalty_factor * np.ones(( ((self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios)) , 1 )) #.reshape(((self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) , ))
        #
        #
        #
        ## -------------------------------------------------------------------------------------------------------------
        # forming the A_eq and b_eq matrices for the optimization
        #
        # set the net nodal generation (note that this is not the net nodal injection)
        # this is the net generation made by the DA and RT generators
        # A_eq_part_1 and b_eq_part_1 corresponds to the equation: P_DA + P_RT - net_nodal_injection_by_generators = 0
        A_eq_part_1 = np.zeros( ( ( self.num_of_nodes * num_of_scenarios ) , (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios) ) )
        b_eq_part_1 = np.zeros(((self.num_of_nodes * num_of_scenarios), 1 ))
        #
        for i in range(0,num_of_scenarios):
            A_eq_part_1[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  , 0 : self.num_of_nodes ] = np.eye( self.num_of_nodes  , self.num_of_nodes)   #
            A_eq_part_1[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  , (i+1) * self.num_of_nodes : (i + 2) * self.num_of_nodes ] = np.eye( self.num_of_nodes , self.num_of_nodes)
            A_eq_part_1[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  ,  self.num_of_nodes + ( num_of_scenarios * self.num_of_nodes ) + i * self.num_of_nodes : self.num_of_nodes + ( num_of_scenarios * self.num_of_nodes ) + (i + 1) * self.num_of_nodes ] = -1 * np.eye(self.num_of_nodes, self.num_of_nodes)
        #
        #
        #
        A_eq = copy.deepcopy(A_eq_part_1)
        b_eq = copy.deepcopy(b_eq_part_1)
        #
        #
        #
        #
        # A_eq_pqrt_2 and b_eq_part_2
        # here we write the code for the KCL equations for the nodes at each scenario
        #
        A_eq_part_2 = np.zeros(((self.num_of_nodes * num_of_scenarios), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_eq_part_2 = np.zeros(((self.num_of_nodes * num_of_scenarios), 1))
        #
        for i in range(0,num_of_scenarios):
            A_eq_part_2[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  ,  ( self.num_of_nodes + (num_of_scenarios * self.num_of_nodes) +  i * self.num_of_nodes ) : ( self.num_of_nodes + (num_of_scenarios * self.num_of_nodes) +  (i + 1) * self.num_of_nodes )  ] = -1 * np.eye( self.num_of_nodes  , self.num_of_nodes)   #
            A_eq_part_2[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  ,  ( self.num_of_nodes + (num_of_scenarios * self.num_of_nodes) + (num_of_scenarios * self.num_of_nodes) + (num_of_scenarios * self.num_of_branches) + (num_of_scenarios * self.num_of_branches) +  i * self.num_of_nodes ) : ( self.num_of_nodes + (num_of_scenarios * self.num_of_nodes) + (num_of_scenarios * self.num_of_nodes) + (num_of_scenarios * self.num_of_branches) + (num_of_scenarios * self.num_of_branches) +  (i + 1) * self.num_of_nodes )  ] =  self.susceptance_matrix.reshape(( self.num_of_nodes , self.num_of_nodes ))
        #
        #
        for i in range(0,num_of_scenarios):
            net_nodal_withdraws_temp = copy.deepcopy(np.array(self.vector_of_nodal_demands, dtype=float))
            for j in range(self.num_of_RPPs):
                the_node_of_current_RPP = int(self.data_of_RPPs[j, 0])
                net_nodal_withdraws_temp[the_node_of_current_RPP] = net_nodal_withdraws_temp[the_node_of_current_RPP] - generated_scenarios[i, j]
                #
            #
            b_eq_part_2[ i * self.num_of_nodes : (i + 1) * self.num_of_nodes  , 0] = -1 * net_nodal_withdraws_temp
        #
        #
        #
        A_eq = np.concatenate((A_eq, A_eq_part_2), axis=0)
        b_eq = np.concatenate((b_eq, b_eq_part_2), axis=0)
        #
        #
        #
        ## setting the injection of the nodes without DA generator eqaul to zero
        if not set(set(list(range(0, self.num_of_nodes)))).issubset(self.data_of_DA_conventional_generators[:, 0]):  # there exists some nodes without DA generator on them
            set_of_nodes_without_DA_generator = set(list(range(0, self.num_of_nodes))) - set(self.data_of_DA_conventional_generators[:, 0])
            num_of_nodes_without_DA_generator = len(set_of_nodes_without_DA_generator)
            A_eq_part_3__DA = np.zeros((num_of_nodes_without_DA_generator, ((self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios))))
            b_eq_part_3__DA = np.zeros((num_of_nodes_without_DA_generator, 1))
            #
            temp_counter = 0
            for i in set_of_nodes_without_DA_generator:  # the insert power at node i is zero because the there is no conventional DA generator at this node
                A_eq_part_3__DA[temp_counter, i] = 1
                b_eq_part_3__DA[temp_counter, 0] = 0
                temp_counter += 1
            #
            #
            # A_eq = [A_eq ; addtional_block_to__A_eq];
            # b_eq = [b_eq ; addtional_block_to__b_eq];
            A_eq = np.concatenate((A_eq, A_eq_part_3__DA), axis=0)
            b_eq = np.concatenate((b_eq, b_eq_part_3__DA), axis=0)
            #
            #
        else:
            pass
            #
        #
        #
        #
        #
        ## setting the injection of the nodes without DA generator eqaul to zero
        if not set(set(list(range(0, self.num_of_nodes)))).issubset(self.data_of_RT_conventional_generators[:,0]):  # there exists some nodes without DA generator on them
            set_of_nodes_without_RT_generator = set(list(range(0, self.num_of_nodes))) - set(self.data_of_RT_conventional_generators[:, 0])
            num_of_nodes_without_RT_generator = len(set_of_nodes_without_RT_generator)
            A_eq_part_3__RT = np.zeros(((num_of_nodes_without_RT_generator * num_of_scenarios), ((self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios))))
            b_eq_part_3__RT = np.zeros(((num_of_nodes_without_RT_generator * num_of_scenarios), 1))
            #
            for j in range(0,num_of_scenarios):
                temp_counter = 0
                for i in set_of_nodes_without_RT_generator:  # the insert power at node i is zero because the there is no conventional DA generator at this node
                    A_eq_part_3__RT[ ( (j * num_of_nodes_without_RT_generator) +  temp_counter ) , ( self.num_of_nodes + (j * self.num_of_nodes) + i) ] = 1
                    b_eq_part_3__RT[ ( (j * num_of_nodes_without_RT_generator) +  temp_counter ) , 0] = 0
                    temp_counter += 1
            #
            #
            # A_eq = [A_eq ; addtional_block_to__A_eq];
            # b_eq = [b_eq ; addtional_block_to__b_eq];
            A_eq = np.concatenate((A_eq, A_eq_part_3__RT), axis=0)
            b_eq = np.concatenate((b_eq, b_eq_part_3__RT), axis=0)
            #
            #
        else:
            pass
            #
        #
        #
        #
        #
        #
        #
        #
        ## setting the angle of the slack node equal to zero for each scenario
        #
        A_eq_part_5 = np.zeros(( num_of_scenarios , (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_eq_part_5 = np.zeros(( num_of_scenarios , 1 ))
        #
        for i in range(0,num_of_scenarios):
            A_eq_part_5[i , (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + self.num_of_branches * num_of_scenarios + i * self.num_of_nodes + self.index_of_slack_node) ] = 1
            b_eq_part_5[i] = 0
        #
        A_eq = np.concatenate((A_eq, A_eq_part_5), axis=0)
        b_eq = np.concatenate((b_eq, b_eq_part_5), axis=0)
        #
        #
        #
        #
        #
        # # this part is for calculating the expected cost at a NE, and hence we fix the generation of DA conventional generators
        A_eq_part_6 = np.zeros((self.num_of_DA_conv_generators, (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_eq_part_6 = np.zeros((self.num_of_DA_conv_generators, 1))
        # For NE #1
        for ii in range(0,self.num_of_DA_conv_generators):
            A_eq_part_6[int(ii), int(self.data_of_DA_conventional_generators[int(ii),0])] = 1
            b_eq_part_6[int(ii), 0] = DA_Generations_at_NE[int(ii)]
        #
        A_eq = np.concatenate((A_eq, A_eq_part_6), axis=0)
        b_eq = np.concatenate((b_eq, b_eq_part_6), axis=0)
        #
        #
        #
        ## -------------------------------------------------------------------------------------------------------------
        ## note that in this stochastic programming, there is no inequality equation
        #
        #
        #
        ## ## ## ## ## ## setting the slack variables for the + and - line flow limit inequalities
        # for + line flows
        A_ineq_lines_positive = np.zeros(((self.num_of_branches * num_of_scenarios), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_ineq_lines_positive = np.zeros(((self.num_of_branches * num_of_scenarios), 1)).reshape(((self.num_of_branches * num_of_scenarios), 1))
        #
        for i in range(0, num_of_scenarios):
            A_ineq_lines_positive[(i * self.num_of_branches): ((i + 1) * self.num_of_branches), (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + i * self.num_of_branches): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + (i + 1) * self.num_of_branches)] = -1 * np.eye(self.num_of_branches, self.num_of_branches)
            A_ineq_lines_positive[(i * self.num_of_branches): ((i + 1) * self.num_of_branches), (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + self.num_of_branches * num_of_scenarios + i * self.num_of_nodes): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + self.num_of_branches * num_of_scenarios + (i + 1) * self.num_of_nodes)] = +1 * self.matrix_of_angles_to_line_flows
            #
            b_ineq_lines_positive[(i * self.num_of_branches): ((i + 1) * self.num_of_branches)] = copy.deepcopy(self.branch_capacities.reshape((self.num_of_branches, 1)))
        #
        #
        A_ineq = A_ineq_lines_positive
        b_ineq = b_ineq_lines_positive
        #
        #
        #
        ##
        # for - line flows
        A_ineq_lines_negative = np.zeros(((self.num_of_branches * num_of_scenarios), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_ineq_lines_negative = np.zeros(((self.num_of_branches * num_of_scenarios), 1)).reshape(((self.num_of_branches * num_of_scenarios), 1))
        #
        #
        for i in range(0, num_of_scenarios):
            A_ineq_lines_negative[(i * self.num_of_branches): ((i + 1) * self.num_of_branches), (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + i * self.num_of_branches): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + (i + 1) * self.num_of_branches)] = -1 * np.eye(self.num_of_branches, self.num_of_branches)
            A_ineq_lines_negative[(i * self.num_of_branches): ((i + 1) * self.num_of_branches), (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + self.num_of_branches * num_of_scenarios + i * self.num_of_nodes): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + self.num_of_branches * num_of_scenarios + (i + 1) * self.num_of_nodes)] = -1 * self.matrix_of_angles_to_line_flows
            #
            b_ineq_lines_negative[(i * self.num_of_branches): ((i + 1) * self.num_of_branches)] = copy.deepcopy(self.branch_capacities.reshape((self.num_of_branches, 1)))
        #
        #
        A_ineq = np.concatenate((A_ineq, A_ineq_lines_negative), axis=0)
        b_ineq = np.concatenate((b_ineq, b_ineq_lines_negative), axis=0)
        #
        #
        # # #
        A_ineq___big_submtrx_for_positive_dual_lines_direct_flow = np.zeros(((num_of_scenarios * self.num_of_branches), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_ineq___big_submtrx_for_positive_dual_lines_direct_flow = np.zeros(((num_of_scenarios * self.num_of_branches) , 1))
        for i in range(0, num_of_scenarios):
            A_ineq_positive_dual_lines_direct_flow = np.zeros(((self.num_of_branches), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
            A_ineq_positive_dual_lines_direct_flow[:, (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + i * self.num_of_branches): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + (i + 1) * self.num_of_branches)] = -1*np.eye((self.num_of_branches) , (self.num_of_branches))
            b_ineq_positive_dual_lines_direct_flow = np.zeros((self.num_of_branches , 1))
            #
            A_ineq___big_submtrx_for_positive_dual_lines_direct_flow[i*self.num_of_branches:(i+1)*self.num_of_branches , :] = A_ineq_positive_dual_lines_direct_flow
            b_ineq___big_submtrx_for_positive_dual_lines_direct_flow[i * self.num_of_branches:(i + 1) * self.num_of_branches, :] = b_ineq_positive_dual_lines_direct_flow
            #
        A_ineq = np.concatenate((A_ineq, A_ineq___big_submtrx_for_positive_dual_lines_direct_flow), axis=0)
        b_ineq = np.concatenate((b_ineq, b_ineq___big_submtrx_for_positive_dual_lines_direct_flow), axis=0)
        #
        #
        #
        A_ineq___big_submtrx_for_negative_dual_lines_direct_flow = np.zeros(((num_of_scenarios * self.num_of_branches),(self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
        b_ineq___big_submtrx_for_negative_dual_lines_direct_flow = np.zeros(((num_of_scenarios * self.num_of_branches) , 1))
        for i in range(0, num_of_scenarios):
            A_ineq_negative_dual_lines_direct_flow = np.zeros(((self.num_of_branches), (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)))
            A_ineq_negative_dual_lines_direct_flow[:, (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + i * self.num_of_branches): (self.num_of_nodes + self.num_of_nodes * num_of_scenarios + self.num_of_nodes * num_of_scenarios + self.num_of_branches * num_of_scenarios + (i + 1) * self.num_of_branches)] = -1 * np.eye((self.num_of_branches),(self.num_of_branches))
            b_ineq_negative_dual_lines_direct_flow = np.zeros((self.num_of_branches, 1))
            #
            A_ineq___big_submtrx_for_negative_dual_lines_direct_flow[i * self.num_of_branches:(i + 1) * self.num_of_branches, :] = A_ineq_negative_dual_lines_direct_flow
            b_ineq___big_submtrx_for_negative_dual_lines_direct_flow[i * self.num_of_branches:(i + 1) * self.num_of_branches, :] = b_ineq_negative_dual_lines_direct_flow
            #
        A_ineq = np.concatenate((A_ineq, A_ineq___big_submtrx_for_negative_dual_lines_direct_flow), axis=0)
        b_ineq = np.concatenate((b_ineq, b_ineq___big_submtrx_for_negative_dual_lines_direct_flow), axis=0)
        ####
        H_cvxpy = np.array(H, dtype=float)
        f_cvxpy = np.array(f, dtype=float).reshape(((self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios) , 1))
        A_ineq_cvxpy = np.array(A_ineq, dtype=float)
        b_ineq_cvxpy = np.array(b_ineq, dtype=float).reshape(b_ineq.shape[0] , )
        A_eq_cvxpy = np.array(A_eq, dtype=float)
        b_eq_cvxpy = np.array(b_eq, dtype=float)
        b_eq_cvxpy = b_eq_cvxpy.reshape(len(b_eq_cvxpy) , )
        # b_eq_cvxpy = np.array(b_eq_no_formatting, dtype=float)
        # # # # # # # # # # # # # # # # # # # #
        number_of_variable = (self.num_of_nodes + (self.num_of_nodes * num_of_scenarios) + (self.num_of_nodes * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + (self.num_of_branches * num_of_scenarios) + self.num_of_nodes * num_of_scenarios)
        x_variables_cvxpy = cvxpy.Variable(number_of_variable)
        #
        obj_fun_cvxpy = 0.5 * cvxpy.quad_form(x_variables_cvxpy, H_cvxpy) + f_cvxpy.T * x_variables_cvxpy.T
        #
        # constraints_equl_and_inequl = [A_eq_cvxpy * x_variables_cvxpy.T == b_eq_cvxpy]
        #
        constraints_equl_and_inequl = [A_eq_cvxpy * x_variables_cvxpy.T == b_eq_cvxpy ,
                                        A_ineq_cvxpy * x_variables_cvxpy.T <= b_ineq_cvxpy]
        #
        #
        problem__DA_dc_OPF = cvxpy.Problem(cvxpy.Minimize(obj_fun_cvxpy), constraints_equl_and_inequl)
        #
        try:
            # ... of the objective function at the optimal point
            value_of_obj_fun_at_optimal_point = problem__DA_dc_OPF.solve(solver=cvxpy.GUROBI, verbose=True , max_iter=400000,eps_abs=10**(-12), eps_rel=10**(-12))
#            value_of_obj_fun_at_optimal_point = problem__DA_dc_OPF.solve(solver=cvxpy.GUROBI, verbose=True , max_iter=400000,eps_abs=10**(-8), eps_rel=10**(-8))
#            value_of_obj_fun_at_optimal_point = problem__DA_dc_OPF.solve(verbose=True , max_iter=400000,eps_abs=10**(-12), eps_rel=10**(-12))
            #
        except cvxpy.error.SolverError:
            value_of_obj_fun_at_optimal_point = None
            DC_power_flow_has_feasible_solution = False  # it does not
            DA_nodal_injections = np.array([])
            #
            return value_of_obj_fun_at_optimal_point , DA_nodal_injections , DC_power_flow_has_feasible_solution , _ , _
        #
        #
        if problem__DA_dc_OPF.status == 'optimal':
            DC_power_flow_has_feasible_solution = True
            #
            #
            DA_nodal_injections = x_variables_cvxpy.value[0:self.num_of_nodes]
            #
            # to check the values of different parameters
            RT_nodal_injections_for_scenarios = x_variables_cvxpy.value[self.num_of_nodes:self.num_of_nodes + num_of_scenarios * self.num_of_nodes]
            net_nodal_injections_of_generators_for_scenarios = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios * self.num_of_nodes: self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes]
            slack_variables_for_flows_in_defined_direction = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes: self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches]
            slack_variables_for_flows_in_reverse_of_defined_direction = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches: self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches + num_of_scenarios * self.num_of_branches]
            nodal_angles_for_scenarios = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches + num_of_scenarios * self.num_of_branches: self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches + num_of_scenarios * self.num_of_branches + num_of_scenarios * self.num_of_nodes]
            #
                #
        else:
            DC_power_flow_has_feasible_solution = 'Wierd situation! Please check the code'
            #
            #
        #
        DA_nodal_injections = x_variables_cvxpy.value[0:self.num_of_nodes]
        #
        #
        #
        #
        dual_variables_for_directed_line_flows = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios*self.num_of_nodes + num_of_scenarios*self.num_of_nodes : self.num_of_nodes + num_of_scenarios*self.num_of_nodes + num_of_scenarios*self.num_of_nodes + num_of_scenarios*self.num_of_branches]
        #
        dual_variables_for_reverse_of_directed_line_flows = x_variables_cvxpy.value[self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches : self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_nodes + num_of_scenarios * self.num_of_branches + num_of_scenarios * self.num_of_branches]
        #
        #
        # if the line is not congested, then the corresponding value should be negative ( = flow - Line_Capacity)
        extra_flow_of_lines_in_defined_direction_with_slack_variables = np.add(np.matmul( A_ineq_lines_positive , x_variables_cvxpy.value.reshape(len(x_variables_cvxpy.value),1) ) , -1*b_ineq_lines_positive.reshape(len(b_ineq_lines_positive),1))
        #
        #
        # if the line is not congested, then the corresponding value should be negative ( = flow - Line_Capacity)
        extra_flow_of_lines_in_reverse_of_defined_direction_with_slack_variables = np.add(np.matmul( A_ineq_lines_negative , x_variables_cvxpy.value.reshape(len(x_variables_cvxpy.value),1) ) , -1*b_ineq_lines_negative.reshape(len(b_ineq_lines_negative),1))
        #
        #
        #
        # if the line is not congested, then the corresponding value should be negative ( = flow - Line_Capacity)
        extra_flow_of_lines_in_defined_direction_without_slack_variables = np.add(extra_flow_of_lines_in_defined_direction_with_slack_variables.reshape(len(extra_flow_of_lines_in_defined_direction_with_slack_variables),1) , dual_variables_for_directed_line_flows.reshape(len(dual_variables_for_directed_line_flows),1))
        #
        #
        # if the line is not congested, then the corresponding value should be negative ( = flow - Line_Capacity)
        extra_flow_of_lines_in_reverse_of_defined_direction_without_slack_variables = np.add(extra_flow_of_lines_in_reverse_of_defined_direction_with_slack_variables.reshape(len(extra_flow_of_lines_in_reverse_of_defined_direction_with_slack_variables),1), dual_variables_for_reverse_of_directed_line_flows.reshape(len(dual_variables_for_reverse_of_directed_line_flows),1))
        #
        #
        #
        #
        #
        # # # here we plot the slack variables for analyzing purposes
        # fig = plt.figure(figsize=(12, 8))
        # plt.scatter(range(0,num_of_scenarios*self.num_of_branches) , slack_variables_for_flows_in_defined_direction , label = 'Slack Var - flow in def direction')
        # plt.scatter(range(0, num_of_scenarios * self.num_of_branches), slack_variables_for_flows_in_reverse_of_defined_direction , label = 'Slack Var - flow in reverse of def direction')
        # plt.plot(range(0, num_of_scenarios * self.num_of_branches), np.zeros((num_of_scenarios * self.num_of_branches) , ) , 'k' , label = 'Zero line')
        # plt.title('Slack variables of transmission line constraints for penalty = {0} and # scenarios = {1}'.format(penalty_factor , num_of_scenarios))
        # plt.legend(loc = "lower right")
        # plt.show()
        #
        #
        # here we find the congestion pattern for each scenario
        matrix_of_congestion_pttrns_for_scenarios = np.zeros(( num_of_scenarios  , self.num_of_branches ))
        #
        threshold_cong = 10**(-3)
        for i in range(0,num_of_scenarios):
            print('i = {}'.format(i))
            for j in range(0,self.num_of_branches):
                if extra_flow_of_lines_in_defined_direction_without_slack_variables[  i*self.num_of_branches + j  ] >= -1*threshold_cong:
                    matrix_of_congestion_pttrns_for_scenarios[i,j] = 1
                elif extra_flow_of_lines_in_reverse_of_defined_direction_without_slack_variables[ i*self.num_of_branches + j  ] >= -1*threshold_cong:
                    matrix_of_congestion_pttrns_for_scenarios[i,j] = 2
                else:
                    pass
                #
            #
        #
        #
        #
        #
        # This part is for the calculation of the dual variables
        LMPs_for_all_scenarios = constraints_equl_and_inequl[0].dual_value[num_of_scenarios * self.num_of_nodes:2*num_of_scenarios * self.num_of_nodes]
        #
        LMP_sum_for_all_scenarios = np.zeros((self.num_of_nodes , 1)).reshape(self.num_of_nodes , 1)
        for i in range(0,num_of_scenarios):
            LMP_sum_for_all_scenarios = np.add( LMP_sum_for_all_scenarios , LMPs_for_all_scenarios[i*self.num_of_nodes : (i+1)*self.num_of_nodes].reshape(self.num_of_nodes , 1) )
        #
        LMP_average = LMP_sum_for_all_scenarios # note that the objective function already included the probability of each scenario@
        #
        #
#        DA_commitments_at_NE = [42.4811 , 42.4811 , 22.1016 , 22.1016]
#        node_of_RPPs = [4 , 4 , 11 , 11]
#        DA_LMPs = [12.9111, 12.9111, 5.14984, 5.14984]
#        expected_payoff_of_RPPs = [0,0,0,0]
#        for j in range(0,num_of_scenarios):
#            expected_payoff_of_RPPs[0] = DA_LMPs[0] * DA_commitments_at_NE[0] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[0]] * (generated_scenarios[j, 0] - DA_commitments_at_NE[0])
#            expected_payoff_of_RPPs[1] = DA_LMPs[1] * DA_commitments_at_NE[1] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[1]] * (generated_scenarios[j, 1] - DA_commitments_at_NE[1])
#            expected_payoff_of_RPPs[2] = DA_LMPs[2] * DA_commitments_at_NE[2] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[2]] * (generated_scenarios[j, 2] - DA_commitments_at_NE[2])
#            expected_payoff_of_RPPs[3] = DA_LMPs[3] * DA_commitments_at_NE[3] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[3]] * (generated_scenarios[j, 3] - DA_commitments_at_NE[3])
        #
        #
#        DA_commitments_at_NE = [77.270 , 46.095]
#        node_of_RPPs = [4 , 11]
#        DA_LMPs = [13.1127, 5.07846]
#        expected_payoff_of_RPPs = [0, 0]
#        for j in range(0, num_of_scenarios):
#            expected_payoff_of_RPPs[0] = DA_LMPs[0] * DA_commitments_at_NE[0] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[0]] * (generated_scenarios[j, 0] - DA_commitments_at_NE[0])
#            expected_payoff_of_RPPs[1] = DA_LMPs[1] * DA_commitments_at_NE[1] + LMPs_for_all_scenarios[j * self.num_of_nodes + node_of_RPPs[1]] * (generated_scenarios[j, 1] - DA_commitments_at_NE[1])
        #
        #
        return value_of_obj_fun_at_optimal_point , DA_nodal_injections , matrix_of_congestion_pttrns_for_scenarios , DC_power_flow_has_feasible_solution , LMP_average
    #
    #
    #
    def calc_mean_and_cov_mtrx_of_dividing_two_RPPs_to_smaller_ones(data_of_RPPs , r_correlation , num_of_smaller_RPPs_from_dividing_each_RPP):
        # note that this function takes the mean and  cov matrix of original RPPs (i,e, 2 RPPs) and divides them to ...
        # ... the smaller RPPs. Each RPP would be divided into "num_of_smaller_RPPs_from_dividing_each_RPP" smaller RPPs
        #
        #
        mean_of_divided_RPPs = np.zeros((1,2*num_of_smaller_RPPs_from_dividing_each_RPP)).reshape((1,2*num_of_smaller_RPPs_from_dividing_each_RPP))
        cov_mtrx_of_divided_RPPs = np.zeros((2*num_of_smaller_RPPs_from_dividing_each_RPP,2*num_of_smaller_RPPs_from_dividing_each_RPP))
        data_of_RPPs_of_divided_RPPs = np.zeros((2*num_of_smaller_RPPs_from_dividing_each_RPP,3))
        #
        #
        #
        for i in range(0,2*num_of_smaller_RPPs_from_dividing_each_RPP):
            if i < num_of_smaller_RPPs_from_dividing_each_RPP:
                mean_of_divided_RPPs[0,i] = data_of_RPPs[0,1] / num_of_smaller_RPPs_from_dividing_each_RPP
                data_of_RPPs_of_divided_RPPs[i,0] = data_of_RPPs[0,0]
                data_of_RPPs_of_divided_RPPs[i, 1] = data_of_RPPs[0, 1] / num_of_smaller_RPPs_from_dividing_each_RPP
            else:
                mean_of_divided_RPPs[0,i] = data_of_RPPs[1, 1] / num_of_smaller_RPPs_from_dividing_each_RPP
                data_of_RPPs_of_divided_RPPs[i, 0] = data_of_RPPs[1, 0]
                data_of_RPPs_of_divided_RPPs[i, 1] = data_of_RPPs[1, 1] / num_of_smaller_RPPs_from_dividing_each_RPP
        #
        #
        #
        for i in range(0,2*num_of_smaller_RPPs_from_dividing_each_RPP):
            for j in range(i,2*num_of_smaller_RPPs_from_dividing_each_RPP):
                if (i < num_of_smaller_RPPs_from_dividing_each_RPP) and (j < num_of_smaller_RPPs_from_dividing_each_RPP) and (i == j):
                    # diagonal element
                    cov_mtrx_of_divided_RPPs[i, j] = (((data_of_RPPs[0,2])**0.5) / num_of_smaller_RPPs_from_dividing_each_RPP)**2
                    data_of_RPPs_of_divided_RPPs[i, 2] = (((data_of_RPPs[0,2])**0.5) / num_of_smaller_RPPs_from_dividing_each_RPP)**2
                elif (i >= num_of_smaller_RPPs_from_dividing_each_RPP) and (j >= num_of_smaller_RPPs_from_dividing_each_RPP) and (i == j):
                    # diagonal element
                    cov_mtrx_of_divided_RPPs[i, j] = (((data_of_RPPs[1,2])**0.5) / num_of_smaller_RPPs_from_dividing_each_RPP)**2
                    data_of_RPPs_of_divided_RPPs[i, 2] = (((data_of_RPPs[1,2])**0.5) / num_of_smaller_RPPs_from_dividing_each_RPP)**2
                elif (i < num_of_smaller_RPPs_from_dividing_each_RPP) and (j < num_of_smaller_RPPs_from_dividing_each_RPP) and (i != j):
                    # non_diagonal element
                    cov_mtrx_of_divided_RPPs[i,j] = (((data_of_RPPs[0,2])**0.5) / num_of_smaller_RPPs_from_dividing_each_RPP)**2
                    cov_mtrx_of_divided_RPPs[j,i] = (((data_of_RPPs[0,2])**0.5) / num_of_smaller_RPPs_from_dividing_each_RPP)**2
                elif (i >= num_of_smaller_RPPs_from_dividing_each_RPP) and (j >= num_of_smaller_RPPs_from_dividing_each_RPP) and (i != j):
                    # non_diagonal element
                    cov_mtrx_of_divided_RPPs[i,j] = (((data_of_RPPs[1,2])**0.5) / num_of_smaller_RPPs_from_dividing_each_RPP)**2
                    cov_mtrx_of_divided_RPPs[j,i] = (((data_of_RPPs[1,2])**0.5) / num_of_smaller_RPPs_from_dividing_each_RPP)**2
                elif (i >= num_of_smaller_RPPs_from_dividing_each_RPP) and (j < num_of_smaller_RPPs_from_dividing_each_RPP) and (i != j):
                    # non_diagonal element
                    cov_mtrx_of_divided_RPPs[i,j] = r_correlation * (math.sqrt(data_of_RPPs[0,2])/num_of_smaller_RPPs_from_dividing_each_RPP) * (math.sqrt(data_of_RPPs[1,2])/num_of_smaller_RPPs_from_dividing_each_RPP)
                    cov_mtrx_of_divided_RPPs[j,i] = r_correlation * (math.sqrt(data_of_RPPs[0,2])/num_of_smaller_RPPs_from_dividing_each_RPP) * (math.sqrt(data_of_RPPs[1,2])/num_of_smaller_RPPs_from_dividing_each_RPP)
                elif (i < num_of_smaller_RPPs_from_dividing_each_RPP) and (j >= num_of_smaller_RPPs_from_dividing_each_RPP) and (i != j):
                    # non_diagonal element
                    cov_mtrx_of_divided_RPPs[i,j] = r_correlation * (math.sqrt(data_of_RPPs[0,2])/num_of_smaller_RPPs_from_dividing_each_RPP) * (math.sqrt(data_of_RPPs[1,2])/num_of_smaller_RPPs_from_dividing_each_RPP)
                    cov_mtrx_of_divided_RPPs[j,i] = r_correlation * (math.sqrt(data_of_RPPs[0,2])/num_of_smaller_RPPs_from_dividing_each_RPP) * (math.sqrt(data_of_RPPs[1,2])/num_of_smaller_RPPs_from_dividing_each_RPP)
                else:
                    pass
                #
                #
                #
        return data_of_RPPs_of_divided_RPPs , mean_of_divided_RPPs , cov_mtrx_of_divided_RPPs
    #
    #
    #







    #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#
#
#
#
#
#
#
#
5+10
