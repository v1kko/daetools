#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             aztecoo_options.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""

"""
"""

class daeAztecOptions(object):
    # Constants for solver types
    AZ_cg               = 0 # preconditioned conjugate gradient method
    AZ_gmres            = 1 # preconditioned gmres method
    AZ_cgs              = 2 # preconditioned cg squared method
    AZ_tfqmr            = 3 # preconditioned transpose-free qmr method
    AZ_bicgstab         = 4 # preconditioned stabilized bi-cg method
    AZ_slu              = 5 # super LU direct method.
    AZ_symmlq           = 6 # indefinite symmetric like symmlq
    AZ_GMRESR           = 7 # recursive GMRES (not supported)
    AZ_fixed_pt         = 8 # fixed point iteration
    AZ_analyze          = 9 # fixed point iteration
    AZ_lu               = 10 # sparse LU direct method. Also used for a
    AZ_cg_condnum       = 11
    AZ_gmres_condnum    = 12


    # Constants for scaling types
    AZ_none          = 0 # no scaling
    AZ_Jacobi        = 1 # Jacobi scaling
    AZ_BJacobi       = 2 # block Jacobi scaling
    AZ_row_sum       = 3 # point row-sum scaling
    AZ_sym_diag      = 4 # symmetric diagonal scaling
    AZ_sym_row_sum   = 5 # symmetric diagonal scaling
    AZ_equil         = 6 # equilib scaling
    AZ_sym_BJacobi   = 7


    # Constants for preconditioner types
    AZ_none            =  0 # no preconditioning. Note: also used for scaling, output, overlap options options
    AZ_Jacobi          =  1 # Jacobi preconditioning. Note: also used for scaling options
    AZ_sym_GS          =  2 # symmetric Gauss-Siedel preconditioning
    AZ_Neumann         =  3 # Neumann series polynomial preconditioning
    AZ_ls              =  4 # least-squares polynomial preconditioning
    AZ_ilu             =  6 # domain decomp with  ilu in subdomains
    AZ_bilu            =  7 # domain decomp with block ilu in subdomains
    AZ_lu              = 10 # domain decomp with   lu in subdomains
    AZ_icc             =  8 # domain decomp with incomp Choleski in domains
    AZ_ilut            =  9 # domain decomp with ilut in subdomains
    AZ_rilu            = 11 # domain decomp with rilu in subdomains
    AZ_recursive       = 12 # Recursive call to AZ_iterate()
    AZ_smoother        = 13 # Recursive call to AZ_iterate()
    AZ_dom_decomp      = 14 # Domain decomposition using subdomain solver given by options[AZ_subdomain_solve]
    AZ_multilevel      = 15 # Do multiplicative domain decomp with coarse grid (not supported).
    AZ_user_precond    = 16 #  user's preconditioning
    # Begin Aztec 2.1 mheroux mod
    AZ_bilu_ifp        = 17 # dom decomp with bilu using ifpack in subdom
    # End Aztec 2.1 mheroux mod


    # Constants for convergence types
    AZ_r0               = 0 # ||r||_2 / ||r^{(0)}||_2
    AZ_rhs              = 1 # ||r||_2 / ||b||_2
    AZ_Anorm            = 2 # ||r||_2 / ||A||_infty
    AZ_sol              = 3 # ||r||_infty/(||A||_infty ||x||_1+||b||_infty)
    AZ_weighted         = 4 # ||r||_WRMS
    AZ_expected_values  = 5 # ||r||_WRMS with weights taken as |A||x0|
    AZ_noscaled         = 6 # ||r||_2
    AZTECOO_conv_test   = 7 # Convergence test will be done via AztecOO
    AZ_inf_noscaled     = 8 # ||r||_infty


    # Constants for output types
    AZ_all             = -4 # Print out everything including matrix
    AZ_none            =  0 # Print out no results (not even warnings)
    AZ_last            = -1 # Print out final residual and warnings
    AZ_summary         = -2 # Print out summary, final residual and warnings
    AZ_warnings        = -3 # Print out only warning messages


    # Constants for matrix output
    AZ_input_form       = 0
    # Print out the matrix arrays as they appear
    # along with some additional information. The
    # idea here is to print out the information
    # that the user must supply as input to the
    # function AZ_transform()
    AZ_global_mat       = 1
    # Print out the matrix as a(i,j) where i and j
    # are the global indices. This option must
    # be invoked only after AZ_transform() as the
    # array update_index[] is used.
    # NOTE: for VBR matrices the matrix is printed
    # as a(I(i),J(j)) where I is the global block
    # row and J is the global block column and i
    # and j are the row and column indices within
    # the block.
    AZ_explicit         = 2
    # Print out the matrix as a(i,j) where i and j
    # are the local indices.
    # NOTE: for VBR matrices the matrix is printed
    # as a(I(i),J(j)) where I is the global block
    # row and J is the global block column and i
    # and j are the row and column indices within
    # the block.


    # Constants for using factorization information
    AZ_calc             = 1 # use no previous information
    AZ_recalc           = 2 # use last symbolic information
    AZ_reuse            = 3 # use a previous factorization to precondition
    AZ_sys_reuse        = 4 # use last factorization to precondition


    # Constants for domain decompositon overlap
    AZ_none            =  0 # No overlap
    AZ_diag            = -1 # Use diagonal blocks for overlapping
    AZ_full            =  1 # Use external rows   for overlapping


    # Constants to determine if overlapped values are added
    # (symmetric) or just taken from the closest processor.
    AZ_standard        = 0
    AZ_symmetric       = 1


    # Constants for GMRES orthogonalization procedure
    AZ_classic          = 0 # Does double classic
    AZ_modified         = 1 # Does single modified
    AZ_single_classic   = 2
    AZ_single_modified  = 3
    AZ_double_classic   = 4
    AZ_double_modified  = 5


    # Constants for determining rtilda (used in bicgstab, cgs, tfqmr)
    AZ_resid            = 0
    AZ_rand             = 1


    # Constants indicating reason for iterative method termination
    AZ_normal           = 0 # normal termination
    AZ_param            = 1 # requested option not implemented
    AZ_breakdown        = 2 # numerical breakdown during the computation
    AZ_maxits           = 3 # maximum iterations exceeded
    AZ_loss             = 4 # loss of precision
    AZ_ill_cond         = 5 # GMRES hessenberg is ill-conditioned

    ##########################
    # Options available
    ##########################
    AZ_solver              = 0
    AZ_scaling             = 1
    AZ_precond             = 2
    AZ_conv                = 3
    AZ_output              = 4
    AZ_pre_calc            = 5
    AZ_max_iter            = 6
    AZ_poly_ord            = 7
    AZ_overlap             = 8
    AZ_type_overlap        = 9
    AZ_kspace              = 10
    AZ_orthog              = 11
    AZ_aux_vec             = 12
    AZ_reorder             = 13
    AZ_keep_info           = 14
    AZ_recursion_level     = 15
    AZ_print_freq          = 16
    AZ_graph_fill          = 17
    AZ_subdomain_solve     = 18
    AZ_init_guess          = 19
    AZ_keep_kvecs          = 20
    AZ_apply_kvecs         = 21
    AZ_orth_kvecs          = 22
    AZ_ignore_scaling      = 23
    AZ_check_update_size   = 24
    AZ_extreme             = 25
    AZ_diagnostics         = 26

    ###########################
    # Parameters available
    ###########################
    AZ_tol                 = 0
    AZ_drop                = 1
    AZ_ilut_fill           = 2
    AZ_omega               = 3
    # Begin Aztec 2.1 mheroux mod
    AZ_rthresh             = 4
    AZ_athresh             = 5
    AZ_update_reduction    = 6
    AZ_temp                = 7
    AZ_ill_cond_thresh     = 8
    AZ_weights             = 9
    # End Aztec 2.1 mheroux mod
