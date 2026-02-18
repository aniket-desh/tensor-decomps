import numpy as np
import sys
import time
import os
import csv
from pathlib import Path
from os.path import dirname, join
import tensor_decomposition
import tensor_decomposition.tensors.synthetic_tensors as synthetic_tensors
import tensor_decomposition.tensors.real_tensors as real_tensors
import argparse
import tensor_decomposition.utils.arg_defs as arg_defs
import csv
from tensor_decomposition.utils.utils import save_decomposition_results
from tensor_decomposition.CPD.common_kernels import get_residual,get_residual_sp,compute_condition_number
from tensor_decomposition.CPD.standard_ALS import CP_DTALS_Optimizer, CP_PPALS_Optimizer

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')

def CP_ALS(tenpy,
           A,
           T,
           O,
           num_iter,
           args,
           R,
           R_app,
           r,
           csv_file=None,
           Regu=None,
           method='DT',
           res_calc_freq=1,
           tol=1e-05):


    flag_dt = True
    #print(R)
    if csv_file is not None:
        csv_writer = csv.writer(csv_file,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

    if Regu is None:
        Regu = 0

    normT = tenpy.vecnorm(T)

    time_all = 0.
    if args is None:
        optimizer = CP_DTALS_Optimizer(tenpy, T, A,args)
    else:
        optimizer_list = {
            'DT': CP_DTALS_Optimizer(tenpy, T, A,args),
            'PP': CP_PPALS_Optimizer(tenpy, T, A, args),
        }
        optimizer = optimizer_list[method]

    fitness_old = 0
    residuals = []
    for i in range(num_iter):

        if i % res_calc_freq == 0 or i == num_iter - 1 or not flag_dt:
            if args.fast_residual and i != 0:
               res = optimizer.compute_fast_residual()
            else:
                if args.sp and O is not None:
                    res = get_residual_sp(tenpy,O,T,A)
                else:
                    res = get_residual(tenpy, T, A)
            fitness = 1 - res / normT
            residuals.append(res)
            if args.calc_cond and R < 15 and tenpy.name() == 'numpy':
                cond = compute_condition_number(tenpy, A)
                if tenpy.is_master_proc():
                    print("[", i, "] Residual is", res, "fitness is: ", fitness)
                    # write to csv file
                    if csv_file is not None:
                        csv_writer.writerow([i, time_all, res, fitness, flag_dt,cond])
                        csv_file.flush()
            else:
                if tenpy.is_master_proc():
                    print("[", i, "] Residual is", res, "fitness is: ", fitness)
                    # write to csv file
                    if csv_file is not None:
                        csv_writer.writerow([i, time_all, res, fitness, flag_dt])
                        csv_file.flush()

        if res < tol:
            print('Method converged in', i, 'iterations')
            break
        t0 = time.time()
        if method == 'PP':
            A, pp_restart = optimizer.step(Regu)
            flag_dt = not pp_restart
        else:
            A = optimizer.step(Regu)
        t1 = time.time()
        tenpy.printf("[", i, "] Sweep took", t1 - t0, "seconds")

        time_all += t1 - t0
        fitness_old = fitness

    tenpy.printf(method + " method took", time_all, "seconds overall")

    if args.save_tensor:
        folderpath = join(results_dir, arg_defs.get_file_prefix(args))
        save_decomposition_results(T, A, tenpy, folderpath)

    return A, residuals