from runtime import ASSETS, BUILD
import sys,os,json,site

from pathlib import Path

import numpy as np
from double_pendulum.controller.acados_mpc.acados_mpc import AcadosMpc
from double_pendulum.model.model_parameters import model_parameters
ROOT=Path(__file__).resolve().parent
def create(limit=6.,horizon=.5):
 pars=model_parameters(filepath=str(ASSETS / "model_parameters.yml"));pars.set_torque_limit([0,limit]);pars.set_motor_inertia(0);pars.set_damping([0,0]);pars.set_cfric([0,0])
 model_name=f'acrobot_L{limit}_H{horizon}'.replace('.','p');c=AcadosMpc(model_pars=pars,generate_code_filename=model_name);c.set_start([0,0,0,0]);c.set_goal([np.pi,0,0,0]);c.set_parameters(N_horizon=round(horizon/.025),prediction_horizon=horizon,Nlp_max_iter=40,max_solve_time=.01,solver_type='SQP_RTI',wrap_angle=False,warm_start=True,fallback_on_solver_fail=True,nonuniform_grid=False,cheating_on_inactive_joint=False,mpc_cycle_dt=.0025,outer_cycle_dt=.002,qp_solver_tolerance=.001,qp_solver='PARTIAL_CONDENSING_HPIPM',hpipm_mode='ROBUST');c.set_velocity_constraints(v_max=30,v_final=30);c.set_cost_parameters(Q_mat=2*np.diag([100,100,10,10]),Qf_mat=2*np.diag([10000,10000,100,100]),R_mat=2*np.diag([1e-6,1e-6]))
 work=BUILD/model_name;work.mkdir(parents=True,exist_ok=True);os.chdir(work);c.init();return c

