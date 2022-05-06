import numpy as np
import matplotlib.pyplot as plt

import model
import sb
import prepare_input

import evaluate

import plot_func


p=1

#Input functions
egomotion_input, visual_input = prepare_input.visual_egomotion_input(1000,6,p=p)
inp_func_egomotion = lambda t,dt : prepare_input.visual_egomotion_input_func(egomotion_input, t, presentation_time = int(100/dt), fade_time=int(10/dt))
inp_func_visual = lambda t,dt : prepare_input.visual_egomotion_input_func(visual_input, t, presentation_time = int(100/dt), fade_time=int(10/dt))

#Snapshot input functions
egomotion_input_test, visual_input_test = prepare_input.visual_egomotion_input_test(6)
inp_func_egomotion_snapshot = lambda t,dt : prepare_input.visual_egomotion_input_func(egomotion_input_test, t, presentation_time = int(100/dt), fade_time=int(10/dt))
inp_func_visual_snapshot  = lambda t,dt : prepare_input.visual_egomotion_input_func(visual_input_test, t, presentation_time = int(100/dt), fade_time=int(10/dt))


M2 = sb.Group(2, 'M2', spike_rate = 30,deltaU=0.1, T_learning_rate=3e-2)
V1 = sb.Group(12, 'V1', spike_rate = 8, deltaU = 0.05, T_learning_rate=3e-2)

E = sb.Input(2,inp_func_egomotion, 'Egomotion',inp_func_egomotion_snapshot )
V = sb.Input(6, inp_func_visual, 'Visual', inp_func_visual_snapshot)

E_M2 = sb.fw_Connection(E, M2, 'E -> M2', learning_rate = 3e-3)
E_M2.weights = np.array([[1,0], [0, 1]])*1.0

V_V1 = sb.fw_Connection(V, V1, 'V -> V1', learning_rate = 3e-3)
V_V1.weights += 0.1

V_M2 = sb.fw_Connection(V, M2, 'V -> M2', learning_rate = 3e-3)

M2_re = sb.re_Connection(M2,M2, 'M2 re M2')
V1_re = sb.re_Connection(V1,V1, 'V1 re V1')

M2_re_V1 = sb.re_Connection(M2,V1, 'M2 re V1')
V1_re_M2_re = sb.re_Connection(V1,M2, 'V1 re M2')


sb.init(experiment_name = f"mouse_experiment_full_p{p}", T = 500000, dt = 0.2, learn = True, snapshots=2, snapshot_timesteps=5000)


sb.run()

