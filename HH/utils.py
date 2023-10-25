import inspect
import numpy as np
import os
import pickle


def obs_params(reduced_model=False):
    """Parameters for x_o
    
    Parameters
    ----------
    reduced_model : bool
        If True, outputs two parameters
    Returns
    -------
    true_params : array
    labels_params : list of str
    """
    
    if reduced_model:
        true_params = np.array([50., 5.])
    else:
        true_params = np.array([50., 5., 0.1, 0.07, 6e2, 60., 0.1, 70.])
        
    labels_params = ['g_Na', 'g_K', 'g_leak', 'g_M',
                          't_max', '-V_T', 'noise','-E_leak']
    labels_params = labels_params[0:len(true_params)]

    return true_params, labels_params

def syn_current(duration=120, dt=0.01, t_on = 10, step_current=True,
                curr_level = 5e-4, seed=None):
    t_offset = 0.
    duration = duration
    t_off = duration - t_on
    t = np.arange(0, duration+dt, dt)

    # external current
    A_soma = np.pi*((70.*1e-4)**2)  # cm2
    I = np.zeros_like(t)
    I[int(np.round(t_on/dt)):int(np.round(t_off/dt))] = curr_level/A_soma # muA/cm2
    if step_current is False:
        rng_input = np.random.RandomState(seed=seed)

        times = np.linspace(0.0, duration, int(duration / dt) + 1)
        I_new = I*1.
        tau_n = 3.
        nois_mn = 0.2*I
        nois_fact = 2*I*np.sqrt(tau_n)
        for i in range(1, times.shape[0]):
            I_new[i] = I_new[i-1] + dt*(-I_new[i-1] + nois_mn[i-1] +
                        nois_fact[i-1]*rng_input.normal(0)/(dt**0.5))/tau_n
        I = I_new*1.

    return I, t_on, t_off, dt

def param_transform(prior_log, x):
    if prior_log:
        return np.log(x)
    else:
        return x

def param_invtransform(prior_log, x):
    if prior_log:
        return np.exp(x)
    else:
        return 