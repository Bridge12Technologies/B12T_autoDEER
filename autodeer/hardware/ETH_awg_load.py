import numpy as np
import scipy.signal as sig
from autodeer.classes import Parameter
from autodeer.dataset import create_dataset_from_axes, create_dataset_from_sequence
from scipy.integrate import cumulative_trapezoid
from deerlab import correctphase



def uwb_load(matfile: np.ndarray, options: dict = dict(), verbosity=0,
             compress: bool = True, mask=None, sequence=None):
    """uwb_load This function is based upon the uwb_eval matlab function 
    developed by Andrin Doll. This is mostly a Python version of this software,
     for loading data generated by Doll style spectrometers.

    Parameters
    ----------
    matfile : np.ndarray
        _description_
    options : dict, optional
        _description_, by default None
    compress : bool, optional
        Compresses all the scans into a single scan.
    """

    # Extract Data
    estr = matfile[matfile['expname']]
    conf = matfile['conf'] 
    
    def extract_data(matfile):
        if "dta" in matfile.keys():
            nAvgs = matfile["nAvgs"]
            dta = [matfile["dta"]]
        elif "dta_001" in matfile.keys():
            dta = []
            for ii in range(1, estr["avgs"]+1):
                actname = 'dta_%03u' % ii 
                if actname in matfile.keys():
                    dta.append(matfile[actname])
                    # Only keep it if the average is complete, unless it is 
                    # the first
                    if sum(dta[ii-1][:, -1]) == 0 and ii > 1:
                        dta = dta[:-1]
                    elif sum(dta[ii-1][:, -1]) == 0 and ii == 1:
                        nAvgs = 0
                    else:
                        nAvgs = ii
            
            if compress:
                dta_sum = np.zeros(dta[0].shape)
                for avg in dta:
                    dta_sum += avg
                dta = [dta_sum]
            
        else:
            raise ValueError('The file has no attached data.')

        return [dta, nAvgs]
    
    dta, nAvgs = extract_data(matfile)

    if mask is not None:
        dta = [dta[0][:, mask]]

    # Eliminate Phase cycles

    if "postsigns" not in estr.keys():
        print("TODO: check uwb_eval")
        raise RuntimeError("This is not implemented yet")

    if np.isscalar(estr["postsigns"]["signs"]):
        estr["postsigns"]["signs"] = [estr["postsigns"]["signs"]]

    if type(estr["parvars"]) is dict:
        estr["parvars"] = [estr["parvars"]]

    cycled = np.array(list(map(np.size, estr["postsigns"]["signs"]))) > 1

    #  decide on wheteher the phase cycle should be eliminated or not
    if any(cycled == 0):
        elim_pcyc = 1  # if there is any non-phasecycling parvar
    else:
        elim_pcyc = 0  # if all parvars cycle phases, do not reduce them 
    
    if "elim_pcyc" in options.keys():
        elim_pcyc = options["elim_pcyc"]
    
    # Get the cycles out
    if elim_pcyc:
        for ii in range(0, len(cycled)):
            if cycled[ii]:
                if ii > 1:
                    n_skip = np.prod(estr["postsigns"]["dims"][0:ii-1])
                else:
                    n_skip = 1
                plus_idx = np.where(estr["postsigns"]["signs"][ii] == 1)[0]
                minus_idx = np.where(estr["postsigns"]["signs"][ii] == -1)[0]
                plus_mask = np.arange(0, n_skip) + (plus_idx - 1) * n_skip
                minus_mask = np.arange(0, n_skip) + (minus_idx - 1) * n_skip
                n_rep = np.size(dta[0], 1) // \
                    (n_skip * estr["postsigns"]["dims"][ii])

                for kk in range(0, len(dta)):
                    #  re-allocate
                    tmp = dta[kk]
                    dta[kk] = np.zeros((np.size(tmp, 0), n_rep*n_skip))
                    # subtract out
                    for jj in range(0, n_rep):
                        curr_offset = (jj) * n_skip
                        full_offset = (jj) * n_skip * \
                            estr["postsigns"]["dims"][ii]
                        dta[kk][:, np.arange(0, n_skip)+curr_offset] = \
                            tmp[:, plus_mask+full_offset] - \
                            tmp[:, minus_mask+full_offset]

    #  Find all the axes
    dta_x = []
    ii_dtax = 0
    relevant_parvars = []

    for ii in range(0, len(cycled)):
        estr["postsigns"]["ids"] = \
            np.array(estr["postsigns"]["ids"]).reshape(-1)
        if not (elim_pcyc and cycled[ii]):
            if type(estr["parvars"]) is list:
                vecs = estr["parvars"][estr["postsigns"]["ids"][ii]-1]["axis"]
                if np.ndim(vecs) == 1:
                    dta_x.append(vecs.astype(np.float32))

                elif np.ndim(vecs) == 2:
                    unique_axes = np.unique(vecs, axis=1)
                    dta_x.append(unique_axes.astype(np.float32))
            else:
                dta_x.append(estr["parvars"]["axis"].astype(np.float32))
            relevant_parvars.append(estr["postsigns"]["ids"][ii]-1)
            ii_dtax += 1
    
    exp_dim = ii_dtax
    if ii_dtax == 0:
        raise RuntimeError("Your dataset does not have any swept dimensions." 
                           "Uwb_eval does not work for experiments without"
                           "any parvars")
    elif ii_dtax > 2:
        raise RuntimeError("Uwb_eval cannot handle more than two dimensions")
    
    # mask dta_x if necessary
    if mask is not None:
        for ii in range(0, len(dta_x)):
            dta_x[ii] = dta_x[ii][mask]
            
    det_frq = estr["events"][estr["det_event"]-1]["det_frq"]
    det_frq_dim = 0
    fsmp = conf["std"]["dig_rate"]

    # Check for any frequency changes as well as any fixed downconversion 
    # frequencies

    if "det_frq" in options.keys():
        det_frq = options["det_frq"]
    else:

        det_frq_dim = 0
        for ii in range(0, len(relevant_parvars)):
            act_par = estr["parvars"][relevant_parvars[ii]]
            frq_change = np.zeros((len(act_par["variables"]), 1))
            for jj in range(0, len(act_par["variables"])):
                if not any('nu_' in word for word
                           in estr["parvars"][0]["variables"]):
                    frq_change[jj] = 1

        if any(frq_change):  # was there a frequency change
            # is the frequency change relevant
            if "det_frq_id" in estr["events"][estr["det_event"]-1]:
                frq_pulse = estr["events"][estr["det_event"]-1]["det_frq_id"]
                nu_init_change = 0
                nu_final_change = 0
                for jj in range(0, np.size(act_par["variables"])):
                    if ('events{' + str(frq_pulse+1) + "}.pulsedef.nu_i") in \
                            act_par["variables"][jj]:
                        nu_init_change = jj
                    if ('events{' + str(frq_pulse+1) + "}.pulsedef.nu_f") in \
                            act_par["variables"][jj]:
                        nu_final_change = jj

                if any([nu_init_change, nu_final_change]):
                    # There is a frequency change on the frequency encoding 
                    # pulse

                    # dimension that will determine the detection frequency
                    det_frq_dim = ii  

                    if "nu_final" != estr["events"][frq_pulse]['pulsedef']:
                        # rectangular pulse
                        
                        if nu_init_change == 0:
                            print(
                                "uwb_eval has no idea how to guess your "
                                "detection frequency. You were setting a "
                                "rectangular pulse in event" + str(frq_pulse) 
                                + ", but are now increasing its end frequency"
                                ". You may obtain unexpected results.")

                        # get the frequencies either from 
                        # the vectorial definition
                        if "vec" in act_par.keys():
                            det_frq = act_par["vec"][:, nu_init_change]
                        else:
                            # or from the parametric definition
                            nu_init = estr["events"][frq_pulse]["pulsedef"][
                                "nu_init"]
                            if np.isnan(act_par["strt"][nu_init_change]):
                                det_frq = np.arange(0, act_par["dim"]) * \
                                    act_par["inc"][nu_init_change] + nu_init
                            else:
                                det_frq = np.arange(0, act_par["dim"]) * \
                                    act_par["inc"][nu_init_change] + \
                                    act_par["strt"][nu_init_change]
                    
                    else:
                        #  chirp pulse, both nu_init and nu_final need to be 
                        # considered

                        nu_init = estr["events"][frq_pulse]["pulsedef"][
                            "nu_init"]
                        nu_final = estr["events"][frq_pulse]["pulsedef"][
                            "nu_final"]

                        # get the frequencies either from the 
                        # vectorial definition
                        if "vec" in act_par.keys():
                            if nu_init_change != 0:
                                nu_init = act_par["vec"][:, nu_init_change]
                            if nu_final_change != 0:
                                nu_final = act_par["vec"][:, nu_final_change]
                        else:
                            # or from the parametric definition
                            if nu_init_change != 0:
                                if np.isnan(act_par["strt"][nu_init_change]):
                                    nu_init = np.arange(0, act_par["dim"]) * \
                                        act_par["inc"][nu_init_change] + \
                                        nu_init
                                else:
                                    nu_init = np.arange(0, act_par["dim"]) * \
                                        act_par["inc"][nu_init_change] + \
                                        act_par["strt"][nu_init_change]
                            if nu_final_change != 0:
                                if np.isnan(act_par["strt"][nu_init_change]):
                                    nu_final = np.arange(0, act_par["dim"]) * \
                                        act_par["inc"][nu_final_change] + \
                                        nu_final
                                else:
                                    nu_final = np.arange(0, act_par["dim"]) * \
                                        act_par["inc"][nu_final_change] + \
                                        act_par["strt"][nu_final]
                        
                        det_frq = (nu_init + nu_final) / 2
            else:
                # we can only land here, if there was no det_frq_id given, but
                # det_frq was explicitly provided in the experiment. This could
                # be intentional, but could even so be a mistake of the user.
                print("uwb_eval has no idea how to guess your detection"
                      "frequency. You were changing some pulse frequencies, "
                      "but did not provide det_frq_id for your detection event"
                      ". I will use det_frq, as you provided it in the "
                      "experiment.")

    #  ****Check digitizer level

    parvar_pts = np.zeros(np.size(estr["parvars"]))
    for ii in range(0, len(estr["parvars"])):
        if "vec" in estr["parvars"][ii]:
            parvar_pts[ii] = np.size(estr["parvars"][ii]["vec"], 0)
        else:
            parvar_pts[ii] = estr["parvars"][ii]["dim"]
    
    # the number of data points entering one echo transient (due to reduction
    # during acquisition or reduction of phasecycles just above)
    n_traces = np.prod(parvar_pts) / np.prod(list(map(len, dta_x)))

    if "dig_max" in conf.keys():
        trace_maxlev = n_traces * estr["shots"] * conf["dig_max"]
    else:
        trace_maxlev = n_traces * estr["shots"] * 2**11
    
    #  ***** Extract all the echoes
    echopos = estr["events"][estr["det_event"]-1]["det_len"]/2 - estr[
        "events"][estr["det_event"]-1]["det_pos"] * fsmp
    dist = min([echopos, estr["events"][estr["det_event"]-1]["det_len"] - 
                echopos])
    ran_echomax = np.arange(echopos - dist, echopos + dist, dtype=np.int64)

    # get the downconversion to LO
    t_ax_full = np.arange(0, len(ran_echomax)) / fsmp
    if not np.isscalar(det_frq):
        tf = np.matmul(t_ax_full[:, None], det_frq[None, :])
        LO = np.exp(-2 * np.pi * 1j * tf)
    else:
        LO = np.exp(-2 * np.pi * 1j * t_ax_full * det_frq)

    flipback = 0

    # 1D or 2D
    if exp_dim == 2:
        dta_ev = np.zeros((len(dta_x[0]), len(dta_x[1])), dtype=np.complex128)
        dta_avg = np.zeros(
            (len(ran_echomax), len(dta_x[0]), len(dta_x[1])),
            dtype=np.complex128)
        perm_order = [0, 1, 2]
        if det_frq_dim == 1:
            perm_order = [0, 2, 1]
            flipback = 1
            dta_ev = np.transpose(dta_ev)
            dta_avg = np.transpose(dta_avg, perm_order)
    elif exp_dim == 1:
        dta_ev = np.zeros((np.size(dta_x[0],axis=0)), dtype=np.complex128)
        dta_avg = np.zeros((len(ran_echomax), np.size(dta_x[0],axis=0)),
                           dtype=np.complex128)


    dta_scans = np.zeros((len(dta),) + dta_ev.shape, dtype=np.complex128)
    
    for ii in range(0, len(dta)):
        dta_c = dta[ii][ran_echomax, :]
        dta_c = np.conj(np.apply_along_axis(sig.hilbert, 0, dta_c))

        # reshape the 2D data
        if exp_dim == 2:
            dta_resort = np.reshape(dta_c, (len(ran_echomax), len(dta_x[0]), 
                                    len(dta_x[1]))
                                    )
            dta_resort = np.transpose(dta_resort, perm_order)
        else:
            dta_resort = dta_c

        # downconvert
        dta_dc = (dta_resort.T * LO.T).T

        # refine the echo position for further evaluation, 
        # based on the first average
        if ii == 0:

            # put a symetric window to mask the expected echo position
            window = sig.windows.chebwin(np.size(dta_dc, 0), 100)
            dta_win = np.transpose(np.transpose(dta_dc) * window)
            
            # absolute part of integral, since phase is not yet determined
            absofsum = np.squeeze(np.abs(np.sum(dta_win, 0)))

            # get the strongest echo of that series
            ref_echo = np.argmax(absofsum.flatten('F'))

            # use this echo to inform about the digitizer scale

            max_amp = np.amax(dta[ii][ran_echomax, ref_echo], 0)
            dig_level = max_amp / trace_maxlev

            if "IFgain_levels" in conf["std"]:
                # Check about eventual improvemnets by changing IF levels
                possible_levels = dig_level * conf["std"]["IFgain_levels"] / \
                    conf["std"]["IFgain_levels"][estr["IFgain"]]

                possible_levels[possible_levels > 0.75] = 0
                best_lev = np.amax(possible_levels)
                best_idx = np.argmax(possible_levels)
                if (best_idx != estr["IFgain"]) & (verbosity > 0):
                    print(
                        f"You are currently using {dig_level} of the maximum"
                        f"possible level of the digitizer at an IFgain setting"
                        f"of {estr['IFgain']} \n It may be advantageous to use"
                        f"an IFgain setting of {best_idx} , where the maximum "
                        f"level will be on the order of {best_lev}.")

            # for 2D data, only a certain slice may be requested

            if "ref_echo_2D_idx" in options.keys():
                if "ref_echo_2D_dim" not in options.keys():
                    options["ref_echo_2D_dim"] = 1
                
                if flipback:
                    if options["ref_echo_2D_dim"] == 1:
                        options["ref_echo_2D_dim"] = 2
                    else:
                        options["ref_echo_2D_dim"] = 1
                
                if options["ref_echo_2D_idx"] == "end":
                    options["ref_echo_2D_idx"] = np.size(
                        absofsum, options["ref_echo_2D_dim"]-1)

                if options["ref_echo_2D_dim"] == 1:
                    ii_ref = options["ref_echo_2D_idx"] - 1
                    jj_ref = np.argmax(absofsum[ii_ref, :])
                else:
                    jj_ref = options["ref_echo_2D_idx"] - 1
                    ii_ref = np.argmax(absofsum[jj_ref, :])
                # convert the ii_ref,jj_ref to a linear index, as this is how
                # the convolution is done a few lines below

                ref_echo = np.ravel_multi_index(
                    [ii_ref, jj_ref], absofsum.shape)

            if "ref_echo" in options.keys():
                if "end" == options["ref_echo"]:
                    ref_echo = len(absofsum[:])
                else:
                    ref_echo = options["ref_echo"]
            
            # look for zerotime by crosscorrelation with 
            # echo-like window (chebwin..),
            # use conv istead of xcorr, because of the 
            # life-easy-making 'same' option
            # TODO turn this into a matched filter
            # this is where one could use a matched echo shape 

            convshape = sig.windows.chebwin(min([100, len(ran_echomax)]), 100) 

            if exp_dim == 2:
                ref_echo_unravel = np.unravel_index(ref_echo, absofsum.shape)
                e_idx = np.argmax(sig.convolve(
                    np.abs(dta_dc[:, ref_echo_unravel[0],
                    ref_echo_unravel[1]]), convshape, mode="same"))

            else:
                e_idx = np.argmax(sig.convolve(
                    np.abs(dta_dc[:, ref_echo]), convshape, mode="same"))

            # now get the final echo window, which is centered around the
            # maximum position just found

            dist = min([e_idx, np.size(dta_dc, 0)-e_idx])
            evlen = 2 * dist

            if "evlen" in options.keys():
                evlen = options["evlen"]
            if "find_echo" in options.keys():
                e_idx = np.floor(np.size(dta_dc, 0)/2)
            
            # here the final range...
            ran_echo = np.arange(e_idx-evlen/2, e_idx+evlen/2, dtype=np.int16)
            # ... and a check wether this is applicable

            if not (ran_echo[0] >= 0 and ran_echo[-1] <= np.size(dta_dc, 0)):
                raise RuntimeError(
                    f"Echo position at {e_idx} with evaluation length of "
                    f"{evlen} is not valid, since the dataset has only "
                    f"{np.size(dta_dc,0)} points.")

            # here the final time axis of the dataset
            t_ax = np.arange(-evlen/2, evlen/2) / fsmp

            # get also indices of reference echo in case of 2D data
            if absofsum.ndim == 2:
                [ii_ref, jj_ref] = np.unravel_index(
                    ref_echo, absofsum.shape, order='F')
        
        # window the echo
        if exp_dim == 2:
            dta_win = np.multiply(dta_dc[ran_echo, :, :].T, 
                                  sig.windows.chebwin(evlen, 100)).T
        else:
            dta_win = np.multiply(dta_dc[ran_echo, :].T, 
                                  sig.windows.chebwin(evlen, 100)).T

        # get all the phases and use reference echo for normalization
        dta_ang = np.angle(np.sum(dta_win, 0))

        # for frequency changes, we phase each frequency

        if det_frq_dim != 0:
            if exp_dim == 2:
                corr_phase = dta_ang[..., jj_ref]
            else:
                corr_phase = dta_ang
        else:
            corr_phase = dta_ang[ref_echo]
        
        # check if a fixed phase was provided
        if "corr_phase" in options.keys():
            corr_phase = options["corr_phase"]
        
        # check if any datapoint needs to be phased individually
        if "phase_all" in options.keys() and options["phase_all"] == 1:
            corr_phase = dta_ang
        
        bfunc = lambda x: x * np.exp(-1j * corr_phase)
        # dta_pha = np.multiply(dta_win, np.exp(-1j * corr_phase))
        
        dta_pha = np.apply_along_axis(bfunc, 1, dta_win)
        
        dta_this_scan = np.squeeze(np.sum(dta_pha, 0)) / \
            sum(sig.windows.chebwin(evlen, 100))
        dta_ev = dta_ev + dta_this_scan

        dta_scans[ii, :] = dta_this_scan  # This will not work for 2D
        
        if exp_dim == 2:
            dta_avg[0:evlen, :, :] = dta_avg[0:evlen, :, :] + \
                np.apply_along_axis(bfunc, 1, dta_win)
#                np.multiply(dta_dc[ran_echo, :, :], np.exp(-1j * corr_phase))
        else:
            dta_avg[0:evlen, :] = dta_avg[0:evlen, :] + \
                np.multiply(dta_dc[ran_echo, :], np.exp(-1j * corr_phase))

    dta_avg = dta_avg[0:evlen, ...]
    # keyboard
    # flip back 2D Data
    if flipback:
        dta_avg = np.transpose(dta_avg, perm_order)
        dta_ev = np.transpose(dta_ev)

    if sequence is None:
        params = {'nAvgs': nAvgs, 'LO': estr['LO']+1.5, 'B': estr['B'], 
                  'reptime': estr['reptime'], 'shots': estr['shots']}
        axis = [];
        for i in range(exp_dim):
            if (dta_x[i].ndim == 1) or (dta_x[i].shape[1] == 1):
                axis.append(dta_x[i])
            else:
                axis.append(dta_x[i][:, 0])
        output = create_dataset_from_axes(dta_ev, axis, params)
    else:
        params = {'nAvgs': nAvgs}
        output = create_dataset_from_sequence(dta_ev, sequence,params)
    

    return output


# --------------------------------------------------------------------------- 
# uwb_eval rewritten to use a matched filter
# ---------------------------------------------------------------------------

def uwb_eval_match(matfile, sequence=None, scans=None, mask=None,filter_pulse=None,verbosity=0, **kwargs):
    # imports Andrin Doll AWG datafiles using a matched filter

    estr = matfile[matfile['expname']]
    conf = matfile['conf'] 

    def extract_data(matfile,scans):
        if "dta" in matfile.keys():
            nAvgs = matfile["nAvgs"]
            dta = [matfile["dta"]]
        elif "dta_001" in matfile.keys():
            dta = []
            if scans is None:
                for ii in range(1, estr["avgs"]+1):
                    actname = 'dta_%03u' % ii 
                    if actname in matfile.keys():
                        dta.append(matfile[actname])
                        # Only keep it if the average is complete, unless it is 
                        # the first
                        if sum(dta[ii-1][:, -1]) == 0 and ii > 1:
                            dta = dta[:-1]
                        elif sum(dta[ii-1][:, -1]) == 0 and ii == 1:
                            nAvgs = 0
                        else:
                            nAvgs = ii
            else:
                for i,ii in enumerate(scans):
                    actname = 'dta_%03u' % ii 
                    scans = 0
                    if actname in matfile.keys():
                        dta.append(matfile[actname])
                        # Only keep it if the average is complete, unless it is 
                        # the first
                        if sum(dta[i-1][:, -1]) == 0 and ii > 1:
                            dta = dta[:-1]
                            nAvgs = len(dta)
                        elif sum(dta[i-1][:, -1]) == 0 and ii == 1:
                            nAvgs = 0
                        else:
                            nAvgs = len(dta)
                        

            
            dta_sum = np.zeros(dta[0].shape)
            for avg in dta:
                dta_sum += avg
            dta = [dta_sum]
            
        else:
            raise ValueError('The file has no attached data.')

        return [dta, nAvgs]
    
    dta, nAvgs = extract_data(matfile,scans)

    if mask is not None:
        dta = [dta[0][:, mask]]

    # Eliminate Phase cycles

    if "postsigns" not in estr.keys():
        print("TODO: check uwb_eval")
        raise RuntimeError("This is not implemented yet")

    if np.isscalar(estr["postsigns"]["signs"]):
        estr["postsigns"]["signs"] = [estr["postsigns"]["signs"]]

    if type(estr["parvars"]) is dict:
        estr["parvars"] = [estr["parvars"]]

    cycled = np.array(list(map(np.size, estr["postsigns"]["signs"]))) > 1

    #  decide on wheteher the phase cycle should be eliminated or not
    if any(cycled == 0):
        elim_pcyc = 1  # if there is any non-phasecycling parvar
    else:
        elim_pcyc = 0  # if all parvars cycle phases, do not reduce them 
        
    # Get the cycles out
    if elim_pcyc:
        for ii in range(0, len(cycled)):
            if cycled[ii]:
                if ii > 1:
                    n_skip = np.prod(estr["postsigns"]["dims"][0:ii-1])
                else:
                    n_skip = 1
                plus_idx = np.where(estr["postsigns"]["signs"][ii] == 1)[0]
                minus_idx = np.where(estr["postsigns"]["signs"][ii] == -1)[0]
                plus_mask = np.arange(0, n_skip) + (plus_idx - 1) * n_skip
                minus_mask = np.arange(0, n_skip) + (minus_idx - 1) * n_skip
                n_rep = np.size(dta[0], 1) // \
                    (n_skip * estr["postsigns"]["dims"][ii])

                for kk in range(0, len(dta)):
                    #  re-allocate
                    tmp = dta[kk]
                    dta[kk] = np.zeros((np.size(tmp, 0), n_rep*n_skip))
                    # subtract out
                    for jj in range(0, n_rep):
                        curr_offset = (jj) * n_skip
                        full_offset = (jj) * n_skip * \
                            estr["postsigns"]["dims"][ii]
                        dta[kk][:, np.arange(0, n_skip)+curr_offset] = \
                            tmp[:, plus_mask+full_offset] - \
                            tmp[:, minus_mask+full_offset]


    #  Find all the axes
    dta_x = []
    ii_dtax = 0
    relevant_parvars = []

    for ii in range(0, len(cycled)):
        estr["postsigns"]["ids"] = \
            np.array(estr["postsigns"]["ids"]).reshape(-1)
        if not (elim_pcyc and cycled[ii]):
            if type(estr["parvars"]) is list:
                vecs = estr["parvars"][estr["postsigns"]["ids"][ii]-1]["axis"]
                if np.ndim(vecs) == 1:
                    dta_x.append(vecs.astype(np.float32))

                elif np.ndim(vecs) == 2:
                    unique_axes = np.unique(vecs, axis=1)
                    dta_x.append(unique_axes.astype(np.float32))
            else:
                dta_x.append(estr["parvars"]["axis"].astype(np.float32))
            relevant_parvars.append(estr["postsigns"]["ids"][ii]-1)
            ii_dtax += 1
    
    exp_dim = ii_dtax
    if ii_dtax == 0:
        raise RuntimeError("Your dataset does not have any swept dimensions." 
                           "Uwb_eval does not work for experiments without"
                           "any parvars")
    elif ii_dtax > 2:
        raise RuntimeError("Uwb_eval cannot handle more than two dimensions")
    
    det_frq = estr["events"][estr["det_event"]-1]["det_frq"]
    det_frq_dim = 0
    fsmp = conf["std"]["dig_rate"]

    for ii in range(0, len(relevant_parvars)):
        act_par = estr["parvars"][relevant_parvars[ii]]
        frq_change = np.zeros((len(act_par["variables"]), 1))
        for jj in range(0, len(act_par["variables"])):
            if not any('nu_' in word for word
                        in estr["parvars"][0]["variables"]):
                frq_change[jj] = 1

        if any(frq_change):  # was there a frequency change
            # is the frequency change relevant
            if "det_frq_id" in estr["events"][estr["det_event"]-1]:
                frq_pulse = estr["events"][estr["det_event"]-1]["det_frq_id"]
                nu_init_change = 0
                nu_final_change = 0
                for jj in range(0, np.size(act_par["variables"])):
                    if ('events{' + str(frq_pulse+1) + "}.pulsedef.nu_i") in \
                            act_par["variables"][jj]:
                        nu_init_change = jj
                    if ('events{' + str(frq_pulse+1) + "}.pulsedef.nu_f") in \
                            act_par["variables"][jj]:
                        nu_final_change = jj

                if any([nu_init_change, nu_final_change]):
                    # There is a frequency change on the frequency encoding 
                    # pulse

                    # dimension that will determine the detection frequency
                    det_frq_dim = ii  

                    if "nu_final" != estr["events"][frq_pulse]['pulsedef']:
                        # rectangular pulse
                        
                        if nu_init_change == 0:
                            print(
                                "uwb_eval has no idea how to guess your "
                                "detection frequency. You were setting a "
                                "rectangular pulse in event" + str(frq_pulse) 
                                + ", but are now increasing its end frequency"
                                ". You may obtain unexpected results.")

                        # get the frequencies either from 
                        # the vectorial definition
                        if "vec" in act_par.keys():
                            det_frq = act_par["vec"][:, nu_init_change]
                        else:
                            # or from the parametric definition
                            nu_init = estr["events"][frq_pulse]["pulsedef"][
                                "nu_init"]
                            if np.isnan(act_par["strt"][nu_init_change]):
                                det_frq = np.arange(0, act_par["dim"]) * \
                                    act_par["inc"][nu_init_change] + nu_init
                            else:
                                det_frq = np.arange(0, act_par["dim"]) * \
                                    act_par["inc"][nu_init_change] + \
                                    act_par["strt"][nu_init_change]
                    
                    else:
                        #  chirp pulse, both nu_init and nu_final need to be 
                        # considered

                        nu_init = estr["events"][frq_pulse]["pulsedef"][
                            "nu_init"]
                        nu_final = estr["events"][frq_pulse]["pulsedef"][
                            "nu_final"]

                        # get the frequencies either from the 
                        # vectorial definition
                        if "vec" in act_par.keys():
                            if nu_init_change != 0:
                                nu_init = act_par["vec"][:, nu_init_change]
                            if nu_final_change != 0:
                                nu_final = act_par["vec"][:, nu_final_change]
                        else:
                            # or from the parametric definition
                            if nu_init_change != 0:
                                if np.isnan(act_par["strt"][nu_init_change]):
                                    nu_init = np.arange(0, act_par["dim"]) * \
                                        act_par["inc"][nu_init_change] + \
                                        nu_init
                                else:
                                    nu_init = np.arange(0, act_par["dim"]) * \
                                        act_par["inc"][nu_init_change] + \
                                        act_par["strt"][nu_init_change]
                            if nu_final_change != 0:
                                if np.isnan(act_par["strt"][nu_init_change]):
                                    nu_final = np.arange(0, act_par["dim"]) * \
                                        act_par["inc"][nu_final_change] + \
                                        nu_final
                                else:
                                    nu_final = np.arange(0, act_par["dim"]) * \
                                        act_par["inc"][nu_final_change] + \
                                        act_par["strt"][nu_final]
                        
                        det_frq = (nu_init + nu_final) / 2

    parvar_pts = np.zeros(np.size(estr["parvars"]))
    for ii in range(0, len(estr["parvars"])):
        if "vec" in estr["parvars"][ii]:
            parvar_pts[ii] = np.size(estr["parvars"][ii]["vec"], 0)
        else:
            parvar_pts[ii] = estr["parvars"][ii]["dim"]
    
    # the number of data points entering one echo transient (due to reduction
    # during acquisition or reduction of phasecycles just above)
    n_traces = np.prod(parvar_pts) / np.prod(list(map(len, dta_x)))

    if "dig_max" in conf.keys():
        trace_maxlev = n_traces * estr["shots"] * conf["dig_max"]
    else:
        trace_maxlev = n_traces * estr["shots"] * 2**11
    
    max_amp = np.amax(dta[0],0)
    dig_level = np.amax(max_amp)

    if "IFgain_levels" in conf["std"]:
                # Check about eventual improvemnets by changing IF levels
                possible_levels = dig_level * conf["std"]["IFgain_levels"] / \
                    conf["std"]["IFgain_levels"][estr["IFgain"]]

                possible_levels[possible_levels > 0.75] = 0
                best_lev = np.amax(possible_levels)
                best_idx = np.argmax(possible_levels)
                if (best_idx != estr["IFgain"]) & (verbosity > 0):
                    print(
                        f"You are currently using {dig_level} of the maximum"
                        f"possible level of the digitizer at an IFgain setting"
                        f"of {estr['IFgain']} \n It may be advantageous to use"
                        f"an IFgain setting of {best_idx} , where the maximum "
                        f"level will be on the order of {best_lev}.")
                    
    det_frqs_perc = calc_percieved_freq(fsmp,det_frq)
    
    # Create the matched filter
    if filter_pulse is None:
        # If no filter pulse is given, use a rectangular pulse matching the length of the longest fixed pulse
        echo_len = dta[0].shape[0]
        dt = 1/fsmp
        t = np.linspace(0,echo_len//2,echo_len,endpoint=False)
        tp = find_max_pulse_length(estr)
        tp *= fsmp
        AM,FM = np.zeros((2,echo_len))
        AM[echo_len//2-tp//2:echo_len//2+tp//2] = 1
        FM[echo_len//2-tp//2:echo_len//2+tp//2] = 0
        FM_arg = 2*np.pi*cumulative_trapezoid(FM, initial=0) * dt
        complex = AM * (np.cos(FM_arg) +1j* np.sin(FM_arg))

    else:
        complex = filter_pulse


    # Apply the matched filter
    dta_c = np.apply_along_axis(sig.hilbert, 0, dta[0])

    echo_len = dta_c.shape[0]
    if exp_dim ==2:
        dims = [len(dta_x[0]),len(dta_x[1])]
        
    else:
        dims = [len(dta_x[0])]
        
    dta_c=dta_c.reshape(echo_len,*dims,order='F')
    if isinstance(det_frqs_perc,np.ndarray) and (len(det_frq) > 1):
        n_det_frqs = len(det_frqs_perc)
        dta_filt_dc = np.array([np.apply_along_axis(match_filter_dc, 0, np.take(dta_c,i,det_frq_dim+1),t, complex, det_frqs_perc[i]) for i in range(n_det_frqs)])
        dta_filt_dc = np.moveaxis(dta_filt_dc,0,det_frq_dim+1)
    else:
        dta_filt_dc = np.apply_along_axis(match_filter_dc, 0, dta_c,t, complex, det_frqs_perc)
    peak_echo_idx = np.unravel_index(np.argmax(np.abs(dta_filt_dc).max(axis=0)),dims)
    echo_pos = np.argmax(np.abs(dta_filt_dc[:,*peak_echo_idx]))
    dta_ev = dta_filt_dc[echo_pos,:]
    # dta_ev.reshape(30,40)
    if sequence is None:
        params = {'nAvgs': nAvgs, 'LO': estr['LO']+1.5, 'B': estr['B'], 
                  'reptime': estr['reptime'], 'shots': estr['shots']}
        axis = [];
        for i in range(exp_dim):
            if (dta_x[i].ndim == 1) or (dta_x[i].shape[1] == 1):
                axis.append(dta_x[i])
            else:
                axis.append(dta_x[i][:, 0])
        output = create_dataset_from_axes(dta_ev, axis, params)
    else:
        params = {'nAvgs': nAvgs}
        output = create_dataset_from_sequence(dta_ev, sequence,params)

    return output

def find_max_pulse_length(estr):
    n_pulses = len(estr["events"])
    tps = []
    for i in range(n_pulses):
        if 'pulsedef' in estr["events"][i].keys():
            tps.append(estr["events"][i]['pulsedef']['tp'])
    return max(tps)

def digitally_upconvert(t,complex,fc):
    upconvert = np.exp(1j*2*np.pi*fc*t)
    return complex * upconvert

def calc_percieved_freq(sampling_freq,fc):
    return np.abs(fc - sampling_freq * np.around(sampling_freq/2))

def match_filter_dc(pulse,t, win, sampling_freq):
    win_fc = digitally_upconvert(t,win,sampling_freq)
    filtered = sig.convolve(pulse,win_fc,mode='same')
    filtered_dc = digitally_upconvert(t,filtered,-sampling_freq)
    return filtered_dc

