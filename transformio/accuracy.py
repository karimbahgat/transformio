
import numpy as np
import math

# residuals

def distances(obsx, obsy, predx, predy, metric='euclidean'):
    # to arrays
    obsx = np.array(obsx)
    obsy = np.array(obsy)
    predx = np.array(predx)
    predy = np.array(predy)

    # calc distances
    if metric == 'euclidean':
        # euclidean distances
        resids = np.sqrt((predx - obsx)**2 + (predy - obsy)**2)
    elif metric == 'geodesic':
        # geodesic is geodesic distance between lat-lon coordinates
        def haversine(lon1, lat1, lon2, lat2):
            """
            Calculate the great circle distance between two points 
            on the earth (specified in decimal degrees)
            """
            # https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
            # convert decimal degrees to radians 
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            # haversine formula 
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            a = (np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a)) 
            km = 6367 * c
            return km
        resids = haversine(predx, predy, obsx, obsy)
    else:
        raise ValueError(metric)

    return resids

def residuals(transform, inpoints, outpoints, distance='euclidean'):
    # fit
    inx,iny = zip(*inpoints)
    outx,outy = zip(*outpoints)
    transform.fit(inx, iny, outx, outy)
    
    # residual is difference bw model prediction and observed output
    inx,iny = zip(*inpoints)
    predx,predy = transform.predict(inx,iny)

    # determine observed reference points
    obsx,obsy = zip(*outpoints) # outpoints are the ones being compared

    # calc distances
    resids = distances(obsx, obsy, predx, predy, distance)

    return list(zip(predx,predy)), resids

def loo_residuals(transform, inpoints, outpoints, distance='euclidean'):
    # leave-one-out bootstrap method (out of sample)
    # residual is difference bw predicted point when refitting the model without each point
    predpoints = []
    for inpoint,outpoint in zip(inpoints, outpoints):
        # remove gcp and reestimate transform
        _inpoints = list(inpoints)
        _inpoints.remove(inpoint)
        _outpoints = list(outpoints)
        _outpoints.remove(outpoint)

        inx,iny = zip(*_inpoints)
        outx,outy = zip(*_outpoints)
        transform.fit(inx, iny, outx, outy)

        # calc err bw observed outpoint and predicted outpoint (in sample)
        inx,iny = inpoint
        predx,predy = transform.predict([inx], [iny])
        predpoints.append((predx[0], predy[0]))

    # predicted points
    predx,predy = zip(*predpoints)

    # determine observed reference points
    obsx,obsy = zip(*outpoints) # outpoints are the ones being compared

    # calc distances
    resids = distances(obsx, obsy, predx, predy, distance)

    return predpoints, resids


# accuracy
# MAYBE: remove this in favor of a two-liner residual+metric... 
def model_accuracy(trans, inpoints, outpoints, leave_one_out=False, distance='euclidean', metric='rmse'):
    # convenience function
    resfunc = loo_residuals if leave_one_out else residuals
    predicted,resids = resfunc(trans, inpoints, outpoints, distance) 

    accfunc = {'rmse':RMSE, 'mae':MAE, 'max':MAX}[metric.lower()]
    err = accfunc(resids)

    return predicted, resids, err


# auto refinement

def drop_worst_model(trans, inpoints, outpoints, leave_one_out=False, distance='euclidean', metric='rmse'):
    inpoints = list(inpoints)
    outpoints = list(outpoints)
    trans = trans.copy()
    
    # remove one at a time
    errs = []
    for inp,outp in zip(inpoints,outpoints):
        _inpoints = list(inpoints)
        _inpoints.remove(inp)
        _outpoints = list(outpoints)
        _outpoints.remove(outp)

        predicted,resids,err = model_accuracy(trans, _inpoints, _outpoints, leave_one_out=leave_one_out, distance=distance, metric=metric)
        
        errs.append((inp,outp,predicted,resids,err))

    # drop the gcp leading to lowest error if dropped
    inp,outp,predicted,resids,err = sorted(errs, key=lambda i_o_p_r_e: i_o_p_r_e[-1])[0] 
    inpoints.remove(inp)
    outpoints.remove(outp)

    # refit model with remaining points before returning
    inx,iny = zip(*inpoints)
    outx,outy = zip(*outpoints)
    trans.fit(inx, iny, outx, outy)
    
    return trans, inpoints, outpoints, predicted, resids, err

def auto_drop_models(trans, inpoints, outpoints, improvement_ratio=0.10, minpoints=None, leave_one_out=False, distance='euclidean', metric='rmse', verbose=False):
    _inpoints = list(inpoints)
    _outpoints = list(outpoints)
    trans = trans.copy()
    seq = []

    # determine minimum points
    minpoints = minpoints or trans.minpoints
    minpoints = max(minpoints, trans.minpoints)

    # initial error
    predicted,resids,err = model_accuracy(trans, _inpoints, _outpoints,
                                        leave_one_out, distance, metric)
    seq.append((trans, _inpoints, _outpoints, predicted, resids, err))
    if verbose:
        print(trans)
        print('init error',err)

    # auto refine improvement threshold or minpoints
    while len(_inpoints) > minpoints: #for _ in range(len(inpoints)-trans.minpoints):
        if verbose:
            print(len(_inpoints))
        _trans,_inpoints,_outpoints,_predicted,_resids,_err = drop_worst_model(trans, _inpoints, _outpoints,
                                                                                leave_one_out, distance, metric)
        if verbose:
            print('new error',_err)
        
        _preverr = seq[-1][-1]
        impr = (_err-_preverr)/float(_preverr)
        if impr > -improvement_ratio:
            # no longer improving, exit
            break

        seq.append((_trans,_inpoints,_outpoints,_predicted,_resids,_err))

    # refit model
    _trans,_inpoints,_outpoints,_predicted,_resids,_err = seq[-1]
    inx,iny = zip(*_inpoints)
    outx,outy = zip(*_outpoints)
    _trans.fit(inx, iny, outx, outy)

    # should we return list of errors and points?
    # ...

    return _trans, _inpoints, _outpoints, _predicted, _resids, _err

def auto_choose_model(inpoints, outpoints, transforms, refine_outliers=True, **kwargs):
    # compare and choose optimal among a set of transforms
    inpoints = list(inpoints)
    outpoints = list(outpoints)

    results = []
    for trans in transforms:
        #print trans
        # note that leave_one_out is hardcoded in order to compare across models
        if refine_outliers:
            # Drop outliers based on LOO resids (slower)
            res = auto_drop_models(trans, inpoints, outpoints, leave_one_out=True, **kwargs)
            # Drop outliers based on normal resids (much faster)
            #trans, inpoints, outpoints, err, resids = auto_drop_models(trans, inpoints, outpoints, leave_one_out=False, **kwargs)
            #err,resids = model_accuracy(trans, inpoints, outpoints, leave_one_out=True, **kwargs)
            #res = trans, inpoints, outpoints, err, resids
        else:
            predicted,resids,err = model_accuracy(trans, inpoints, outpoints, leave_one_out=True, **kwargs)
            res = trans, inpoints, outpoints, predicted, resids, err
        results.append(res)

    best = sorted(results, key=lambda res: res[-1])
    trans, inpoints, outpoints, predicted, resids, err = best[0]
    return trans, inpoints, outpoints, predicted, resids, err


##def drop_worst_residual():
##    raise NotImplemented()


##def drop_outliers(transform, inpoints, outpoints, max_residual=None, geodesic=False):
##    inx,iny = zip(*inpoints)
##    outx,outy = zip(*outpoints)
##    predx,predy = transform.predict(inx,iny)
##    if geodesic:
##        resids = residuals(outx,outy,predx,predy,'geodesic')
##    else:
##        resids = residuals(outx,outy,predx,predy)
##
##    # calculate residual stats
##    def diststats(vals):
##        mean = sum(vals) / float(len(vals))
##        sqdev = [(v-mean)**2 for v in vals]
##        stdev = math.sqrt(sum(sqdev)/float(len(vals)))
##        return mean,stdev
##    mean,stdev = diststats(resids)
##    
##    # drop bad points with bad residuals
##    inpoints_new = []
##    outpoints_new = []
##    for i in range(len(inpoints)):
##        resid = resids[i]
##        if mean-stdev*2 < resid < mean+stdev*2:
##            if max_residual and resid > max_residual:
##                continue
##            inpoints_new.append(inpoints[i])
##            outpoints_new.append(outpoints[i])
##            
##    return inpoints_new, outpoints_new


# metrics

def RMSE(residuals):
    residuals = np.array(residuals)
    invalid = np.isnan(residuals) | np.isinf(residuals)
    residuals = residuals[~invalid]
    return math.sqrt( (residuals**2).sum() / float(residuals.shape[0]) )

def MAE(residuals):
    residuals = np.array(residuals)
    invalid = np.isnan(residuals) | np.isinf(residuals)
    residuals = residuals[~invalid]
    return abs(residuals).sum() / float(residuals.shape[0])

def MAX(residuals):
    residuals = np.array(residuals)
    invalid = np.isnan(residuals) | np.isinf(residuals)
    residuals = residuals[~invalid]
    return abs(residuals).max()









    
