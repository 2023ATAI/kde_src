import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import unbiased_rmse, _rmse, _bias,  r2_score,GetNSE,GetKGE
from config import get_args


def lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :, :int(x.shape[2] / 2)] = x[:, :, int(x.shape[2] / 2):]
    x_new[:, :, int(x.shape[2] / 2):] = x[:, :, :int(x.shape[2] / 2)]
    return x_new


def postprocess(cfg):
    PATH = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/'
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    mask = np.load(PATH + file_name_mask)

    # ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['LSTM','BILSTM','GRU','KDE_LSTM','KDE_BILSTM','KDE_GRU']:
        out_path_lstm = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg['workname'] + '/' + cfg['modelname'] + '/focast_time ' + str(cfg['forcast_time']) + '/'
        y_pred_lstm = np.load(out_path_lstm + '_predictions.npy')
        y_test_lstm = np.load(out_path_lstm + 'observations.npy')
        print(y_pred_lstm.shape, y_test_lstm.shape)
        # get shape
        nt, nlat, nlon = y_test_lstm.shape
        # cal perf
        r2_lstm = np.full((nlat, nlon), np.nan)
        GetKGE_lstm = np.full((nlat, nlon), np.nan)
        GetPCC_lstm = np.full((nlat, nlon), np.nan)
        GetNSE_lstm = np.full((nlat, nlon), np.nan)
        urmse_lstm = np.full((nlat, nlon), np.nan)
        r_lstm = np.full((nlat, nlon), np.nan)
        rmse_lstm = np.full((nlat, nlon), np.nan)
        bias_lstm = np.full((nlat, nlon), np.nan)
        rv_lstm = np.full((nlat, nlon), np.nan)
        fhv_lstm = np.full((nlat, nlon), np.nan)
        flv_lstm = np.full((nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_lstm[:, i, j]).any()):
                    #print(' y_pred_lstm[:, i, j] is', y_pred_lstm[:, i, j])
                    #print(' y_test_lstm[:, i, j] is', y_test_lstm[:, i, j])
                    urmse_lstm[i, j] = unbiased_rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    GetKGE_lstm[i, j] = GetKGE(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    #GetPCC_lstm[i, j] = GetPCC(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                   # GetNSE_lstm[i, j] = GetNSE(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    r2_lstm[i, j] = r2_score(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    #rv_lstm[i, j] = _rv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                   # fhv_lstm[i, j] = _fhv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    #flv_lstm[i, j] = _flv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    # print(' r2_lstm[i, j] is', r2_lstm[i, j])
                    #r_lstm[i, j] = np.corrcoef(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])[0, 1]
                    #rmse_lstm[i, j] = _rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    bias_lstm[i, j] = _bias(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
        np.save(out_path_lstm + 'r2_' + cfg['modelname'] + '.npy', r2_lstm)
        np.save(out_path_lstm + 'KGE_' + cfg['modelname'] + '.npy', GetKGE_lstm)
        np.save(out_path_lstm + 'PCC_' + cfg['modelname'] + '.npy', GetPCC_lstm)
        np.save(out_path_lstm + 'NSE_' + cfg['modelname'] + '.npy', GetNSE_lstm)
        np.save(out_path_lstm + 'rv_' + cfg['modelname'] + '.npy', rv_lstm)
        np.save(out_path_lstm + 'fhv_' + cfg['modelname'] + '.npy', fhv_lstm)
        np.save(out_path_lstm + 'flv_' + cfg['modelname'] + '.npy', flv_lstm)
        np.save(out_path_lstm + 'r_' + cfg['modelname'] + '.npy', r_lstm)
        np.save(out_path_lstm + 'rmse_' + cfg['modelname'] + '.npy', rmse_lstm)
        np.save(out_path_lstm + 'bias_' + cfg['modelname'] + '.npy', bias_lstm)
        np.save(out_path_lstm + 'urmse_' + cfg['modelname'] + '.npy', urmse_lstm)
        print('postprocess ove, please go on')
    # ------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = get_args()
    postprocess(cfg)
    #for modelname in ['LSTM','BILSTM','GRU','KDE_LSTM','KDE_BILSTM','KDE_GRU']:
     #   print('processing :'+modelname)
      #  PATH = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/'
       # file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
        #mask = np.load(PATH + file_name_mask)
    
        # ------------------------------------------------------------------------------------------------------------------------------
        #if modelname in ['LSTM','BILSTM','GRU','KDE_LSTM','KDE_BILSTM','KDE_GRU']:
         #   out_path_lstm = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg['workname'] + '/' + str(modelname) + '/focast_time ' + str(cfg['forcast_time']) + '/'
            
         #   y_pred_lstm = np.load(out_path_lstm + '_predictions.npy')
          #  y_test_lstm = np.load(out_path_lstm + 'observations.npy')
           # print(y_pred_lstm.shape, y_test_lstm.shape)
            # get shape
            #nt, nlat, nlon = y_test_lstm.shape
            # cal perf
            #r2_lstm = np.full((nlat, nlon), np.nan)
            #GetKGE_lstm = np.full((nlat, nlon), np.nan)
            #GetPCC_lstm = np.full((nlat, nlon), np.nan)
            #GetNSE_lstm = np.full((nlat, nlon), np.nan)
            #urmse_lstm = np.full((nlat, nlon), np.nan)
            #r_lstm = np.full((nlat, nlon), np.nan)
            #rmse_lstm = np.full((nlat, nlon), np.nan)
            #bias_lstm = np.full((nlat, nlon), np.nan)
            #rv_lstm = np.full((nlat, nlon), np.nan)
            #fhv_lstm = np.full((nlat, nlon), np.nan)
            #flv_lstm = np.full((nlat, nlon), np.nan)
            #for i in range(nlat):
             #   for j in range(nlon):
              #      if not (np.isnan(y_test_lstm[:, i, j]).any()):
                        #print(' y_pred_lstm[:, i, j] is', y_pred_lstm[:, i, j])
                        #print(' y_test_lstm[:, i, j] is', y_test_lstm[:, i, j])
               #         urmse_lstm[i, j] = unbiased_rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                #        GetKGE_lstm[i, j] = GetKGE(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                        #GetPCC_lstm[i, j] = GetPCC(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                 #       GetNSE_lstm[i, j] = GetNSE(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                  #      r2_lstm[i, j] = r2_score(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                        #rv_lstm[i, j] = _rv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                       # fhv_lstm[i, j] = _fhv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                        #flv_lstm[i, j] = _flv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                        # print(' r2_lstm[i, j] is', r2_lstm[i, j])
                   #     r_lstm[i, j] = np.corrcoef(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])[0, 1]
                    #    rmse_lstm[i, j] = _rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                     #   bias_lstm[i, j] = _bias(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
            #np.save(out_path_lstm + 'r2_' + str(modelname) + '.npy', r2_lstm)
            #np.save(out_path_lstm + 'KGE_' +str(modelname)  + '.npy', GetKGE_lstm)
        
            #np.save(out_path_lstm + 'NSE_' + str(modelname)  + '.npy', GetNSE_lstm)
  
            #np.save(out_path_lstm + 'rmse_' + str(modelname)  + '.npy', rmse_lstm)
            #np.save(out_path_lstm + 'bias_' + str(modelname)  + '.npy', bias_lstm)
            #np.save(out_path_lstm + 'urmse_' + str(modelname)  + '.npy', urmse_lstm)
          #  print('postprocess ove, please go on')
    







               


