# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:29:09 2023

@author: lee
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score

from models.train_model_multi import Train_Test
from models.lstm_fcn_multi import LSTM_FCNs
from models.rnn import RNN_model
from models.cnn_1d import CNN_1D
from models.fc import FC

import warnings
warnings.filterwarnings('ignore')

class Multilearning():
    def __init__(self, config, mode):
        """
        

        Parameters
        ----------
        config : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.mode = mode
        self.model_name = config['model']
        self.parameter = config['parameter']
        self.best_model_path = config['best_model_path']
        # build trainer
        self.trainer = Train_Test(config)
        
    def build_model(self):
        """
        
        Returns
        -------
        init_model : TYPE
            DESCRIPTION.

        """
        
        if self.mode == 'transfer' :
            init_model = LSTM_FCNs(
                    input_size=self.parameter['input_size'],
                    num_classes=self.parameter['source_class'],
                    num_layers=self.parameter['num_layers'],
                    lstm_drop_p=self.parameter['lstm_drop_out'],
                    fc_drop_p=self.parameter['fc_drop_out']
                )
        
        else : ## target 자체를 학습시키는 모델 만듬 ## self
            init_model = LSTM_FCNs(
                    input_size=self.parameter['input_size'],
                    num_classes_1=self.parameter['num_classes_1'],
                    num_classes_2=self.parameter['num_classes_2'],
                    num_layers=self.parameter['num_layers'],
                    lstm_drop_p=self.parameter['lstm_drop_out'],
                    fc_drop_p=self.parameter['fc_drop_out']
                )
        
        return init_model
    
    def train_model(self,train_x, train_y, valid_x, valid_y,option='source'):
        """
        

        Parameters
        ----------
        train_x : TYPE
            DESCRIPTION.
        train_y : TYPE
            DESCRIPTION.
        valid_x : TYPE
            DESCRIPTION.
        valid_y : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        train_loader = self.get_dataloader(train_x, train_y, self.parameter['batch_size'], shuffle=True)
        valid_loader = self.get_dataloader(valid_x, valid_y, self.parameter['batch_size'], shuffle=False)

        # build initialized model
        if option == 'target' :
            init_model = self.tuning_model(self.best_model_path,freeze=self.parameter['freeze'])
        else :
            init_model = self.build_model()
        
        # train model
        dataloaders_dict = {'train': train_loader, 'val': valid_loader}
        best_model = self.trainer.train(init_model, dataloaders_dict)
        return best_model
        
        
        
    def save_model(self,best_model,best_model_path):
        """

        Parameters
        ----------
        best_model : TYPE
            DESCRIPTION.
        best_model_path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        torch.save(best_model.state_dict(), best_model_path)
        
    
    def pred_data(self,test_x, test_y, best_model_path):
        """
        """
        
        test_loader = self.get_dataloader(test_x, test_y, self.parameter['batch_size'], shuffle=False)

        # build initialized model
        init_model = self.build_model()

        # load best model
        init_model.load_state_dict(torch.load(best_model_path))

        # get predicted classes
        pred_data_1, pred_data_2 = self.trainer.test(init_model, test_loader)

        # class의 값이 0부터 시작하지 않으면 0부터 시작하도록 변환

        # calculate performance metrics
        acc = accuracy_score(test_y[:,0], pred_data_1)
        
        mse = mean_squared_error(test_y[:,1], pred_data_2)
        MAPE = mean_absolute_percentage_error(test_y[:,1], pred_data_2)
        MAE = mean_absolute_error(test_y[:,1], pred_data_2)
        R2 = r2_score(test_y[:,1], pred_data_2)
        
        # merge true value and predicted value
        pred_df = pd.DataFrame()
        pred_df['actual_value_1'] = test_y[:,0]
        pred_df['predicted_value_1'] = pred_data_1
        pred_df['actual_value_2'] = test_y[:,1]
        pred_df['predicted_value_2'] = pred_data_2
        return pred_df, acc, mse, MAPE, MAE, R2
    
    def get_dataloader(self, x_data, y_data, batch_size, shuffle):
        """
        
        """
        # torch dataset 구축
        dataset = torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))

        # DataLoader 구축
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
        
    
    def tuning_model(self,best_model_path,freeze):
        # config 에 Source / Target dataset 정리       
        ## change / freeze / save
        
        # load best model
        init_model = self.build_model()
        init_model.load_state_dict(torch.load(best_model_path))
        
        # if self.parameter['source_class'] != self.parameter['target_class'] :
            
        #     print('model fc layer output change')
        #     in_features = init_model.fc.in_features
        #     out_features = self.parameter['target_class']
            
        #     init_model.fc = nn.Linear(in_features,out_features)
        
        if freeze:
            for name, param in init_model.named_parameters():
                if name in ['fc.weight','fc.bias']:
                    param.requires_grad = True
                
                else :
                    param.requires_grad = False
                print(param.requires_grad)
        return init_model
