# Adapted from: https://github.com/ml-stat-Sustech/TorchCP/blob/master/torchcp/classification/predictor/split.py
# and https://github.com/ml-stat-Sustech/TorchCP/blob/master/torchcp/regression/predictor/split.py
# Copyright (C) 2022 SUSTech Machine Learning and Statistics Group
# Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0)
# See https://www.gnu.org/licenses/lgpl-3.0.html for license details.

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

import torchcp.regression as cpregression
import torchcp.classification as cpclassification

from torchcp.utils.common import calculate_conformal_value


class PygSplitPredictor(cpregression.predictor.base.BasePredictor):
    """
    Split Conformal Prediction for Regression.
    
    This predictor allows for the construction of a prediction band for the response 
    variable using any estimator of the regression function.
        
    Args:
        score_function (torchcp.regression.scores): A class that implements the score function.
        model (torch.nn.Module): A pytorch regression model that can output predicted point.
        
    Reference:
        Paper: Distribution-Free Predictive Inference For Regression (Lei et al., 2017)
        Link: https://arxiv.org/abs/1604.04173
        Github: https://github.com/ryantibs/conformal
    """

    def __init__(self, score_function, target=0, model=None):
        super().__init__(score_function, model)
        self.target = target

    def train(self, train_dataloader, **kwargs):
        """
        Trains the model using the provided train_dataloader and score_function.

        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
            **kwargs: Additional keyword arguments for training configuration.
                - model (nn.Module, optional): The model to be trained.
                - epochs (int, optional): Number of training epochs.
                - criterion (nn.Module, optional): Loss function.
                - lr (float, optional): Learning rate for the optimizer.
                - optimizer (torch.optim.Optimizer, optional): Optimizer for training.
                - verbose (bool, optional): If True, prints training progress.

        .. note::
            This function is optional but recommended, because the training process for each score_function is different. 
            We provide a default training method, and users can change the hyperparameters :attr:`kwargs` to modify the training process.
            If the train function is not used, users should pass the trained model to the predictor at the beginning.
        """
        model = kwargs.pop('model', None)

        if model is not None:
            self._model = self.score_function.train(
                train_dataloader, model=model, device=self._device, **kwargs
            )
        elif self._model is not None:
            self._model = self.score_function.train(
                train_dataloader, model=self._model, device=self._device, **kwargs
            )
        else:
            self._model = self.score_function.train(
                train_dataloader, device=self._device, **kwargs
            )

    def calculate_score(self, predicts, y_truth):
        """
        Calculate the nonconformity scores based on the model's predictions and true values.

        Args:
            predicts (torch.Tensor): Model predictions.
            y_truth (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Computed scores for each prediction.
        """
        return self.score_function(predicts, y_truth)

    def generate_intervals(self, predicts_batch, q_hat):
        """
        Generate prediction intervals based on the model's predictions and the conformal value.

        Args:
            predicts_batch (torch.Tensor): Batch of predictions from the model.
            q_hat (float): Conformal value computed during calibration.

        Returns:
            torch.Tensor: Prediction intervals.
        """
        return self.score_function.generate_intervals(predicts_batch, q_hat)
    
    def __collect_scores__(self, cal_dataloader, test_dataloader):
        self._model.eval()
        predicts_list, y_truth_list = [], []
        with torch.no_grad():
            for examples in cal_dataloader:
                data, tmp_labels = examples.to(self._device), examples.y

                tmp_predicts = (self._model(data) * self._model.scale + self._model.shift).detach()
                
                
                predicts_list.append(tmp_predicts)
                y_truth_list.append(tmp_labels)

            for examples in test_dataloader:
                data, tmp_labels = examples.to(self._device), examples.y

                tmp_predicts = (self._model(data) * self._model.scale + self._model.shift).detach()
                
                
                predicts_list.append(tmp_predicts)
                y_truth_list.append(tmp_labels)

        
        predicts = torch.cat(predicts_list).float().to(self._device)
        y_truth = torch.cat(y_truth_list).to(self._device)

        self.n_total = predicts.shape[0]

        return predicts, y_truth
    
    def run_trials(self, cal_loader, test_loader, n_trials, alpha, cal_ratio=0.5):

        predictions, y_truth = self.__collect_scores__(cal_loader, test_loader)

        cal_size = int(self.n_total * cal_ratio)

        scores = self.calculate_score(predictions, y_truth)

        coverages, widths = [], []

        for _ in range(n_trials):
            indices = torch.randperm(self.n_total)
            cal_indices = indices[:cal_size].long()
            test_indices = indices[cal_size:].long()

            cal_scores = scores[cal_indices]
            q_hat = self._calculate_conformal_value(cal_scores, alpha)

            test_predictions = predictions[test_indices]
            test_y = y_truth[test_indices]

            test_intervals = self.generate_intervals(test_predictions, q_hat)

            coverages.append(self._metric('coverage_rate')(test_intervals, test_y))
            widths.append(self._metric('average_size')(test_intervals))
        return coverages, widths

    def calibrate(self, cal_dataloader, alpha):
        self._model.eval()
        predicts_list, y_truth_list = [], []
        with torch.no_grad():
            for examples in cal_dataloader:
                data, tmp_labels = examples.to(self._device), examples.y

                tmp_predicts = (self._model(data) * self._model.scale + self._model.shift).detach()#.unsqueeze(1)
                
                predicts_list.append(tmp_predicts)
                y_truth_list.append(tmp_labels)

        predicts = torch.cat(predicts_list).float().to(self._device)
        y_truth = torch.cat(y_truth_list).to(self._device)

        self.scores = self.calculate_score(predicts, y_truth)
        self.q_hat = self._calculate_conformal_value(self.scores, alpha)

    def predict(self, x_batch):
        self._model.eval()
        x_batch.to(self._device)
        with torch.no_grad():
            
            predicts_batch = (self._model(x_batch) * self._model.scale + self._model.shift)#.unsqueeze(1)
            return self.generate_intervals(predicts_batch, self.q_hat)

    def evaluate(self, data_loader):
        y_list, predict_list = [], []
        with torch.no_grad():
            for batch in data_loader:
                tmp_x, tmp_y = batch.to(self._device), batch.y
                # tmp_x, tmp_y = tmp_x.to(self._device), tmp_y.to(self._device)
                tmp_prediction_intervals = self.predict(tmp_x)
                y_list.append(tmp_y)
                predict_list.append(tmp_prediction_intervals)

        predicts = torch.cat(predict_list, dim=0).to(self._device)
        test_y = torch.cat(y_list).to(self._device)

        res_dict = {
            "coverage_rate": self._metric('coverage_rate')(predicts, test_y),
            "average_size": self._metric('average_size')(predicts)
        }
        return res_dict



class PygClassificationSplitPredictor(cpclassification.predictor.base.BasePredictor):
    """
    Split Conformal Prediction (Vovk et a., 2005).
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.
    
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module, optional): A PyTorch model. Default is None.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
    """

    def __init__(self, score_function, model=None, temperature=1):
        super().__init__(score_function, model, temperature)

    def __collect_scores__(self, cal_dataloader, test_dataloader):
        self._model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples.to(self._device), examples.y.to(self._device)
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()

                
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)


            for examples in test_dataloader:
                tmp_x, tmp_labels = examples.to(self._device), examples.y.to(self._device)
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()

                
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)


        logits = torch.cat(logits_list).float().to(self._device)
        labels = torch.cat(labels_list).to(self._device)


        self.n_total = logits.shape[0]

        return logits, labels
    
    def run_trials(self, cal_loader, test_loader, n_trials, alpha, cal_ratio=0.5):

        logits, labels = self.__collect_scores__(cal_loader, test_loader)

        cal_size = int(self.n_total * cal_ratio)

        coverages, widths = [], []

        for _ in range(n_trials):
            indices = torch.randperm(self.n_total)
            cal_indices = indices[:cal_size].long()
            test_indices = indices[cal_size:].long()

            cal_labels = labels[cal_indices]
            cal_scores = self.score_function(logits[cal_indices], cal_labels)

            q_hat = self._calculate_conformal_value(cal_scores, alpha)
            
            test_scores = self.score_function(logits[test_indices])
            # test_predictions = logits[test_indices]
            test_labels = labels[test_indices]

            S = self._generate_prediction_set(test_scores, q_hat)
            coverages.append(self._metric('coverage_rate')(S, test_labels))
            widths.append(self._metric('average_size')(S, test_labels))
        return coverages, widths
    
    #############################
    # The calibration process
    ############################
    def calibrate(self, cal_dataloader, alpha):

        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0, 1).")

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        self._model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples.to(self._device), examples.y.to(self._device)
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha):
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        scores = self.score_function(logits, labels)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def _calculate_conformal_value(self, scores, alpha):
        return calculate_conformal_value(scores, alpha)

    #############################
    # The prediction process
    ############################
    def predict(self, x_batch):
        """
        Generate prediction sets for a batch of instances.

        Args:
            x_batch (torch.Tensor): A batch of instances.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        self._model.eval()
        x_batch = self._model(x_batch.to(self._device)).float()
        x_batch = self._logits_transformation(x_batch).detach()
        sets = self.predict_with_logits(x_batch)
        return sets

    def predict_with_logits(self, logits, q_hat=None):
        """
        Generate prediction sets from logits.

        Args:
            logits (torch.Tensor): Model output before softmax.
            q_hat (torch.Tensor, optional): The conformal threshold. Default is None.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """

        scores = self.score_function(logits).to(self._device)
        if q_hat is None:
            if self.q_hat is None:
                raise ValueError("Ensure self.q_hat is not None. Please perform calibration first.")
            q_hat = self.q_hat

        S = self._generate_prediction_set(scores, q_hat)

        return S

    #############################
    # The evaluation process
    ############################

    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate prediction sets on validation dataset.
        
        Args:
            val_dataloader (torch.utils.data.DataLoader): Dataloader for validation set.
        
        Returns:
            dict: Dictionary containing evaluation metrics:
                - Coverage_rate: Empirical coverage rate on validation set
                - Average_size: Average size of prediction sets
        """
        predictions_sets_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        # Evaluate in inference mode
        self._model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device and get predictions
                inputs = batch.to(self._device)
                labels = batch.y.to(self._device)

                # Get predictions as bool tensor (N x C)
                batch_predictions = self.predict(inputs)
                
                # Accumulate predictions and labels
                predictions_sets_list.append(batch_predictions)
                labels_list.append(labels)

        # Concatenate all batches
        val_prediction_sets = torch.cat(predictions_sets_list, dim=0)  # (N_val x C)
        val_labels = torch.cat(labels_list, dim=0)  # (N_val,)

        # Compute evaluation metrics
        metrics = {
            "coverage_rate": self._metric('coverage_rate')(val_prediction_sets, val_labels),
            "average_size": self._metric('average_size')(val_prediction_sets, val_labels)
        }

        return metrics