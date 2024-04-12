import numpy as np
import wandb
from typing import List

class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        # precision = TP / (TP + FP)
        tp = 0
        fp = 0
        precisions = []
        for i in range(len(predicted)):
            for result in predicted[i]:
                    if result in actual[i]:
                        tp += 1
                    else:
                        fp += 1
            precisions.append(tp / float(tp + fp))
            tp, fp = 0, 0
        precision = sum(precisions) / len(precisions)

        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        tp = 0
        recalls = []
        for i in range(len(predicted)):
            for result in predicted[i]:
                if result in actual[i]:
                    tp += 1
            recalls.append(tp / len(actual[i]))
            tp, fp = 0, 0
        recall = sum(recalls) / len(recalls)

        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        pr = self.calculate_recall(actual, predicted) * self.calculate_precision(actual, predicted)
        p_plus_r = self.calculate_recall(actual, predicted) + self.calculate_precision(actual, predicted)
        f1 = 2 * pr / p_plus_r
        return f1
    
    def calculate_AP(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[str]
            The actual results
        predicted : List[str]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0
        tp = 0
        for i in range(len(predicted)):
            if predicted[i] in actual:
                tp += 1
                AP += tp / (i + 1)

        return AP
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        APs = []

        for i in range(len(predicted)):
            APs.append(self.calculate_AP(actual[i], predicted[i]))

        MAP = sum(APs) / len(APs)
        return MAP
    
    def cacluate_DCG(self, actual: [str], predicted: List[str]) -> float:
        """
        Calculates the Discounted Cumulative Gain (DCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        dcg = 0.0

        relevances = []
        for a in predicted:
            if a in actual:
                relevances.append(len(actual) - actual.index(a))
            else:
                relevances.append(0)
        dcg = self.DCG(relevances)

        return dcg

    def DCG(self, relevances):
        if len(relevances) == 0:
            return 0

        discounts = np.log2(np.arange(len(relevances)) + 2)
        return np.sum(relevances / discounts)

    def NDCG(self, relevances):
        dcg = self.DCG(relevances)
        perfect_dcg = self.DCG(sorted(relevances, reverse=True))
        if perfect_dcg == 0:
            return 0
        else:
            return dcg / perfect_dcg

    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        ndcgs = []
        ndcg = 0.0

        for i in range(len(predicted)):
            relevances = []
            for a in predicted[i]:
                if a in actual[i]:
                    relevances.append(len(actual[i]) - actual[i].index(a))
                else:
                    relevances.append(0)
            ndcgs.append(self.NDCG(relevances))
        if len(ndcgs) > 0:
            ndcg = sum(ndcgs) / len(ndcgs)

        return ndcg
    
    def cacluate_RR(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = 0.0
        if actual[0] in predicted:
            RR = 1 / (predicted.index(actual[0]) + 1)

        return RR
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        RRs = []

        for i in range(len(predicted)):
            RRs.append(self.cacluate_RR(actual[i], predicted[i]))

        MRR = sum(RRs) / len(RRs)
        return MRR
    

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """

        print(f"name = {self.name}")
        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1}")
        print(f"Average Precision = {ap}")
        print(f"Mean Average Precision = {map}")
        print(f"DCG = {dcg}")
        print(f"NDCG = {ndcg}")
        print(f"Reciprocal Rank = {rr}")
        print(f"Mean Reciprocal Rank = {mrr}")
      

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """

        # Initialize Wandb
        wandb.init(project="IMDB-MIR", entity="evaluation")

        # Log evaluation metrics
        wandb.log({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Average Precision": ap,
            "Mean Average Precision": map,
            "DCG": dcg,
            "NDCG": ndcg,
            "Reciprocal Rank": rr,
            "Mean Reciprocal Rank": mrr
        })

        wandb.finish()


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)



