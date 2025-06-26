import os
from datetime import datetime
from joblib import load
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier

from business_entities.portf_position import PortfolioPosition
from common.enums.machine_learning_algos import MachineLearningAlgos
from common.util.dataframe_filler import DataframeFiller
from common.util.file_writer import FileWriter
from common.util.light_logger import LightLogger
from framework.common.logger.message_type import MessageType
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle

from logic_layer.model_creators.random_forest_model_creator import RandomForestModelCreator
from logic_layer.trading_algos.direct_prediction_backtester import DirectPredictionBacktester

_OUTPUT_PATH="./output/"
_LOGISTIC_REGRESSION_MODEL_NAME="logistic_regression"
_SVM_MODEL_NAME="support_vector_machine"
_KNN_MODEL_NAME="k_nearest_neighbour"
_DECISSION_TREE_MODEL_NAME="decission_tree"

class MLModelAnalyzer():


    def __init__(self,p_logger):
        self.logger=p_logger


    @staticmethod
    def __build_model_base_name__(series_csv, d_from, d_to,p_classif_key):

        variables_def=series_csv.replace(",","+")
        s_from=d_from.strftime('%d%m%Y')
        s_to=d_to.strftime('%d%m%Y')
        timestamp=datetime.now().strftime('%d%m%Y%H%M%S')

        model_base_name=f"{p_classif_key}_{s_from}_{s_to}_{timestamp}"
        return model_base_name,variables_def




    def __val_invalid_values__(self,df):

        #columns_with_inf = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
        #columns_with_large_values = df.columns[(df.abs() > np.finfo(np.float64).max).any()].tolist()
        #columns_with_nan = df.columns[df.isna().any()].tolist()
        pass



    def __map_num_to_cat_array__(self,y_hat_num,y_mapping):
        y_hat_cat=[]
        for pred_value in y_hat_num:
            for key, value in y_mapping.items():
                if value == pred_value:
                    y_hat_cat.append(key)
                    break

        return  y_hat_cat

    def __persist_model__(self,trained_model,algo_name,model_base_name,y_mapping,variables_def=None):
        #We create the
        FileWriter.__create_directory__(_OUTPUT_PATH,algo_name)

        #Root paths and File Names
        model_root_path= f"{_OUTPUT_PATH}{algo_name}/"
        file_name = f"{algo_name}_{model_base_name}"

        #The Model for storage
        FileWriter.__dump_model__(model_root_path, file_name, trained_model, y_mapping, "wb")

        #Description File
        FileWriter.__write_file__(f"{model_root_path}{file_name}_desc.txt",variables_def,"w")

        #The quick access model --> To be used by the next EvalBiasedTradingAlgo cmd
        FileWriter.__dump_model__(_OUTPUT_PATH, algo_name, trained_model, y_mapping, "wb")

    def __fetch_model__(self,algo_name):
        file_path = "{}{}".format(_OUTPUT_PATH, "{}.pkl".format(algo_name))
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
            return loaded_model

    def __get_int_mapping__(self,df,col):
        unique_values = df[col].unique()
        i=0
        mapping={}
        for val in unique_values:
            mapping[val]=i
            i+=1

        return mapping

    def __map_categorical_Y__(self,df_Y,classification_col):
        # We normalize the Y categorical axis
        mapping = self.__get_int_mapping__(df_Y, classification_col)  # We gte SHORT=1,LONG=2, etc.
        Y = df_Y[classification_col].map(mapping)
        return Y,mapping

    def __normalize_X__(self,df_X):

        # Fill NaN values with the values from the previous row
        df_X = df_X.fillna(method='ffill')

        # We need to normalize just the numeric columns of the X dataframe
        all_colls=df_X.columns
        X_num_cols = df_X.select_dtypes(include='number').columns
        X_non_num_cols = df_X.columns.difference(X_num_cols)

        X_numeric = df_X[X_num_cols]
        X_num_scal = preprocessing.StandardScaler().fit_transform(X_numeric)
        X = pd.concat([pd.DataFrame(X_num_scal, columns=X_num_cols), df_X[X_non_num_cols]], axis=1)
        # After the concat --> THe X will have all its numeric column values properly fit
        X=X[all_colls]
        return X

    def __clean_NaN__(self,X_train,X_test,y_train,y_test):
        X_train = X_train.fillna(method='ffill')  # we remove NaN from Y --> prev value
        y_train = y_train.fillna(method='ffill')  # we remove NaN from Y --> prev value
        X_test = X_test.fillna(method='ffill')  # we remove NaN from Y --> prev value
        y_test = y_test.fillna(method='ffill')  # we remove NaN from Y --> prev value
        return X_train, X_test, y_train, y_test

    def  __extract_non_numeric__(self,X):
        X_num_cols = X.select_dtypes(include='number').columns
        X_numeric = X[X_num_cols]
        return  X_numeric

    def run_logistic_regression_eval(self,X_train, y_train,X_test,y_test,y_mapping,model_base_name,variables_def):
        # TRAINING - Logistic Regression w/GridSearchcv
        resp_row = {'Model': 'Logistic Regression', 'Train Accuracy': None,'Test Accuracy':None}

        X_train_num= self.__extract_non_numeric__(X_train)

        parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l2'], 'solver': ['lbfgs']}
        lr = LogisticRegression()
        logreg_cv = GridSearchCV(lr, parameters)
        logreg_cv.fit(X_train_num, y_train)
        self.logger.do_log("tuned hpyerparameters :(best parameters):{} ".format( logreg_cv.best_params_),MessageType.INFO)
        self.logger.do_log("Logistic Regression - Training Params Accuracy :{}".format(logreg_cv.best_score_), MessageType.INFO)
        resp_row["Train Accuracy"] = logreg_cv.best_score_

        # TEST - Accuracy of test data versus predictions + conf. matrix of predictions
        X_test_num = self.__extract_non_numeric__(X_test)
        lr_accuracy = logreg_cv.score(X_test_num, y_test)
        resp_row["Test Accuracy"] = lr_accuracy
        self.logger.do_log("Logistic Regression - Test Accuracy Score :{}".format(lr_accuracy),MessageType.INFO)

        self.__persist_model__(logreg_cv.best_estimator_, _LOGISTIC_REGRESSION_MODEL_NAME, model_base_name, y_mapping,
                               variables_def)

        #yhat = logreg_cv.predict(X_test)
        # self.plot_confusion_matrix(y_test, yhat)

        return  resp_row

    def build_out_of_sample_report_row(self,name,y_test,y_hat):
        resp_row = {'Model': name,'Accuracy':None,'Precision':None,'Recall':None,'F1':None}

        accuracy = accuracy_score(y_test, y_hat)
        precision = precision_score(y_test, y_hat)
        recall = recall_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat)
        cm = confusion_matrix(y_test, y_hat)
        report = classification_report(y_test, y_hat)

        resp_row["Accuracy"]=accuracy
        resp_row["Precision"] = precision
        resp_row["F1"] = f1
        resp_row["Recall"] = recall
        return resp_row

    def run_logistic_regression_eval_out_of_sample(self,X_test,y_test):
        # TRAINING - Logistic Regression w/GridSearchcv
        lr = self.__fetch_model__(_LOGISTIC_REGRESSION_MODEL_NAME)["model"]

        # TEST - Accuracy of test data versus predictions + conf. matrix of predictions
        X_test_num = self.__extract_non_numeric__(X_test)
        y_hat = lr.predict(X_test_num)
        return  self.build_out_of_sample_report_row("Logistic Regression",y_test,y_hat)

    def run_predictions(self,X_test,key_col,model_name):
        # TRAINING - Logistic Regression w/GridSearchcv
        model_dict = self.__fetch_model__(model_name)

        model=model_dict["model"]
        y_mapping=model_dict["label_mapping"]

        # TEST - Accuracy of test data versus predictions + conf. matrix of predictions
        X_test_num = self.__extract_non_numeric__(X_test)
        y_hat_num = model.predict(X_test_num)

        y_hat_cat= self.__map_num_to_cat_array__(y_hat_num, y_mapping)

        df_Y =pd.DataFrame( pd.Series(y_hat_cat, name='Prediction'))

        preds_df = pd.concat([X_test[key_col], df_Y['Prediction']], axis=1)

        return  preds_df

    def run_support_vector_machine_eval_out_of_sample(self,X_test,y_test):
        # TRAINING - SVM w/GridSearchcv
        svm = self.__fetch_model__(_SVM_MODEL_NAME)["model"]

        # TEST - Accuracy of test data versus predictions + conf. matrix of predictions
        X_test_num = self.__extract_non_numeric__(X_test)
        y_hat = svm.predict(X_test_num)
        return  self.build_out_of_sample_report_row("Support Vector Machine",y_test,y_hat)

    def run_decission_tree_eval_out_of_sample(self,X_test,y_test):
        # TRAINING - Decission Tree w/GridSearchcv
        dec_tree = self.__fetch_model__(_DECISSION_TREE_MODEL_NAME)["model"]

        # TEST - Accuracy of test data versus predictions + conf. matrix of predictions
        X_test_num = self.__extract_non_numeric__(X_test)
        y_hat = dec_tree.predict(X_test_num)
        return  self.build_out_of_sample_report_row("Decission Tree",y_test,y_hat)

    def run_K_nearest_neighbour_eval_out_of_sample(self,X_test,y_test):
        # TRAINING - K-Nearest Neighbour w/GridSearchcv
        knn = self.__fetch_model__(_KNN_MODEL_NAME)["model"]

        # TEST - Accuracy of test data versus predictions + conf. matrix of predictions
        X_test_num = self.__extract_non_numeric__(X_test)
        y_hat = knn.predict(X_test_num)
        return  self.build_out_of_sample_report_row("K-Nearest Neighbour",y_test,y_hat)

    def run_support_vector_machine_eval(self, X_train, y_train, X_test, y_test,y_mapping,model_base_name,variables_def):
        # TRAINING - SVM w/GridSearchcv
        resp_row = {'Model': 'Support Vector Machine', 'Train Accuracy': None, 'Test Accuracy': None}

        X_train_num = self.__extract_non_numeric__(X_train)

        parameters = {'kernel': ('linear', 'rbf', 'poly', 'rbf', 'sigmoid'),
                      'C': np.logspace(-3, 3, 5),
                      'gamma': np.logspace(-3, 3, 5)}
        svm = SVC()
        svm_cv = GridSearchCV(svm, parameters)
        svm_cv.fit(X_train_num, y_train)
        self.logger.do_log("tuned hpyerparameters :(best parameters):{} ".format( svm_cv.best_params_),MessageType.INFO)
        self.logger.do_log("Support Vector Machine - Training Params Accuracy :{}".format( svm_cv.best_score_),MessageType.INFO)
        resp_row["Train Accuracy"] =svm_cv.best_score_

        # TEST Calculate the accuracy on the test data using the method score
        X_test_num = self.__extract_non_numeric__(X_test)
        svm_accuracy = svm_cv.score(X_test_num, y_test)
        resp_row["Test Accuracy"] = svm_accuracy
        self.logger.do_log("Support Vector Machine - Test Accuracy Score :{}".format(svm_accuracy), MessageType.INFO)

        #yhat = logreg_cv.predict(X_test)
        #self.plot_confusion_matrix(y_test, yhat)

        self.__persist_model__(svm_cv.best_estimator_, _SVM_MODEL_NAME, model_base_name, y_mapping, variables_def)

        return resp_row

    def run_decision_tree_eval(self, X_train, y_train, X_test, y_test,y_mapping,model_base_name,variables_def,reuse_last=False):
        # TRAINING - Decission Tree w/GridSearchcv
        resp_row = {'Model': 'Decision Tree', 'Train Accuracy': None, 'Test Accuracy': None}

        X_train_num = self.__extract_non_numeric__(X_train)

        parameters = {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'max_depth': [2 * n for n in range(1, 10)],
                      'max_features': ['auto', 'sqrt'],
                      'min_samples_leaf': [1, 2, 4],
                      'min_samples_split': [2, 5, 10]}

        tree = DecisionTreeClassifier()
        tree_cv = GridSearchCV(tree, parameters)
        tree_cv.fit(X_train_num, y_train)
        self.logger.do_log("tuned hpyerparameters :(best parameters):{} ".format( tree_cv.best_params_),MessageType.INFO)
        self.logger.do_log("Decision Tree - Training Params Accuracy :{}".format(tree_cv.best_score_),MessageType.INFO)
        resp_row["Train Accuracy"] = tree_cv.best_score_

        # TEST Calculate the accuracy on the test data using the method score
        X_test_num = self.__extract_non_numeric__(X_test)
        tree_accuracy = tree_cv.score(X_test_num, y_test)
        self.logger.do_log("Decision Tree - Test Accuracy Score :{}".format(tree_accuracy),MessageType.INFO)
        resp_row["Test Accuracy"] = tree_accuracy

        self.__persist_model__(tree_cv.best_estimator_, _DECISSION_TREE_MODEL_NAME, model_base_name, y_mapping,
                               variables_def)

        return resp_row

    def run_k_nearest_neighbour_eval(self, X_train, y_train, X_test, y_test,y_mapping,model_base_name,variables_def):
        # TRAINING - KNN w/GridSearchcv
        resp_row = {'Model': 'K Nearest Neighbour', 'Train Accuracy': None, 'Test Accuracy': None}

        X_train_num = self.__extract_non_numeric__(X_train)

        parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'p': [1, 2]}

        KNN = KNeighborsClassifier()
        knn_cv = GridSearchCV(KNN, parameters)
        knn_cv.fit(X_train_num, y_train)
        self.logger.do_log("tuned hpyerparameters :(best parameters):{} ".format( knn_cv.best_params_),MessageType.INFO)
        self.logger.do_log("KNN - Training Params Accuracy :{}".format( knn_cv.best_score_),MessageType.INFO)
        resp_row["Train Accuracy"] = knn_cv.best_score_

        # TEST Calculate the accuracy on the test data using the method score
        X_test_num = self.__extract_non_numeric__(X_test)
        knn_accuracy = knn_cv.score(X_test_num, y_test)
        self.logger.do_log("KNN - Test Accuracy Score :{}".format(knn_accuracy),MessageType.INFO)
        resp_row["Test Accuracy"] = knn_accuracy

        self.__persist_model__(knn_cv.best_estimator_, _KNN_MODEL_NAME, model_base_name, y_mapping, variables_def)
        return resp_row


    def fit_and_evaluate(self,series_df,classification_col,model_base_name,variables_def,algos_arr=None):

        comparisson_df = pd.DataFrame(columns=['Model','Train Accuracy','Test Accuracy'])

        series_df = DataframeFiller.fill_missing_values(series_df)

        # STEP 1 - Split the dataframe inot X (indep. variable) and Y (dep variable)
        features=series_df.columns.to_list()
        features.remove(classification_col)
        df_X = series_df[features]
        df_Y = series_df[[classification_col]]

        #STEP 2 - Transform categorical values domension + Normalize the data
        df_X = pd.get_dummies(df_X)  # converts the categorical values into mult. cols

        #We map the Y categorical axis to int values
        Y,y_mapping= self.__map_categorical_Y__(df_Y, classification_col)

        #Then we normalize all the numerical values of X
        X= self.__normalize_X__(df_X)#we normalize X

        self.__val_invalid_values__(X)

        #STEP 3 - Split the data into training and test
        X_train, X_test, y_train, y_test =train_test_split(X, Y,test_size=0.05,random_state=2)
        X_train, X_test, y_train, y_test =self.__clean_NaN__(X_train,X_test,y_train,y_test)

        if algos_arr is None or MachineLearningAlgos.LOGISTIC_REGRESSION.value in algos_arr:
            #LOGISTIC REGRESSION
            resp_row= self.run_logistic_regression_eval(X_train, y_train,X_test,y_test,y_mapping,model_base_name,variables_def)
            comparisson_df = pd.concat([comparisson_df, pd.DataFrame([resp_row])], ignore_index=True)

        if algos_arr is None or MachineLearningAlgos.SVM.value in algos_arr:
            # SUPPORT VECTOR MACHINE
            resp_row = self.run_support_vector_machine_eval(X_train, y_train, X_test, y_test,y_mapping,model_base_name,variables_def)
            comparisson_df = pd.concat([comparisson_df, pd.DataFrame([resp_row])], ignore_index=True)

        if algos_arr is None or MachineLearningAlgos.DECISION_TREE.value in algos_arr:
            # DECISSION TREE
            resp_row = self.run_decision_tree_eval(X_train, y_train, X_test, y_test,y_mapping,model_base_name,variables_def)
            comparisson_df = pd.concat([comparisson_df, pd.DataFrame([resp_row])], ignore_index=True)


        if algos_arr is None or MachineLearningAlgos.KNN.value in algos_arr:
            # K NEAREST NEIGHBOUR
            resp_row = self.run_k_nearest_neighbour_eval(X_train, y_train, X_test, y_test,y_mapping,model_base_name,variables_def)
            comparisson_df = pd.concat([comparisson_df, pd.DataFrame([resp_row])], ignore_index=True)

        return comparisson_df

    def fetch_and_evaluate(self,series_df,classification_col):
        comparisson_df = pd.DataFrame(columns=['Model','Accuracy','Precision','Recall','F1'])

        # STEP 1 - Split the dataframe inot X (indep. variable) and Y (dep variable)
        features=series_df.columns.to_list()
        features.remove(classification_col)
        df_X = series_df[features]
        df_Y = series_df[[classification_col]]

        #STEP 2 - Transform categorical values domension + Normalize the data
        #df_X = pd.get_dummies(df_X)  # converts the categorical values into mult. cols

        #We map the Y categorical axis to int values
        Y,y_mapping= self.__map_categorical_Y__(df_Y, classification_col)

        #Then we normalize all the numerical values of X
        X= self.__normalize_X__(df_X)

        #LOGISTIC REGRESSION
        resp_row= self.run_logistic_regression_eval_out_of_sample(X,Y)
        comparisson_df=comparisson_df.append(resp_row, ignore_index=True)

        # # SUPPORT VECTOR MACHINE
        resp_row = self.run_support_vector_machine_eval_out_of_sample(X,Y)
        comparisson_df = comparisson_df.append(resp_row, ignore_index=True)

        # # DECISSION TREE
        resp_row = self.run_decission_tree_eval_out_of_sample(X,Y)
        comparisson_df = comparisson_df.append(resp_row, ignore_index=True)

        # # K NEAREST NEIGHBOUR
        resp_row = self.run_K_nearest_neighbour_eval_out_of_sample(X,Y)
        comparisson_df = comparisson_df.append(resp_row, ignore_index=True)

        return comparisson_df

    def run_predictions_last_model(self,series_df,algos_arr=None):
        predictions_dict={}

        series_df=DataframeFiller.fill_missing_values(series_df)

        # STEP 1 - Prepare the X to be normalized
        features=series_df.columns.to_list()
        df_X = series_df[features]

        #STEP 2 - Transform categorical values domension + Normalize the data
        df_X = pd.get_dummies(df_X)  # converts the categorical values into mult. cols

        #STEP 3- Then we normalize all the numerical values of X
        X= self.__normalize_X__(df_X)

        if algos_arr is None or MachineLearningAlgos.LOGISTIC_REGRESSION.value in algos_arr:
            #LOGISTIC REGRESSION
            y_hat_lr_df= self.run_predictions(X,"date",_LOGISTIC_REGRESSION_MODEL_NAME)
            predictions_dict["Logistic Regression"]=y_hat_lr_df

        if algos_arr is None or MachineLearningAlgos.SVM.value in algos_arr:
            # SUPPORT VECTOR MACHINE
            y_hat_svm_df= self.run_predictions(X,"date",_SVM_MODEL_NAME  )
            predictions_dict["Support Vector Machine"] = y_hat_svm_df

        if algos_arr is None or MachineLearningAlgos.DECISION_TREE.value in algos_arr:
            # DECISION TREE
            y_hat_dec_tree_df= self.run_predictions(X,"date",_DECISSION_TREE_MODEL_NAME  )
            predictions_dict["Decision Tree"] = y_hat_dec_tree_df

        if algos_arr is None or MachineLearningAlgos.KNN.value in algos_arr:
            # K NEAREST NEIGHBOUR
            y_hat_K_NN_df= self.run_predictions(X,"date",_KNN_MODEL_NAME  )
            predictions_dict["K-Nearest Neighbour"] = y_hat_K_NN_df


        return predictions_dict

    def evaluate_trading_performance_last_model(self,symbol_df,symbol, series_df,bias,
                                                last_trading_dict=None,
                                                n_algo_param_dict={},algos_arr=None):
        predictions_dic = self.run_predictions_last_model(series_df,algos_arr)

        direct_pred_backtester=  DirectPredictionBacktester()
        return direct_pred_backtester.backtest(symbol,symbol_df,predictions_dic,bias,
                                               last_trading_dict,
                                               n_algo_param_dict=n_algo_param_dict)



    def extract_label_encoder_from_model(self, model_to_use):
        try:
            model_bundle = load(model_to_use)

            if "label_encoder" not in model_bundle:
                raise ValueError(f"[RF Test] No 'label_encoder' found in model file: {model_to_use}")

            return model_bundle["label_encoder"]

        except Exception as e:
            raise Exception(f"[RF Test] Failed to extract label_encoder from model: {e}")

    def evaluate_trading_performance_last_model_RF(self, symbol_df, symbol, series_df, model_filename, bias,
                                                   label_encoder,last_trading_dict, n_algo_param_dict):
        """
        Generate prediction DataFrame from RF model.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (result_df, test_series_df)
        """

        rf_creator = RandomForestModelCreator()

        model_file = model_filename
        classif_threshold = float(n_algo_param_dict.get("classif_threshold", 0.6))
        make_stationary = n_algo_param_dict.get("make_stationary", True)

        # Merge symbol prices with feature variables
        test_series_df = pd.merge(symbol_df, series_df, on="date", how="inner")
        test_series_df["trading_symbol"] = symbol  # Required column

        # Filter out invalid rows
        test_series_df = test_series_df.dropna(subset=["date", "trading_symbol"])
        test_series_df = test_series_df[test_series_df["trading_symbol"] == symbol]

        if len(test_series_df) == 0:
            raise Exception(f"Empty test set for {symbol} → Check data availability or date range.")

        # Run RF prediction
        result_df, _ = rf_creator.test_RF_scalping(symbol=symbol, test_series_df=test_series_df,
                                                   model_to_use=model_file, make_stationary=make_stationary,
                                                   normalize=True, series_csv=n_algo_param_dict["series_csv"],
                                                   threshold=classif_threshold, label_encoder=label_encoder)

        # Alias required for backtest compatibility
        result_df[symbol] = result_df["trading_symbol_price"]

        return result_df, test_series_df














