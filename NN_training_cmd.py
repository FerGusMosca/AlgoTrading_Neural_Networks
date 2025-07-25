from common.util.logging.logger import Logger
from common.util.std_in_out.ml_settings_loader import MLSettingsLoader
from framework.common.logger.message_type import MessageType
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic

_DATE_FORMAT="%m/%d/%Y"


last_trading_dict=None

def show_commands():

    print("#1-TrainDeepNeuralNetwork [true_path] [false_path] [true_lavel] [learning_rate] [iterations] [arch_file] [activ_file] [output file] [step size]")
    print("#2-TrainDeepNeuralNetworkReg [true_path] [false_path] [true_lavel] [learning_rate] [iterations] [arch_file] [activ_file] [output file] [step size] [lambd] [useHeInit]")
    print("#3-TestDeepNeuralNetworkModel [true_path] [false_path] [true_lavel] [model file]")
    print("#4-TrainConvolutionalNeuralNetwork [train_true_path] [train_false_path] [test_true_path] [test_false_path] [true_lavel] [arch_file] [padding] [stride] [iterations]")

    print("#n-Exit")

def params_validation(cmd,param_list,exp_len):
    if(len(param_list)!=exp_len):
        raise Exception("Command {} expects {} parameters".format(cmd,exp_len))

def process_train_deep_neural_network(true_path,false_path,true_label,learning_rate,iterations,arch_file, activ_file,output_file,
                                      step_size,lambd=0.0,use_He_init=False):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Train Deep Neural Network",MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings["classification_map_key"], logger)
        dataMgm.train_deep_neural_network(true_path, false_path, true_label,learning_rate,iterations,arch_file, activ_file,
                                        output_file,step_size,lambd=lambd,use_He_init=use_He_init)

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)

def process_train_convolutional_neural_network(train_true_path,train_false_path,test_true_path, test_false_path,true_label,arch_file,padding,stride,iterations):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Train Convolutional Neural Netwokrk", MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        orch_algos_logic = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                                  config_settings["classification_map_key"], logger)
        orch_algos_logic.train_convolutional_neural_network(train_true_path, train_false_path, test_true_path, test_false_path, true_label, arch_file, padding, stride,
                                                            iterations)

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def test_deep_neural_network_model(true_path,false_path,true_label,output_file):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Test Deep Neural Network Model", MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings["classification_map_key"], logger)
        dataMgm.test_deep_neural_network_model(true_path,false_path,true_label,output_file)

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)
def process_commands(cmd):

    cmd_param_list=cmd.split(" ")

    if cmd_param_list[0] == "TrainDeepNeuralNetwork":
        params_validation("TrainDeepNeuralNetwork", cmd_param_list, 10)
        process_train_deep_neural_network(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3],float( cmd_param_list[4]),
                                          int( cmd_param_list[5]),cmd_param_list[6],cmd_param_list[7],cmd_param_list[8],
                                          int(cmd_param_list[9]))
    elif cmd_param_list[0] == "TrainDeepNeuralNetworkReg":
        params_validation("TrainDeepNeuralNetworkReg", cmd_param_list, 12)
        process_train_deep_neural_network(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3],
                                          float(cmd_param_list[4]),
                                          int(cmd_param_list[5]), cmd_param_list[6], cmd_param_list[7],
                                          cmd_param_list[8],
                                          int(cmd_param_list[9]),
                                          lambd=float(cmd_param_list[10]),
                                          use_He_init=True if cmd_param_list[11]=="True" else False)
    elif cmd_param_list[0] == "TestDeepNeuralNetworkModel":
        params_validation("TestDeepNeuralNetworkModel", cmd_param_list, 5)
        test_deep_neural_network_model(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3],cmd_param_list[4])

    elif cmd_param_list[0] == "TrainConvolutionalNeuralNetwork":
        params_validation("TrainConvolutionalNeuralNetwork", cmd_param_list, 10)
        process_train_convolutional_neural_network(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3],
                                                   cmd_param_list[4], cmd_param_list[5], cmd_param_list[6],
                                                   cmd_param_list[7],int(cmd_param_list[8]),int(cmd_param_list[9]))
    else:
        print("Not recognized command {}".format(cmd_param_list[0]))


if __name__ == '__main__':

    while True:

        show_commands()
        cmd=input("Enter a command:")
        try:
            process_commands(cmd)
            if(cmd=="Exit"):
                break
        except Exception as e:
            print("Could not process command:{}".format(str(e)))


    print("Exit")
