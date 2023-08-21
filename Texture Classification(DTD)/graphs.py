import os

#reference 1:https://pytorch.org/docs/
#reference 2:https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
#reference 3:https://www.deeplearningbook.org
#resources used: https://cloud.lambdalabs.com/
#reference 4:https://pytorch.org/docs/

def loadtb_events():
    run_tensorboard = f"tensorboard --logdir={event_path} --port={6007}"
    os.system(run_tensorboard)
    

if __name__ == '__main__':
    print("Launching Tensorboard Event Files")
    #Get myoptimized_model_logs
    mydirec = os.getcwd()
    event_path = "myoptimized_model_log"

    loadtb_events()