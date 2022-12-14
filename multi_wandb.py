from multiprocessing import Pipe, Process
import wandb

########################################################################
# A class to handle multiple wandb sessions in one script.
########################################################################

def run_wandb_session(pipe, config, *args, **kwargs):
    wandb.init(config=config, *args, **kwargs)

    while(True):
        log_dict = pipe.recv()
        if(log_dict is False):
            break
        elif(log_dict is not None):
            wandb.log(log_dict)
    pipe.close()
    wandb.finish()

class multi_wandb():
    def __init__(self, configs, ids=None, *args, **kwargs):
        self.pipes = []
        self.procs = []
        if(ids is None):
            ids = [None]*len(configs)
        for i, (c,_id) in enumerate(zip(configs, ids)):
            child_p, p = Pipe(duplex=False)
            self.pipes+=[p]
            kwargs["id"] = _id
            proc = Process(target=run_wandb_session, args=[child_p,c]+list(args),kwargs=kwargs)
            self.procs+=[proc]
            proc.start()

    def log(self, log_dicts):
        for d,p in zip(log_dicts,self.pipes):
            p.send(d)

    def log_at_index(self, log_dict, index):
        self.pipes[index].send(log_dict)

    def end(self):
        for p in self.pipes:
            p.send(False)
            p.close()
        for proc in self.procs:
            proc.join()
