import os
import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from torch.utils.data import random_split, DataLoader

class Zer0t0rchWrapper():
    """
    -> this is the main class which wraps around the pytorch model and can 
    basically be used as a keras model with .compile(), .fit() and .predict()
    -> just inherit this class and customize any of the methods to your liking if needed
    """
    
    def __init__(self, model, device=None, seed=42):
        """
        initializes the zer0t0rch wrapper with the provided pytorch model and device,
        seeds the environment and initializes the optimizer on the model

        args:
            model (torch.nn.Module): a pytorch model
            device (torch.device): model and loaders will be mounted on this device
            seed (int): used to seed stuff
        """
        self.blank_model = copy.deepcopy(model)
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = model.to(self.device)
        self.seed_everything(seed)
        self.configure_optimizer()
        print(f'using device={self.device} and seed={seed}')

    def seed_everything(self, seed):
        """
        seeds the env

        args:
            seed (int): diz da seed broo
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)

    def configure_optimizer(self):
        """
        initializes the optimizer on the model
        customize this if needed
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def reset_params(self):
        """
        loads the initial parameters given to the wrapper 
        """
        self.model = self.blank_model.to(self.device)
        self.configure_optimizer()

    def prepare_data(self, data, batch_size, num_workers=2, pin_memory=True, val_data=None, val_pct=None, test_data=None, collate_fn=None):
        """
        prepares train, val and test loaders
        customize this if needed

        args:
            data (torch dataset): diz da data
            batch_size (int): data is split into minibatches of this size
            num_workers (int): num workers to be used while initalizing the data loaders
            pin_memory (bool): decide whether or not the data should be moved to page-locked memory
                               if set to True, this speeds up both the training and inference
            val_data (torch dataset): diz da validation data
            val_pct (float): data is split into train_data and val_data with len(data) being val_pct of the entire data
            test_data (torch dataset): diz da test data, can also be passed to the test() method
            collate_fn (def): provide a fucntion which does the stuff you need to the data after splitting it into minibatches,
                              will be provided to the DataLoader, pretty standard PyTorch stuff
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

        if val_pct != None and val_data == None:
            val_len = int(val_pct * len(data))
            train_data, val_data = random_split(data, [len(data) - val_len, val_len])
        else:
            train_data = data

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
        if val_data != None:
            self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
        else:
            self.val_loader = None

        if test_data != None:
            self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
        else:
            self.test_loader = None

    def compile(self, loss_fn, metric_fns=None):
        """
        stores the loss function and metric functions

        args:
            loss_fn (def): calcs the loss, if you want to change the shapes
                           of the inputs to loss_fn, write a custom fn which
                           does that and integrate it into the loss_fn and pass
                           the whole thing here
            metric_fns (dict): a dict with metric name as key and the
                               corresponding metric fn as value,
                               calcs the metrics
        """
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns
        if self.metric_fns != None:
            self.metric_fns_len = len(self.metric_fns)

    def batch_mounting_logic(self, batch):
        """
        mounts the items in a batch onto the device
        customize this if needed

        args:
            batch (torch batch): while looping during training/validataion, 
                                 each batch is provided to this method

        returns:
            batch (list): a list containing all the mounted items in the given batch
        """
        batch = [items.to(self.device) for items in batch]
        return batch

    def loop_forward_logic(self, batch):
        """
        gets the prediction from the model, used during
        training, validation and testing
        customize this if needed

        args:
            batch (list): a list containing all the mounted items in the given batch

        returns:
            outputs (tuple): a tuple containing all the stuff required to calc 
                            loss and metric, usually contains (preds, y)
        """
        x, y = batch
        preds = self.model(x)
        outputs = (preds, y)
        return outputs

    def metric_calc_logic(self, metric_inps):
        """
        calcs the metrics, the metric functions should only
        return an int, coz they are used for logging purposes
        customize this if needed

        args:
            metric_inps (tuple): values required to compute metrics,
                                 contains preds, y and loss, use whatever
                                 you want
        
        returns:
            metric_vals (dict): a dict with metric name as key and the 
                                corresponding metric value as value
        """
        preds, y, loss = metric_inps
        metric_vals = {}
        for metric_name, metric_fn in self.metric_fns.items():
            metric_vals[metric_name] = metric_fn(preds, y)
        return metric_vals

    def backprop_logic(self, loss):
        """
        core backprop logic, used only during training
        customize this if needed

        args:
            loss (torch.tensor): loss calculated by loss_fn
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def single_batch_forward_logic(self, batch, is_train):
        """
        does a single forward pass on the given batch

        args:
            batch (list): a list containing all the mounted items in the given batch
            is_train (bool): used as a flag for backprop

        returns:
            loss (int): loss calculated by loss_fn
            metric_vals (dict): a dict with metric name as key and the 
                                corresponding metric value as value
        """
        batch = self.batch_mounting_logic(batch)
        with torch.set_grad_enabled(is_train):
            outputs = self.loop_forward_logic(batch)
            loss = self.loss_fn(*outputs)

            if self.metric_fns != None:
                metric_inps = (*outputs, loss)
                metric_vals = self.metric_calc_logic(metric_inps)
            else:
                metric_vals = None

        if is_train:
            self.backprop_logic(loss)

        return loss.item(), metric_vals

    def get_final_str(self, split, mean_loss, mean_metrics=None, epoch=None):
        """
        returns the string which is used for logging

        args:
            split (str): 'train' or ' val '
            mean_loss (int): mean loss during this epoch
            mean_metrics (dict): a dict with metric name as the key and the
                                corresponding mean value as the value
        
        returns:
            final_str (str): the string which is used for logging
        """
        if epoch != None:
            loss_str = f'{split}: epoch={epoch}, loss={mean_loss:.4f}'
        else:
            loss_str = f'loss={mean_loss:.4f}'

        if self.metric_fns != None:
            metric_str = ', '
            i = 1
            for metric_name, mean_metric_val in mean_metrics.items():
                metric_str += f'{metric_name}={mean_metric_val:.4f}'
                if i != self.metric_fns_len:
                    metric_str += ', '
                i += 1
        else:
            metric_str = ''

        final_str = loss_str + metric_str
        return final_str

    
    def loop_logic(self, loader, is_train, epoch=None):
        """
        heart of the entire wrapper, loops over the loader, prints a progress bar
        and logs the metrics and loss values

        args:
            loader (torch.utils.data.DataLoader): diz da loader
            is_train (bool): used as a flag for backprop
            epoch (int): used for logging
        
        returns:
            mean_loss (int): mean loss during this epoch
            mean_metrics (dict): a dict with metric name as the key and the
                                corresponding mean value as the value
        """
        self.model.train(is_train)
        losses = []
        if self.metric_fns != None:
            metrics_tracker = {}
            for metric_name, _ in self.metric_fns.items():
                metrics_tracker[metric_name] = []

        if is_train:
            split = 'train'
        else:
            split = ' val '

        pbar = tqdm(loader, total=len(loader))
        for batch in pbar:
            loss, metric_vals = self.single_batch_forward_logic(batch, is_train)
            losses.append(loss)

            if self.metric_fns != None:
                for metric_name, metric_val in metric_vals.items():
                    metrics_tracker[metric_name].append(metric_val)

            mean_loss = np.mean(losses)
            if self.metric_fns != None:
                mean_metrics = {metric_name: np.mean(metric_val_list) for metric_name, metric_val_list in metrics_tracker.items()}
            else:
                mean_metrics = None

            final_str = self.get_final_str(split, mean_loss, mean_metrics, epoch)
            pbar.set_description(final_str)

        return mean_loss, mean_metrics

    def overfit_on_batch_logic(self, batch, is_train, epoch, plot_graphs):
        """
        overfits on a single batch

        args:
            batch (list): a list containing all the mounted items in the given batch
            is_train (bool): used as a flag for backprop
            epoch (int): used for logging
            plot_graphs (bool): if True metrics will not be printed

        returns:
            loss (int): loss calculated by loss_fn
            metric_vals (dict): a dict with metric name as key and the 
                                corresponding metric value as value
        """
        if is_train:
            split = 'train'
        else:
            split = ' val '

        loss, metric_vals = self.single_batch_forward_logic(batch, is_train)
        if plot_graphs == False:
            final_str = self.get_final_str(split, loss, metric_vals, epoch)
            print(final_str)
        return loss, metric_vals

    def fit(self, num_epochs, plot_graphs=False, on_single_batch=False):
        """
        starts the training and validation loop

        args:
            num_epochs (int): diz da num_epochs broo
            plot_graphs (bool): decide whether you want to see the plots or not
            on_single_batch (bool): overfit on a single batch
        """
        if plot_graphs:
            if self.val_loader != None:
                groups = {'loss': ['train_loss', 'val_loss']}

                if self.metric_fns != None:
                    for metric_name, _ in self.metric_fns.items():
                        groups[metric_name] = [f'train_{metric_name}', f'val_{metric_name}']
            else:
                groups = {'loss': ['train_loss']}

                if self.metric_fns != None:
                    for metric_name, _ in self.metric_fns.items():
                        groups[metric_name] = [f'train_{metric_name}']

            liveloss = PlotLosses(groups=groups, outputs=[MatplotlibPlot()])

        if on_single_batch:
            train_batch = next(iter(self.train_loader))
            if self.val_loader != None:
                val_batch = next(iter(self.val_loader))

        for epoch in range(num_epochs):
            if plot_graphs:
                logs = {}

            if on_single_batch:
                train_loss, train_metrics = self.overfit_on_batch_logic(train_batch, True, epoch, plot_graphs)
            else:
                train_loss, train_metrics = self.loop_logic(self.train_loader, True, epoch)

            if plot_graphs:
                logs['train_loss'] = train_loss
                if self.metric_fns != None:
                    for metric_name, train_metric_val in train_metrics.items():
                        logs[f'train_{metric_name}'] = train_metric_val

            if self.val_loader != None:
                if on_single_batch:
                    val_loss, val_metrics = self.overfit_on_batch_logic(val_batch, False, epoch, plot_graphs)
                else:
                    val_loss, val_metrics = self.loop_logic(self.val_loader, False, epoch)

                if plot_graphs:
                    logs['val_loss'] = val_loss
                    if self.metric_fns != None:
                        for metric_name, val_metric_val in val_metrics.items():
                            logs[f'val_{metric_name}'] = val_metric_val

            if plot_graphs:
                liveloss.update(logs)
                liveloss.send()

    def test(self, test_data=None):
        """
        loops over the test_data and calcs the loss and metrics

        args:
            test_data (torch dataset): diz da test_data broo, can also be
                                       passed to the prepare_data() method
        """
        assert test_data != None or self.test_loader != None, "you did not pass in the test_data!!"

        if self.test_loader == None:
            self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.collate_fn)
        _, _ = self.loop_logic(self.test_loader, False)

    @torch.no_grad()
    def predict(self, x):
        """
        gets the prediction from the model, can be used for inference

        args:
            x (torch.tensor): input for the model

        returns:
            preds (torch.tensor): prediction from the model
        """
        self.model.eval()
        preds = self.model(x)
        return preds