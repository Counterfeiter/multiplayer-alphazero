import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

# Object that manages interfacing data with the underlying PyTorch model, as well as checkpointing models.
class NeuralNetwork():

    def __init__(self, game, model_class, lr=1e-3, weight_decay=1e-8, batch_size=64, cuda=False, writer = None):
        self.game = game
        self.batch_size = batch_size
        init_state = game.get_initial_state()
        p_shape = game.get_available_actions(init_state).shape
        v_shape = (game.get_num_players(),)
        self.model = model_class(init_state["obs"], p_shape, v_shape)
        self.cuda = cuda
        if self.cuda:
            self.model = self.model.to('cuda')
            self.model = torch.nn.DataParallel(self.model)
        if len(list(self.model.parameters())) > 0:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        network_name = self.model.module.__class__.__name__ if self.cuda else self.model.__class__.__name__
        name = "{}-{}".format(self.game.__class__.__name__, network_name)

        self.writer = writer

        self.epoche_cnt = 0


    # Incoming data is a numpy array containing (state, prob, outcome) tuples.
    def train(self, data):
        loss_weight_v = 0.95
        self.model.train()
        batch_size=self.batch_size
        idx = np.random.randint(len(data), size=batch_size)
        batch = data[idx][:,:3] # select data, add no action mask
        action_mask = data[idx][:,3]
        states = np.stack(batch[:,0])
        cnn = torch.tensor([],dtype=torch.float32)
        ff = torch.tensor([],dtype=torch.float32)
        for value in states:
            cnn = torch.cat((cnn, torch.from_numpy(value["cnn_input"]).unsqueeze(0)), 0)
            ff = torch.cat((ff, torch.from_numpy(value["ff_input"]).unsqueeze(0)), 0)

        torch_states = {"cnn_input":cnn, "ff_input":ff}

        p_pred, v_pred = self.model(torch_states)
        p_gt, v_gt = batch[:,1], np.stack(batch[:,2])
        loss_p, loss_v = self.loss(torch_states, action_mask, (p_pred, v_pred), (p_gt, v_gt))
        self.epoche_cnt += 1
        if self.writer is not None:
            self.writer.add_scalar('Training/DataSamples', len(data), self.epoche_cnt)
        self.optimizer.zero_grad()
        loss = (loss_p * (1.0 - loss_weight_v)) + (loss_v * loss_weight_v)
        self.writer.add_scalar('Loss/sum', loss, self.epoche_cnt)
        loss.backward()
        self.optimizer.step()
        self.latest_loss = loss


    # Given a single state s, does inference to produce a distribution of valid moves P and a value V.
    def predict(self, state, action_mask):
        self.model.eval()
        input_cnn = np.array([state["obs"]["cnn_input"]], dtype=np.float32)
        input_ff = np.array([state["obs"]["ff_input"]], dtype=np.float32)
        with torch.no_grad():
            input_cnn = torch.from_numpy(input_cnn)
            input_ff = torch.from_numpy(input_ff)
            p_logits, v = self.model({"cnn_input":input_cnn, "ff_input":input_ff})
            p, v = self.get_valid_dist(p_logits[0], action_mask).cpu().numpy().squeeze(), v.cpu().numpy().squeeze() # EXP because log softmax
        return p, v

    # Given a single state s, does inference to produce a distribution of valid moves P and a value V.
    def predict_ray(self, obs, action_mask):
        self.model.eval()
        input_cnn = np.array([obs["cnn_input"]], dtype=np.float32)
        input_ff = np.array([obs["ff_input"]], dtype=np.float32)
        with torch.no_grad():
            input_cnn = torch.from_numpy(input_cnn)
            input_ff = torch.from_numpy(input_ff)
            action_mask = torch.from_numpy(action_mask)
            p_logits, v = self.model({"cnn_input":input_cnn, "ff_input":input_ff})
            assert p_logits.size()[0] <= 1
            #only softmax over selection
            selection = torch.masked_select(p_logits, action_mask)
            dist = torch.nn.functional.log_softmax(selection, dim=-1)
            dist = torch.exp(dist)
            #revert selection, build full vector
            softmax_logits = torch.zeros_like(p_logits[0])
            softmax_logits[action_mask] = dist
        return softmax_logits.cpu().numpy().squeeze(), v.cpu().numpy().squeeze()


    # MSE + Cross entropy
    def loss(self, states, action_mask, prediction, target):
        batch_size = len(states)
        p_pred, v_pred = prediction
        p_gt, v_gt = target
        v_gt = torch.from_numpy(v_gt.astype(np.float32))
        if self.cuda:
            v_gt = v_gt.cuda()
        v_loss = 0
        p_loss = 0
        for i in range(batch_size):
            ### value loss
            current_player_cnt = v_gt[i,-1].int()
            v_loss += ((v_pred[i, :current_player_cnt] - v_gt[i, :current_player_cnt])**2).sum() # Mean squared error

            ### policy loss
            gt = torch.from_numpy(p_gt[i].astype(np.float32))
            if self.cuda:
                gt = gt.cuda()
            #s = states[i]
            logits = p_pred[i]
            pred = self.get_valid_dist(logits, action_mask[i], log_softmax=True)
            p_loss += -torch.sum(gt*pred)

        if self.writer is not None:
            self.writer.add_scalar('Loss/policy', p_loss / batch_size, self.epoche_cnt)
            self.writer.add_scalar('Loss/value', v_loss / batch_size, self.epoche_cnt)
        return (p_loss / batch_size),  (v_loss / batch_size)


    # Takes one state and logit set as input, produces a softmax/log_softmax over the valid actions.
    def get_valid_dist(self, logits, action_mask, log_softmax=False):
        if type(logits) == np.ndarray:
            logits = torch.from_numpy(logits)
            if self.cuda:
                logits = logits.cuda()
        mask = torch.from_numpy(action_mask)
        if self.cuda:
            mask = mask.cuda()
        selection = torch.masked_select(logits, mask)
        dist = torch.nn.functional.log_softmax(selection, dim=-1)
        if log_softmax:
            return dist
        return torch.exp(dist)


    # Saves the current network along with its current pool of training data and training error history.
    # Provide the name of the save file.
    def save(self, name, training_data, error_log):
        network_name = self.model.module.__class__.__name__ if self.cuda else self.model.__class__.__name__
        directory = "checkpoints/{}-{}".format(self.game.__class__.__name__, network_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        network_path = "{}/{}.ckpt".format(directory, name)
        data_path = "{}/training.data".format(directory)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'error_log': error_log,
            }, network_path)
        torch.save({
            'training_data': training_data,
            }, data_path)


    # Loads the network at the given name.
    # Optionally, also load and return the training data and training error history.
    def load(self, name, load_supplementary_data=False):
        network_name = self.model.module.__class__.__name__ if self.cuda else self.model.__class__.__name__
        directory = "checkpoints/{}-{}".format(self.game.__class__.__name__, network_name)
        network_path = "{}/{}.ckpt".format(directory, name)
        network_checkpoint = torch.load(network_path)
        self.model.load_state_dict(network_checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(network_checkpoint['optimizer_state_dict'])
        if load_supplementary_data:
            #file could be broken cause of killing the process
            try:
                data_path = "{}/training.data".format(directory)
                data_checkpoint = torch.load(data_path)
            except:
                pass
            else:
                return data_checkpoint['training_data'], network_checkpoint['error_log']
        return None, None


    # Utility function for listing all available model checkpoints.
    def list_checkpoints(self, suffix_str = ""):
        network_name = self.model.module.__class__.__name__ if self.cuda else self.model.__class__.__name__
        path = "checkpoints/{}-{}{}/".format(self.game.__class__.__name__, network_name,suffix_str)
        if  not os.path.isdir(path):
            return []
        return sorted([int(filename.split(".ckpt")[0]) for filename in os.listdir(path) if filename.endswith(".ckpt")])

