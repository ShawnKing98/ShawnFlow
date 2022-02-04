import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset


class PlanningProblemDataset(Dataset):

    def __init__(self, dataset_filename, use_sdf_grad=True, dubins_car=False, use_U=False):
        super().__init__()

        data = np.load(dataset_filename)
        data = dict(data)

        # When SDF was generated I multiplied in collision values by 1000 for cost calculation
        # I undo this for the normalised sdf for better numerics for the flor
        # In future may change so that the initial sdf is already normalised, and is multiplied by 1000 for cost after
        sdf = data['sdf']
        self.N = sdf.shape[0]
        sdf = np.where(sdf < 0, sdf / 1000.0, sdf)# + .02 * np.random.randn(*sdf.shape)

        if not use_sdf_grad:
            self.sdf_grad = torch.empty(size=(self.N, *sdf.shape[1:], len(sdf.shape[1:])))
        else:
            self.sdf_grad = torch.from_numpy(data['sdf_grad']).float()

        # Convert all stuff to torch
        self.sdf = torch.from_numpy(sdf).float().unsqueeze(1)
        self.starts = torch.from_numpy(data['starts']).float()
        self.goals = torch.from_numpy(data['goals']).float()

        self.U = None
        if use_U:
            self.U = torch.from_numpy(data['U']).float()

        if dubins_car:
            self.starts = self.starts[:, :, :3]
            self.goals = self.goals[:, :, :3]

        self.planning_problem_per_environment = self.starts.shape[1]

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        # randomly sample planning problem for environments
        plan_idx = np.random.randint(low=0, high=self.planning_problem_per_environment)

        if self.U is None:
            U = torch.tensor(0.0)
        else:
            U = self.U[item, plan_idx]

        return self.starts[item, plan_idx], self.goals[item, plan_idx], \
               self.sdf[item], self.sdf_grad[item], U




class PlanningProblemDatasetNoLoad(Dataset):
    '''
    Planning dataset which does not load entire dataset into ram, instead loads when calling __get_item__
    Will be much slower to iterate through, but less memory intensive for larger datasets

    # For each sample we load an entire dataset, so maybe there is a nicer way of doing things

    Recommended to use multiple workers

    # TODO would adding slicing help?
    '''
    def __init__(self, dataset_filename, use_sdf_grad=True, noisy_sdf=False):
        super().__init__()
        self.fname = dataset_filename
        self.use_sdf_grad = use_sdf_grad
        self.add_noise = noisy_sdf
        self.N = 0
        # load data so we can get the size
        starts, _, _, _, _ = self.load_data()
        self.planning_problem_per_env = starts.shape[1]

    def load_data(self):
        print('loading')
        data = np.load(self.fname)
        data = dict(data)

        sdf = data['sdf']
        normalised_sdf = np.where(sdf < 0, sdf / 1000.0, sdf)
        self.N = sdf.shape[0]

        if self.add_noise:
            normalised_sdf += 0.2 * np.random.randn(*sdf.shape)

        # If we don't need gradients we can have a cliff in cost and make all collision v. expensive
        if not self.use_sdf_grad:
            sdf = np.where(sdf < 0, -1e3 * np.ones_like(sdf), np.zeros_like(sdf))
            sdf_grad = np.zeros((self.N, *sdf.shape[1:], len(sdf.shape[1:])))
        else:
            sdf_grad = data['sdf_grad']

        return data['starts'], data['goals'], normalised_sdf, sdf, sdf_grad

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        print('getting item', item)
        # randomly sample planning problem for environments
        plan_idx = np.random.randint(low=0, high=self.planning_problem_per_env)

        # Load data from disk
        starts, goals, normalised_sdf, sdf, sdf_grad = self.load_data()
        print('got data')
        # convert to numpy
        start = torch.from_numpy(starts[item, plan_idx]).float()
        goal = torch.from_numpy(goals[item, plan_idx]).float()
        sdf = torch.from_numpy(sdf[item]).float()
        normalised_sdf = torch.from_numpy(normalised_sdf[item]).float().unsqueeze(0)
        sdf_grad = torch.from_numpy(sdf_grad[item]).float()

        return start, goal, normalised_sdf, sdf, sdf_grad

def dataset_builder(dataset_filenames, use_sdf_grad=False, no_load=True, use_U=False):
    datasets = []
    for dataset_filename in dataset_filenames:
        if no_load:
            datasets.append(PlanningProblemDatasetNoLoad(dataset_filename, use_sdf_grad))
        else:
            datasets.append(PlanningProblemDataset(dataset_filename, use_sdf_grad, use_U=use_U))

    return ConcatDataset(datasets)