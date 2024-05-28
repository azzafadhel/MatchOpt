from nets import *
from torch.autograd import functional as autof

# Implement search
def search_surrogate(x, y, oracle, config, _net, n_sol=128, opt_iter=150, init='topk'):
    candidate_idx = torch.topk(y.reshape(-1), k=n_sol)[1]
    x_start = x[candidate_idx]
    freeze(_net)
    if init=='topk':
        x_opt = nn.Parameter(deepcopy(x_start).to(device), requires_grad=True)
    elif init=='rand_idx':
        x_opt = nn.Parameter(deepcopy(x[torch.randperm(x.shape[0])[:n_sol]]).to(device), requires_grad=True)
    elif init=='rand':
        x_opt = nn.Parameter(torch.randn(x_start.shape).to(device), requires_grad=True)
    optimizer = optim.Adam([x_opt],lr=0.001)
    for itr in range(opt_iter):
        optimizer.zero_grad()
        val = torch.sum(-1.0 * _net(x_opt))
        val.backward()
        optimizer.step()
    sol = x_opt.detach().cpu().numpy()
    y_sol = oracle.predict(sol)
    if config['normalize_ys']:
        y_sol = oracle.denormalize_y(y_sol)
    y_sol = (y_sol - y_min) / (y_max - y_min)
    unfreeze(_net)
    return y_sol

def jacobian(net, xb):
    # xb: batchsize x n_dim
    # return jac: batchsize x n_dim
    jac = autof.jacobian(lambda inp: net(inp).flatten(), xb, create_graph=True)
    return torch.einsum('aab->ab', jac)

def loss_fn(net, batch_data, m=5):
    xu, xv, yu, yv = batch_data
    target = yv - yu
    source = 0.
    for i in range(m):
        t = i * 1.0 / m
        source += (1.0 / m) * jacobian(net, xu * (1 - t) + xv * t)
    pred = torch.sum(source * (xv - xu), dim=1)
    return F.mse_loss(pred, target.view(-1))

def create_buckets(y, n_buckets):
    sorted_y, sorted_idx = torch.sort(y, 0)
    sorted_idx = sorted_idx.flatten()
    bucket_size = y.shape[0] // n_buckets
    buckets = []
    for i in range(n_buckets):
        l = bucket_size * i
        u = bucket_size * (i + 1) if (i < n_buckets - 1) else y.shape[0]
        buckets.append(sorted_idx[l:u])
    return buckets

def create_batches(buckets):
    bucket_size = buckets[0].shape[0]
    batches = []
    for i in range(len(buckets)):
        perm = torch.randperm(buckets[i].shape[0])
        batches.append(buckets[i][perm[:bucket_size]])
    return torch.stack(batches).transpose(0, 1)

# Fitting surrogate
def fit_surrogate(run=0, n_epochs=201, batch_size=128, chk_point=20):
    set_seed(run * 1753571)
    init = 'rand' if task in ['tf-bind-8', 'utr'] else 'topk'
    oracle = data[run]['task']
    config = data[run]['config']
    x, y = torch.tensor(oracle.x), torch.tensor(oracle.y)
    buckets = create_buckets(y, batch_size)
    n_data, n_dim = x.shape[0], np.prod(x.shape[1:])
    net = RegressionNet(n_dim, 32).to(device)
    opt = optim.Adam(net.parameters(),lr=0.0001)
    solutions = []
    for epoch in range(n_epochs):
        batches = create_batches(buckets)
        bar = trange(batches.shape[0])
        bar.set_description_str(f'Run {run}, epoch {epoch}')
        if epoch % chk_point == 0:
            y_sol = search_surrogate(x,y,oracle,config,net, init=init)
            solutions.append(y_sol)
            ymax, ymed, yave = np.max(y_sol), np.median(y_sol), np.mean(y_sol)
            bar.set_postfix_str(f'Max={ymax:.4f}, Med={ymed:.4f}, Ave={yave:.4f}')
        for i in bar:
            opt.zero_grad()
            batch_idx = batches[i]
            yb, sort_idx = torch.sort(y[batch_idx].to(device), 0)
            xb = x[batch_idx[sort_idx.cpu().flatten()]].to(device)
            xu, xv = xb[:-1], xb[1:]
            yu, yv = yb[:-1], yb[1:]
            loss = loss_fn(net, (xu, xv, yu, yv)) + F.mse_loss(net(xb), yb)
            loss.backward()
            opt.step()
    return solutions

torch.cuda.set_device(0)
CACHE_FOLDER = '/pycharm/mbo-cache/'
DATA_PATH = CACHE_FOLDER + 'summarized_data.dat'
SAVE_DIR = '/pycharm/gmombo/artifact'
task = 'dkitty'
y_min, y_max = np.min(DATASETS[task]().y), np.max(DATASETS[task]().y)
all_data = load_object(DATA_PATH)
for algo in all_data.keys():
    print(algo, task)
    algo_save_dir = os.path.join(SAVE_DIR, algo)
    if not os.path.exists(algo_save_dir):
        os.mkdir(algo_save_dir)
    data = all_data[algo][task]
    run_solutions = []
    for r in range(4):
        run_solutions.append(fit_surrogate(r))
        torch.save(run_solutions, os.path.join(algo_save_dir, f'{task}-gm.pt'))

