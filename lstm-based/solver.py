from utils import *

output_path = '/output/'


class OneCycleScheduler(object):
    # one-cycle scheduler

    def __init__(self, optimizer, epochs, train_loader, max_lr=3e-3,
                 moms=(.95, .85), div_factor=25, sep_ratio=0.3, final_div=None):

        self.optimizer = optimizer

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
            self.init_lrs = [lr/div_factor for lr in self.max_lrs]
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
            self.init_lrs = [max_lr/div_factor] * len(optimizer.param_groups)

        self.final_div = final_div
        if self.final_div is None: self.final_div = div_factor*1e4
        self.final_lrs = [lr/self.final_div for lr in self.max_lrs]
        self.moms = moms

        self.total_iteration = epochs * len(train_loader)
        self.up_iteration = int(self.total_iteration * sep_ratio)
        self.down_iteration = self.total_iteration - self.up_iteration

        self.curr_iter = 0
        self._assign_lr_mom(self.init_lrs, [moms[0]]*len(optimizer.param_groups))

    def _assign_lr_mom(self, lrs, moms):
        for param_group, lr, mom in zip(self.optimizer.param_groups, lrs, moms):
            param_group['lr'] = lr
            param_group['betas'] = (mom, 0.999)

    def _annealing_cos(self, start, end, pct):
        cos_out = np.cos(np.pi * pct) + 1
        return end + (start-end)/2 * cos_out

    def step(self):
        self.curr_iter += 1

        if self.curr_iter <= self.up_iteration:
            pct = self.curr_iter / self.up_iteration
            curr_lrs = [self._annealing_cos(min_lr, max_lr, pct) \
                            for min_lr, max_lr in zip(self.init_lrs, self.max_lrs)]
            curr_moms = [self._annealing_cos(self.moms[0], self.moms[1], pct) \
                            for _ in range(len(self.optimizer.param_groups))]
        else:
            pct = (self.curr_iter-self.up_iteration) / self.down_iteration
            curr_lrs = [self._annealing_cos(max_lr, final_lr, pct) \
                            for max_lr, final_lr in zip(self.max_lrs, self.final_lrs)]
            curr_moms = [self._annealing_cos(self.moms[1], self.moms[0], pct) \
                            for _ in range(len(self.optimizer.param_groups))]

        self._assign_lr_mom(curr_lrs, curr_moms)


def lr_range_test(train_loader, model, optimizer, criterion, start_lr=1e-7,
                  end_lr=10, num_it=100, stop_div=True):
    epochs = int(np.ceil(num_it/len(train_loader)))
    n_groups = len(optimizer.param_groups)

    if isinstance(start_lr, list) or isinstance(start_lr, tuple):
        if len(start_lr) != n_groups:
            raise ValueError("expected {} max_lr, got {}".format(n_groups, len(start_lr)))
        start_lrs = list(start_lr)
    else:
        start_lrs = [start_lr] * n_groups

    if isinstance(end_lr, list) or isinstance(end_lr, tuple):
        if len(end_lr) != n_groups:
            raise ValueError("expected {} max_lr, got {}".format(n_groups, len(end_lr)))
        end_lrs = list(end_lr)
    else:
        end_lrs = [end_lr] * n_groups

    curr_lrs = start_lrs*1
    for param_group, lr in zip(optimizer.param_groups, curr_lrs):
        param_group['lr'] = lr

    n, lrs_logs, loss_log = 0, [], []

    for e in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.float32)
            scores = model(x)
            loss = criterion(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lrs_logs.append(curr_lrs)
            loss_log.append(loss.item())

            # update best loss
            if n == 0:
                best_loss, n_best = loss.item(), n
            else:
                if loss.item() < best_loss:
                    best_loss, n_best = loss.item(), n

            # update lr per iter with exponential schedule
            n += 1
            curr_lrs = [lr * (end_lr/lr) ** (n/num_it) for lr, end_lr in zip(start_lrs, end_lrs)]
            for param_group, lr in zip(optimizer.param_groups, curr_lrs):
                param_group['lr'] = lr

            # stopping condition
            if n == num_it or (stop_div and (loss.item() > 4*best_loss or torch.isnan(loss))):
                break

    print('minimum loss {}, at lr {}'.format(best_loss, lrs_logs[n_best]))
    return lrs_logs, loss_log


def train(train_loader, model, optimizer, criterion, scheduler, epk, ema=None):
    model.train()
    running_loss = 0.

    for x, y in train_loader:
        x = x.to(device=device, dtype=torch.long)
        y = y.to(device=device, dtype=torch.float32)

        scores = model(x)
        loss = criterion(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)

    N = len(train_loader.dataset)
    train_loss = running_loss / N
    print('{"metric": "Loss", "value": %.4f, "epoch": %d}' % (train_loss, epk+1))
    try:
        train_auc, _l, _s = check_auc(train_loader, model, criterion, num_batches=50)
        print('{"metric": "AUC", "value": %.4f, "epoch": %d}' % (train_auc, epk+1))
    except:
        pass

    scheduler.step()


def train_one_cycle(train_loader, model, optimizer, criterion, scheduler, epk, ema=None):
    model.train()
    running_loss = 0.

    for x, y in train_loader:
        x = x.to(device=device, dtype=torch.long)
        y = y.to(device=device, dtype=torch.float32)

        scores = model(x)
        loss = criterion(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)

        scheduler.step()
        ema.on_batch_end(model)

    N = len(train_loader.dataset)
    train_loss = running_loss / N
    print('{"metric": "Loss", "value": %.4f, "epoch": %d}' % (train_loss, epk+1))
    try:
        train_auc, _l, _s = check_auc(train_loader, model, criterion, num_batches=50)
        print('{"metric": "AUC", "value": %.4f, "epoch": %d}' % (train_auc, epk+1))
    except:
        pass


def validate(val_loader, model, criterion, val_df, epk, val_original_indices):
    val_auc, val_loss, val_scores = check_auc(val_loader, model, criterion)
    val_unbias_auc = check_unbias_auc(val_df, val_scores[val_original_indices])
    print('{"metric": "Val. Loss", "value": %.4f, "epoch": %d}' % (val_loss, epk+1))
    print('{"metric": "Val. AUC", "value": %.4f, "epoch": %d}' % (val_auc, epk+1))
    print('{"metric": "Val. Unbiased AUC", "value": %.4f, "epoch": %d}' % (val_unbias_auc, epk+1))
    return val_scores[val_original_indices], val_unbias_auc


def check_auc(loader, model, criterion, num_batches=None):
    model.eval()
    targets, scores, losses = [], [], []

    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.float32)
            score = model(x)
            l = criterion(score, y)
            targets.append((y[:,0].cpu().numpy()>=0.5).astype(int))
            scores.append(torch.sigmoid(score[:,0]).cpu().numpy())
            losses.append(l.item())
            if num_batches is not None and (t+1) == num_batches:
                break

    targets = np.concatenate(targets)
    scores = np.concatenate(scores)
    auc = roc_auc_score(targets, scores)
    loss = np.mean(losses)

    return auc, loss, scores


def check_unbias_auc(df, scores, print_table=False):
    df[pred_column] = scores
    bias_metrics_df = compute_bias_metrics_for_model(df, identity_columns, pred_column, label_column)
    unbias_auc = get_final_metric(bias_metrics_df, calculate_overall_auc(df, pred_column, label_column))
    if print_table:
        print(bias_metrics_df)
    return unbias_auc


