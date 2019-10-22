from utils import *
from model import *
from solver import *

SEED = 2019
path = '../input/jigsaw-unintended-bias-in-toxicity-classification/'
model_path = '../input/jgs-trained/'
output_path = './'
EMBEDDING_FILES = [
    # '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl',
    # '../input/pickled-paragram-300-vectors-sl999/paragram_300_sl999.pkl',
    '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
]

parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', type=int, default=220)
parser.add_argument('--vocab-size', type=int, default=100000)
parser.add_argument('--n-splits', type=int, default=5,
                    help='splits of n-fold cross validation')
parser.add_argument('--nb-models', type=int, default=5,
                    help='number of models (folds) to ensemble')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--enable-ckpt-ensemble', type=bool, default=1)
parser.add_argument('--ckpt-per-fold', type=bool, default=1)
args = parser.parse_args()


def train_val_split(train_x, train_y, nb_models=args.nb_models):
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    cv_indices = [(tr_idx, val_idx) for tr_idx, val_idx in kf.split(train_x, train_y)]
    if nb_models:
        return cv_indices[:nb_models]
    return cv_indices

def convert_dataframe_to_bool(df):
    def convert_to_bool(df, col_name):
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    bool_df = df.copy()
    for col in [label_column] + identity_columns + aux_columns:
        convert_to_bool(bool_df, col)
    return bool_df

def load_and_preproc():
    train_df = pd.read_csv(path+'train.csv')
    test_df = pd.read_csv(path+'test.csv')
    train_df[identity_columns] = train_df[identity_columns].copy().fillna(0)

    print('cleaning text...')
    t0 = time.time()
    train_df[text_column] = train_df[text_column].apply(clean_text)
    test_df[text_column] = test_df[text_column].apply(clean_text)
    print('cleaning complete in {:.0f} seconds.'.format(time.time()-t0))

    id_cols = train_df[identity_columns].copy().fillna(0).values
    train_tars = train_df[[label_column]+aux_columns].values
    train_tars = np.hstack([train_tars, id_cols]).astype('float32')

    train_df = convert_dataframe_to_bool(train_df)
    df = train_df[[label_column]+identity_columns].copy()
    df[label_column] = df[label_column].astype('uint8')

    return train_df[text_column], train_tars, test_df[text_column], df, test_df['id']



## main()
np.random.seed(SEED)
train_seq, train_tars, x_test, trn_df, test_id = load_and_preproc()

print('tokenizing...')
t0 = time.time()
word_to_idx, idx_to_word = word_idx_map(train_seq.tolist()+x_test.tolist(), args.vocab_size)
train_seq = tokenize(train_seq, word_to_idx, args.maxlen)
x_test = tokenize(x_test, word_to_idx, args.maxlen)
print('tokenizing complete in {:.0f} seconds.'.format(time.time()-t0))


print('loading embeddings...')
t0 = time.time()
embed_mat = np.concatenate(
    [get_embedding(f, word_to_idx) for f in EMBEDDING_FILES], axis=-1)
# embed_mat = np.mean(
#     [get_embedding(f, word_to_idx) for f in EMBEDDING_FILES], axis=0)
print('loading complete in {:.0f} seconds.'.format(time.time()-t0))


# create pseudo labels
test_preds = []
ema_test_preds = []
test_loader, test_original_indices = prepare_loader(x_test, split='test')
models = torch.load(model_path+'models.pt')

for i in range(args.nb_models):
    model = JigsawNet(*embed_mat.shape, 128, embed_mat)
    model.to(device)
    model.load_state_dict(models[f'fold_{i}'])
    test_preds.append(eval_model(model, test_loader, target_only=False)[test_original_indices])

for i in range(args.nb_models):
    ema_model = JigsawNet(*embed_mat.shape, 128, embed_mat)
    ema_model.to(device)
    ema_model.load_state_dict(models[f'ema_fold_{i}'])
    ema_test_preds.append(eval_model(ema_model, test_loader, target_only=False)[test_original_indices])

test_preds = np.mean(test_preds, 0)
ema_test_preds = np.mean(ema_test_preds, 0)
y_test = (0.5*test_preds + 0.5*ema_test_preds)


# training preparation
fold_val_preds = []     # oof predictions for ensemble of folds
ckpt_val_preds = []     # oof predictions for ensemble of ckpts
ema_val_preds = []      # oof predictions for ensemble from ema of weights
oof_tars = []           # for the oof targets
oof_idxs = []           # for the oof sample ids
cv_indices = train_val_split(train_seq, (train_tars[:,0]>=0.5).astype(int))
cv_indices_test = list(KFold(n_splits=args.n_splits, shuffle=True, random_state=SEED).split(x_test))

print()
models = {}
for i, ((trn_idx, val_idx), (trn_idx_test, _)) in enumerate(zip(cv_indices, cv_indices_test)):
    print(f'Fold {i + 1}')
    torch.cuda.empty_cache()

    # train/val split
    x_train, x_val = np.concatenate([train_seq[trn_idx], x_test[trn_idx_test]]), train_seq[val_idx]
    y_train, y_val = np.concatenate([train_tars[trn_idx], y_test[trn_idx_test]]), train_tars[val_idx]
    train_loader = prepare_loader(x_train, y_train, args.batch_size, split='train')
    val_loader, val_original_indices = prepare_loader(x_val, y_val, split='valid')
    val_df = trn_df.iloc[val_idx]
    oof_tars.append(y_val[:,0])
    oof_idxs.append(val_idx)

    # model setup
    seed_torch(SEED+i)
    # model = JigsawNet(*embed_mat.shape, 128, embed_mat)
    # for name, param in model.named_parameters():
    #   if 'emb' in name:
    #       param.requires_grad = False
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                        lr=args.lr)
    ft_lrs = [args.lr*0.08, args.lr, args.lr]
    model, optimizer = model_optimizer_init(128, embed_mat, ft_lrs)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.6)
    scheduler = OneCycleScheduler(optimizer, args.epochs, train_loader, max_lr=ft_lrs)
    model = model.to(device)
    criterion = UnbiasLoss().to(device)

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    ema = WeightEMA(model, sample_rate=50)

    # train
    t0 = time.time()
    if args.enable_ckpt_ensemble: single_val_preds = []
    for e in range(args.epochs):
        print(f'Epoch: {e+1}')

        train_one_cycle(train_loader, model, optimizer, criterion, scheduler, e, ema)
        val_scores, val_unbias_auc = validate(val_loader, model, criterion, val_df.copy(),
                                              e, val_original_indices)

        if args.enable_ckpt_ensemble:
            single_val_preds.append(val_scores)
            models[f'fold_{i}_epk_{e}'] = model.state_dict()
            # torch.save(model.state_dict(), output_path+f'jigsaw_epk_{e}.pth')

        if args.ckpt_per_fold:
            if e == 0:
                best_auc = val_unbias_auc
            if val_unbias_auc > best_auc:
                print('updating best val auc and model...')
                best_auc = val_unbias_auc
                models[f'fold_{i}'] = model.state_dict()
                # torch.save(model.state_dict(), output_path+f'jigsaw_fold_{i}.pth')
                best_val_scores = val_scores

    time_elapsed = time.time() - t0
    print('time = {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

    # inference
    if args.ckpt_per_fold:
        fold_val_preds.append(best_val_scores)

    if args.enable_ckpt_ensemble:
        ckpt_weights = [2**e for e in range(args.epochs)]

        val_scores = np.average(single_val_preds, weights=ckpt_weights, axis=0)
        val_unbias_auc = check_unbias_auc(val_df.copy(), val_scores)
        ckpt_val_preds.append(val_scores)
        print('{"metric": "Ckpt CV Val. Unbiased AUC", "value": %.4f}' % (val_unbias_auc, ))

    ema.set_weights(ema_model)
    # https://stackoverflow.com/questions/53231571/what-does-flatten-parameters-do
    ema_model.rnns.lstm.flatten_parameters()
    ema_model.rnns.gru.flatten_parameters()
    models[f'ema_fold_{i}'] = ema_model.state_dict()

    val_scores = eval_model(ema_model, val_loader, 'val')[val_original_indices]
    val_unbias_auc = check_unbias_auc(val_df.copy(), val_scores)
    ema_val_preds.append(val_scores)
    print('{"metric": "EMA Val. Unbiased AUC", "value": %.4f}' % (val_unbias_auc, ))

    print()


torch.save(models, output_path+'models.pt')

# total set validation auc
oof_tars = (np.concatenate(oof_tars)>=0.5).astype(int)
oof_idxs = np.concatenate(oof_idxs)
oof_df = trn_df.iloc[oof_idxs]

if args.ckpt_per_fold:
    fold_val_preds = np.concatenate(fold_val_preds)
    fold_cv_auc = roc_auc_score(oof_tars, fold_val_preds)
    fold_cv_unbias_auc = check_unbias_auc(oof_df.copy(), fold_val_preds)
    print(f'For whole oof set, fold cv val auc score: {fold_cv_auc}, val unbiased auc: {fold_cv_unbias_auc}')

ema_val_preds = np.concatenate(ema_val_preds)
ema_auc = roc_auc_score(oof_tars, ema_val_preds)
ema_unbias_auc = check_unbias_auc(oof_df.copy(), ema_val_preds)
print(f'For whole oof set, ema val auc score: {ema_auc}, val unbiased auc: {ema_unbias_auc}')

if args.enable_ckpt_ensemble:
    ckpt_val_preds = np.concatenate(ckpt_val_preds)
    mix_val_preds = np.mean((ckpt_val_preds, ema_val_preds), 0)
    ckpt_cv_auc = roc_auc_score(oof_tars, ckpt_val_preds)
    mix_auc = roc_auc_score(oof_tars, mix_val_preds)
    ckpt_cv_unbias_auc = check_unbias_auc(oof_df.copy(), ckpt_val_preds)
    mix_unbias_auc = check_unbias_auc(oof_df.copy(), mix_val_preds)
    print(f'For whole oof set, ckpt cv val auc score: {ckpt_cv_auc}, val unbiased auc: {ckpt_cv_unbias_auc}')
    print(f'For whole oof set, mix val auc score: {mix_auc}, val unbiased auc: {mix_unbias_auc}')


